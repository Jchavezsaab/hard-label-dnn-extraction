#!/usr/bin/env python3
# Steals the network data/unitary_32_32x3_10_float64.keras from hard labels, then checks the result against the truth.
# The attack only ever talks to oracle.label(); each layer is recovered behind the layers we recovered before it.
#
#   1. walk                  find_duals       points where the decision boundary bends, i.e. some neuron is zero -> out/duals/
#   2. per hidden layer L:   cluster          par_cluster      group the duals of layer L by neuron
#                            solve            recover_weights  one [row | bias] per cluster, good to ~1e-7          -> out/layerL_rows.npy
#                            refine           refine           re-fit every row from fresh points, good to ~1e-14  -> out/layerL_refined.npy
#                            signs            signs            which side of each row is the ReLU's positive side  -> out/layerL.npy
#   3. head                  recover_head     the final linear layer                                               -> out/net.npz
#   4. validate              validate.py      compare with the true weights and with the oracle's labels (this reads the truth)
#
# Usage: run.py [WORKERS]

import json
import multiprocessing
import os
import pickle
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 32

WALKS = 8
WALK_PATHS = 45
HIDDEN_LAYERS = 3
WIDTH = 32
KINKS = 64
VOTES_MIN, VOTES_MAX = 20, 1000
HEAD_SEGMENTS, HEAD_HALVINGS = 2000, 50

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "1"           # parallelism comes from WORKERS processes, each single-threaded
sys.path.insert(0, HERE)
from oracle import query_count, label


# ----------------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------------
def layer_file(L):
    return os.path.join(OUT, "layer%d.npy" % L)


def prefix_files(L):
    """The layers below L, as we recovered them."""
    return [layer_file(k) for k in range(L)]


def parallel(function, jobs):
    """function(job) in WORKERS forked processes; function must return (result, queries it made)."""
    with multiprocessing.get_context("fork").Pool(min(WORKERS, len(jobs))) as pool:
        outputs = pool.map(function, jobs, chunksize=1)
    results = [result for result, _ in outputs]
    queries = sum(q for _, q in outputs)
    return results, queries


def step(name):
    print("== " + name, flush=True)
    return time.time()


def record(name, started, queries, **extra):
    """Append a step's query count and wall time to out/stats.json."""
    stats = dict(queries=queries, seconds=round(time.time() - started), **extra)
    print("   %s: %s" % (name, "  ".join("%s=%s" % item for item in sorted(stats.items()))), flush=True)
    path = os.path.join(OUT, "stats.json")
    all_stats = json.load(open(path)) if os.path.exists(path) else {}
    all_stats[name] = stats
    json.dump(all_stats, open(path, "w"), indent=1, sort_keys=True)


def unit_rows(rb):
    """A ReLU unit's scale is arbitrary; every row we store has unit norm."""
    return rb / np.linalg.norm(rb[:, :-1], axis=1, keepdims=True)


# ----------------------------------------------------------------------------------------------------------------------
# 1. dual points
# ----------------------------------------------------------------------------------------------------------------------
def one_walk(seed):
    import torch
    import find_duals

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    before = query_count()
    duals = []
    for _ in range(WALK_PATHS):
        bends = find_duals.find_dual_points()
        # Each bend is a dual point.  The clustering wants it together with a point and the boundary normal on each
        # side of it: (point before, dual, point after, normal before, normal after).  The point / normal after one
        # bend are the point / normal before the next one.
        for (point_before, dual, normal_before), (point_after, _, normal_after) in zip(bends, bends[1:]):
            duals.append((point_before, dual, point_after, normal_before, normal_after))
    return duals, query_count() - before


def walk():
    duals_dir = os.path.join(OUT, "duals")
    if os.path.exists(duals_dir):
        return duals_dir
    started = step("walk: %d x %d walks along the decision boundary" % (WALKS, WALK_PATHS))
    per_walk, queries = parallel(one_walk, range(1, WALKS + 1))
    partial = duals_dir + ".partial"
    os.makedirs(partial, exist_ok=True)
    for seed, duals in enumerate(per_walk, start=1):
        pickle.dump(duals, open(os.path.join(partial, "walk%d.p" % seed), "wb"))
    os.rename(partial, duals_dir)
    record("walk", started, queries, duals=sum(len(d) for d in per_walk))
    return duals_dir


# ----------------------------------------------------------------------------------------------------------------------
# 2. one hidden layer
# ----------------------------------------------------------------------------------------------------------------------
def solve(L, duals_dir):
    """Cluster the duals of layer L and solve every cluster -> rows [w | b] up to sign, ~1e-7 accurate."""
    import par_cluster
    import recover_weights

    rows_file = os.path.join(OUT, "layer%d_rows.npy" % L)
    if os.path.exists(rows_file):
        return np.load(rows_file)

    started = step("layer %d: cluster" % L)
    clusters = par_cluster.cluster_layer(L, duals_dir, prefix_files(L), WORKERS, WIDTH)
    record("layer%d/cluster" % L, started, 0, clusters=len(clusters))         # the duals carry their normals: no queries needed

    started = step("layer %d: solve" % L)
    weights, biases = recover_weights.recover_layer(L, [c["cluster"] for c in clusters], prefix_files(L))
    found = int((np.abs(weights).sum(axis=1) > 0).sum())
    assert found == WIDTH, "layer %d: only %d of %d neurons found; walk more" % (L, found, WIDTH)
    rows = np.concatenate([weights, biases[:, None]], axis=1)
    np.save(rows_file, rows)
    # clusters beyond WIDTH were merged into an existing neuron or dropped by recover_layer (see its printout)
    record("layer%d/solve" % L, started, 0, clusters=len(clusters), neurons=found, dropped_or_merged=len(clusters) - found)
    return rows


def refine_rows(L, rows):
    """Re-fit every row from points placed exactly on its hyperplane -> ~1e-14 accurate."""
    import refine

    refined_file = os.path.join(OUT, "layer%d_refined.npy" % L)
    if os.path.exists(refined_file):
        return np.load(refined_file)
    started = step("layer %d: refine" % L)
    prefix = [np.load(f) for f in prefix_files(L)]
    refined, stats = refine.refine_layer(rows, prefix, K=KINKS, procs=WORKERS)
    kept_old = stats["passes"][-1]["kept_old"]
    assert not kept_old, "layer %d: rows %s could not be refined" % (L, kept_old)
    np.save(refined_file, refined)
    record("layer%d/refine" % L, started, stats["queries"])
    return refined


def one_neuron_sign(job):
    import signs

    L, neuron, refined = job
    np.random.seed(1000 * L + neuron)
    before = query_count()
    # The sign code wants keras-style layers, prefix first and the target layer last: weights (in, out), biases (out,).
    layers = [np.load(f) for f in prefix_files(L)] + [refined]
    weights = [rb[:, :-1].T.copy() for rb in layers]
    biases = [rb[:, -1].copy() for rb in layers]
    with open(os.path.join(OUT, "logs", "signs_layer%d_neuron%d.log" % (L, neuron)), "w") as log:
        sign, _ = signs.recover_neuron_sign(label, weights, biases, L + 1, neuron, VOTES_MIN, VOTES_MAX, log)
    return sign, query_count() - before


def recover_signs(L, refined):
    """Find the sign of every row; the result is the finished layer."""
    if os.path.exists(layer_file(L)):
        return
    started = step("layer %d: signs" % L)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    row_signs, queries = parallel(one_neuron_sign, [(L, j, refined) for j in range(WIDTH)])
    np.save(layer_file(L), refined * np.array(row_signs)[:, None])
    record("layer%d/signs" % L, started, queries)


def hidden_layer(L, duals_dir):
    rows = solve(L, duals_dir)
    refined = refine_rows(L, unit_rows(rows))
    recover_signs(L, refined)


# ----------------------------------------------------------------------------------------------------------------------
# 3. the head
# ----------------------------------------------------------------------------------------------------------------------
def head():
    import recover_head

    net_file = os.path.join(OUT, "net.npz")
    if os.path.exists(net_file):
        return
    started = step("head")
    layers = [np.load(f) for f in prefix_files(HIDDEN_LAYERS)]
    R, info, _ = recover_head.recover(layers, label, query_count, HEAD_SEGMENTS, HEAD_HALVINGS)
    assert info["self_check"], "head: the boundary points did not pin down a single head: %s" % info
    np.savez(net_file, L0=layers[0], L1=layers[1], L2=layers[2], R=R)
    record("head", started, info["queries"])


# ----------------------------------------------------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    duals_dir = walk()
    for L in range(HIDDEN_LAYERS):
        hidden_layer(L, duals_dir)
    head()

    stats = json.load(open(os.path.join(OUT, "stats.json")))
    total = sum(s["queries"] for s in stats.values())
    print("== done: {:,} oracle queries; the network is in {}".format(total, os.path.join(OUT, "net.npz")), flush=True)

    print("== validation (this part reads the true weights)", flush=True)
    return subprocess.call([sys.executable, os.path.join(HERE, "validate.py"), OUT])


if __name__ == "__main__":
    sys.exit(main())
