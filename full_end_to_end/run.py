#!/usr/bin/env python3
# Steals a ReLU network (a .keras file of Dense layers, by default data/unitary_32_32x3_10_float64.keras) from hard
# labels, then checks the result against the truth.
# The attack only ever talks to oracle.label(); each layer is recovered behind the layers we recovered before it.
#
#   1. walk                  find_duals       points where the decision boundary bends, i.e. some neuron is zero -> out/duals/
#   2. per hidden layer L:   cluster          par_cluster      group the duals of layer L by neuron                 -> out/layerL_clusters.p
#                            solve            recover_weights  one [row | bias] per cluster, good to ~1e-7          -> out/layerL_rows.npy
#                            refine           refine           re-fit every row from fresh points, good to ~1e-14  -> out/layerL_refined.npy
#                            signs            signs            which side of each row is the ReLU's positive side  -> out/layerL.npy
#   3. head                  recover_head     the final linear layer                                               -> out/net.npz
#   4. validate              validate.py      compare with the true weights and with the oracle's labels (this reads the truth)
#
# The attack knows the target's architecture (input dimension, hidden widths, number of classes; oracle.architecture())
# but none of its weights.  Outputs go to out/<model name>/ unless --out says otherwise.
#
# Usage: run.py [WORKERS] [--model MODEL.keras] [--out OUT_DIR]

import argparse
import itertools
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
sys.path.insert(0, HERE)
import oracle  # noqa: E402

# Parsed at import: worker processes started with "spawn" re-import this file with the same argv.
_parser = argparse.ArgumentParser(description="Steal a ReLU network from hard labels.")
_parser.add_argument("workers", nargs="?", type=int, default=32, help="worker processes (default 32)")
_parser.add_argument("--model", default=oracle.model_path(), help="the target .keras file (default %(default)s)")
_parser.add_argument("--out", help="output directory (default out/<model name>)")
ARGS = _parser.parse_args()

MODEL = os.path.abspath(ARGS.model)
oracle.set_model(MODEL)                 # before any worker starts: they inherit it through the environment
OUT = os.path.abspath(ARGS.out or os.path.join(HERE, "out", os.path.splitext(os.path.basename(MODEL))[0]))
WORKERS = ARGS.workers

ARCHITECTURE = oracle.architecture()
WIDTHS = ARCHITECTURE["widths"]         # neurons in each hidden layer
HIDDEN_LAYERS = len(WIDTHS)
CLASSES = ARCHITECTURE["classes"]

WALKS = 5                               # walks made up front; more are made when a layer's clustering needs them
MAX_WALKS_CLUSTERING = 64               # walks a layer's clustering may use at most (more than WALKS: walks made then)
WALK_PATHS = 45
PATH_POINTS = 20                        # duals a path records at most (ids paths averaged ~52 uncapped)
# Where each path starts, in turn: on the boundary between two random points of different labels (most land between
# the commonest classes), or between a given pair of classes, so that rare classes' boundaries get paths too.
PATH_STARTS = [None] + list(itertools.combinations(range(CLASSES), 2))
KINKS = 64
VOTES_MIN, VOTES_MAX = 20, 250            # sign experiments per neuron: at least / at most (then the majority decides)
SIGN_PASSES_MAX = 6                     # sign passes at most (the first votes, the others re-vote: recover_signs)
STORED_DIST = 1e-6                      # a walk dual within this distance of a row's hyperplane is a dual of that row (signs)
HEAD_SEGMENTS, HEAD_HALVINGS = 2000, 50

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "1"           # parallelism comes from WORKERS processes, each single-threaded
from oracle import query_count, label  # noqa: E402
import activation  # noqa: E402  (after set_model: the model's LEAKY_COEFF)


# ----------------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------------
def layer_file(L):
    return os.path.join(OUT, "layer%d.npy" % L)


def prefix_files(L):
    """The layers below L, as we recovered them."""
    return [layer_file(k) for k in range(L)]


def call_indexed(args):
    i, function, job = args
    return i, function(job)


def parallel(function, jobs, progress=None):
    """function(job) in WORKERS forked processes; function must return (result, queries it made).

    Results come back in the order of jobs; progress(finished so far, job, result), if given, is called in this process
    as each job finishes."""
    jobs = list(jobs)
    outputs = [None] * len(jobs)
    context = multiprocessing.get_context("spawn" if sys.platform == "win32" else "fork")
    with context.Pool(min(WORKERS, len(jobs))) as pool:
        tasks = [(i, function, job) for i, job in enumerate(jobs)]
        for finished, (i, output) in enumerate(pool.imap_unordered(call_indexed, tasks, chunksize=1), start=1):
            outputs[i] = output
            if progress:
                progress(finished, jobs[i], output[0])
    results = [result for result, _ in outputs]
    queries = sum(q for _, q in outputs)
    return results, queries


def step(name):
    print("== " + name, flush=True)
    return time.time()


def record(name, started, queries, **extra):
    """Append a step's query count and wall time to out/stats.json.

    The total reported at the end is the sum over this file, so a step restored from its output files is counted through
    the entry it wrote when it ran.  Every step calls this BEFORE writing its output, so no output exists without its
    entry (a crash in between only means the step runs again and its entry is overwritten)."""
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
def one_path(job):
    """One path along the decision boundary (find_duals) -> its dual points.  A walk is WALK_PATHS of them.

    job = (k, p): path p of walk k.  The paths take the PATH_STARTS in turn, counted across walks, so every walk (also
    one made on demand) carries on the cycle where the one before it stopped."""
    import torch
    import find_duals

    k, p = job
    seed = 1000 * k + p
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    before = query_count()
    start = PATH_STARTS[((k - 1) * WALK_PATHS + p) % len(PATH_STARTS)]
    bends = find_duals.find_dual_points(max_points=PATH_POINTS + 1, classes=start)   # n + 1 bends make n duals
    # Each bend is a dual point.  The clustering wants it together with a point and the boundary normal on each side of
    # it: (point before, dual, point after, normal before, normal after).  The point / normal after one bend are the
    # point / normal before the next one.
    duals = [(point_before, dual, point_after, normal_before, normal_after)
             for (point_before, dual, normal_before), (point_after, _, normal_after) in zip(bends, bends[1:])]
    return duals, query_count() - before


def make_walks(numbers):
    """Walks with the given numbers, their paths spread over the workers -> ({number: duals}, queries)."""
    jobs = [(k, p) for k in numbers for p in range(WALK_PATHS)]
    per_path, queries = parallel(one_path, jobs)
    walks = {k: [] for k in numbers}
    for (k, _), duals in zip(jobs, per_path):
        walks[k].extend(duals)
    return walks, queries


def save_walk(directory, k, duals):
    name = "walk%d.p" % k
    with open(os.path.join(directory, name + ".partial"), "wb") as fh:
        pickle.dump(duals, fh)
    os.replace(os.path.join(directory, name + ".partial"), os.path.join(directory, name))
    return name


def extra_walk(k):
    """Walk k (1-based) made after the first WALKS, when a layer's clustering needs more walks; returns its file.  It
    joins the others in out/duals, so every later step and layer uses it too."""
    started = step("walk %d (more walks for the clustering)" % k)
    walks, queries = make_walks([k])
    record("walk%d" % k, started, queries, duals=len(walks[k]))
    return save_walk(os.path.join(OUT, "duals"), k, walks[k])


def walk():
    duals_dir = os.path.join(OUT, "duals")
    if os.path.exists(duals_dir):
        return duals_dir
    started = step("walk: %d x %d paths along the decision boundary" % (WALKS, WALK_PATHS))
    walks, queries = make_walks(range(1, WALKS + 1))
    partial = duals_dir + ".partial"
    os.makedirs(partial, exist_ok=True)
    for k, duals in walks.items():
        save_walk(partial, k, duals)
    record("walk", started, queries, duals=sum(len(d) for d in walks.values()))
    os.rename(partial, duals_dir)
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

    clusters_file = os.path.join(OUT, "layer%d_clusters.p" % L)
    state = par_cluster.load_clusters(clusters_file) if os.path.exists(clusters_file) else None
    if state is not None and state["complete"]:
        clusters = state["clusters"]
        print("== layer %d: cluster (%d clusters from %s)" % (L, len(clusters), clusters_file), flush=True)
    else:
        # The clustering saves its state after every cluster (marked incomplete): an interrupted run goes on from it.
        started = step("layer %d: cluster" % L)
        if state is not None:
            print("   going on from %d clusters (%d rejected) in %s" % (len(state["clusters"]), len(state["rejected"]),
                  clusters_file), flush=True)
        clusters = par_cluster.cluster_layer(L, duals_dir, prefix_files(L), WORKERS, WIDTHS[L], checkpoint=clusters_file,
                                             max_walks=MAX_WALKS_CLUSTERING, new_walk=extra_walk,
                                             previous=state)
        record("layer%d/cluster" % L, started, 0, clusters=len(clusters))     # the duals carry their normals: no queries needed
        par_cluster.mark_complete(clusters_file)

    started = step("layer %d: solve" % L)
    weights, biases = recover_weights.recover_layer(L, [c["cluster"] for c in clusters], prefix_files(L), WIDTHS[L])
    solved = np.abs(weights).sum(axis=1) > 0
    found = int(solved.sum())
    rows = np.concatenate([weights, biases[:, None]], axis=1)
    if found < WIDTHS[L]:
        # The neurons never found are taken to be dead (always off: they output 0 everywhere, so leaving them out
        # changes nothing): they are amputated, the layer has fewer rows and the next layer fewer inputs.
        rows = rows[solved]
        print("   layer %d: only %d of %d neurons found; the %d missing taken to be dead and left out (amputated)"
              % (L, found, WIDTHS[L], WIDTHS[L] - found), flush=True)
    # clusters beyond the layer's width were merged into an existing neuron or dropped by recover_layer (see its printout)
    record("layer%d/solve" % L, started, 0, clusters=len(clusters), neurons=found, dropped_or_merged=len(clusters) - found,
           amputated=WIDTHS[L] - found)
    np.save(rows_file, rows)
    return rows


def refine_rows(L, rows):
    """Re-fit every row from points placed exactly on its hyperplane -> ~1e-14 accurate."""
    import refine

    refined_file = os.path.join(OUT, "layer%d_refined.npy" % L)
    if os.path.exists(refined_file):
        return np.load(refined_file)
    started = step("layer %d: refine" % L)
    prefix = [np.load(f) for f in prefix_files(L)]
    kinks = max(KINKS, 2 * (rows.shape[1] - 1))
    refined, stats = refine.refine_layer(rows, prefix, K=kinks, procs=WORKERS)
    kept_old = stats["passes"][-1]["kept_old"]
    assert not kept_old, "layer %d: rows %s could not be refined" % (L, kept_old)
    record("layer%d/refine" % L, started, stats["queries"], kinks=kinks)
    np.save(refined_file, refined)
    return refined


def stored_duals(L, refined):
    """For every row of layer L, the walk's dual points that lie on its hyperplane (within STORED_DIST in input space),
    at most VOTES_MAX each: they carry the decision boundary's normals on both sides, so they are sign experiments
    without queries (signs.analyzeStoredDual).  Duals on a hyperplane of the layers below are left out."""
    import par_cluster

    duals = []
    for f in par_cluster.walk_files(os.path.join(OUT, "duals")):
        duals.extend(pickle.load(open(os.path.join(OUT, "duals", f), "rb")))
    prefix = [np.load(f) for f in prefix_files(L)]
    h = np.array([d[1] for d in duals], dtype=np.float64)
    usable = np.ones(len(duals), bool)
    for rb in prefix:
        z = h @ rb[:, :-1].T + rb[:, -1]
        usable &= np.all(np.abs(z) >= 1e-5, axis=1)     # as par_cluster.reject_on_earlier_layer
        h = activation.relu(z)
    z = h @ refined[:, :-1].T + refined[:, -1]            # (duals, rows)
    stored = []
    for j in range(len(refined)):
        near = np.flatnonzero(usable & (np.abs(z[:, j]) < 1e-3))
        mine = []
        for i in near:
            # input-space distance to row j's hyperplane, in the dual's own on/off pattern
            x = np.asarray(duals[i][1], dtype=np.float64)
            A = np.eye(len(x))
            for rb in prefix:
                pre = rb[:, :-1] @ x + rb[:, -1]
                A = (rb[:, :-1] @ A) * activation.slope(pre)[:, None]
                x = activation.relu(pre)
            if abs(z[i, j]) < STORED_DIST * np.linalg.norm(A.T @ refined[j, :-1]):
                mine.append(duals[i])
            if len(mine) >= VOTES_MAX:
                break
        stored.append(mine)
    return stored


def one_neuron_sign(job):
    import signs

    L, neuron, refined, stored = job
    np.random.seed(1000 * L + neuron)
    before = query_count()
    # The sign code wants keras-style layers, prefix first and the target layer last: weights (in, out), biases (out,).
    layers = [np.load(f) for f in prefix_files(L)] + [refined]
    weights = [rb[:, :-1].T.copy() for rb in layers]
    biases = [rb[:, -1].copy() for rb in layers]
    with open(os.path.join(OUT, "logs", "signs_layer%d_neuron%d.log" % (L, neuron)), "w") as log:
        sign, confident, votes = signs.recover_neuron_sign(label, weights, biases, L + 1, neuron, VOTES_MIN, VOTES_MAX, log,
                                                           stored=stored)
    experiments = [{key: r[key] for key in ('M', 'z', 'mON', 'mOFF', 'right')} for r in votes]     # for the later passes
    from_walk = sum(r['stored'] for r in votes)
    return (sign, confident, len(votes), experiments, from_walk), query_count() - before


def recover_signs(L, refined):
    """Find the sign of every row; the result is the finished layer.

    The first pass votes over every experiment: first those at the walk's own dual points on the row's hyperplane, whose
    normals the walk already measured (no queries, stored_duals), then new ones only if those were not enough.  Later
    passes make no queries: with the previous pass's signs telling which target-layer neurons are active at each dual
    point, they compute every experiment's probe again so that it leaves only those alone, and vote again with the
    measured normals (signs.reassess), until no sign changes.  On a tie, the previous sign stands."""
    import signs

    if os.path.exists(layer_file(L)):
        return
    started = step("layer %d: signs" % L)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)

    def progress(finished, job, result):
        _, neuron, _, _ = job
        sign, confident, votes, _, from_walk = result
        print("sign %d of %d: neuron %d %s after %d votes (%d from the walk)%s" % (finished, len(refined), neuron,
              "kept" if sign > 0 else "flipped", votes, from_walk, "" if confident else " (majority without confidence)"),
              flush=True)

    stored = stored_duals(L, refined)
    print("   layer %d: walk duals on each row's hyperplane: %s" % (L, [len(s) for s in stored]), flush=True)
    outcomes, queries = parallel(one_neuron_sign, [(L, j, refined, stored[j]) for j in range(len(refined))], progress)
    first = np.array([sign for sign, _, _, _, _ in outcomes])
    row_signs = first.copy()
    confident = [c for _, c, _, _, _ in outcomes]
    # Later passes re-vote every experiment with the previous pass's signs (no queries), until no sign changes: a wrong
    # estimate of one neuron misleads the probes of the others, so fixing one can fix the next.
    for sign_pass in range(2, SIGN_PASSES_MAX + 1):
        estimates = row_signs.copy()
        for j, (_, _, votes, experiments, _) in enumerate(outcomes):
            sign, sure, _ = signs.reassess(experiments, j, estimates, VOTES_MIN)
            if sign is not None:
                row_signs[j] = sign
                confident[j] = sure
        flips = np.flatnonzero(row_signs != estimates).tolist()
        print("   pass %d: changed the signs of neurons %s" % (sign_pass, flips), flush=True)
        if not flips:
            break
    changed = np.flatnonzero(row_signs != first).tolist()
    print("   layer %d: the later passes changed the signs of neurons %s" % (L, changed), flush=True)
    unsure = [j for j, sure in enumerate(confident) if not sure]
    if unsure:
        # out of dual points / usable experiments before the vote was confident: the majority is used anyway
        print("   layer %d: signs of neurons %s are a majority without confidence (see logs/)" % (L, unsure), flush=True)
    record("layer%d/signs" % L, started, queries, unsure=unsure, second_pass_changed=changed)
    np.save(layer_file(L), refined * row_signs[:, None])


def report_error(L, stage, rows):
    """Diagnostics only: compare rows of layer L with the true weights (validate.py reads them; no oracle queries, and
    nothing here feeds back into the attack)."""
    import validate

    prefix = [np.load(f) for f in prefix_files(L)]
    relative, absolute, wrong_signs = validate.layer_error(L, rows, prefix)
    signs = ", %d of %d signs correct" % (len(rows) - wrong_signs, len(rows)) if stage == "signs" else ""
    if len(rows) < WIDTHS[L]:
        signs += " (%d amputated neurons left out)" % (WIDTHS[L] - len(rows))
    print("   layer %d after %s vs. truth: max relative row error %.1e, max parameter error %.1e%s"
          % (L, stage, relative, absolute, signs), flush=True)


def hidden_layer(L, duals_dir):
    rows = solve(L, duals_dir)
    report_error(L, "solve", rows)
    refined = refine_rows(L, unit_rows(rows))
    report_error(L, "refine", refined)
    recover_signs(L, refined)
    report_error(L, "signs", np.load(layer_file(L)))


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
    # More unknowns (wider top layer, more classes) need more boundary points: double them until the head is pinned down.
    queries = 0
    for segments in HEAD_SEGMENTS * 2 ** np.arange(4):
        R, info, _ = recover_head.recover(layers, label, query_count, CLASSES, int(segments), HEAD_HALVINGS)
        queries += info["queries"]
        if info["self_check"]:
            break
        print("   %d segments did not pin down a single head; doubling" % segments, flush=True)
    assert info["self_check"], "head: the boundary points did not pin down a single head: %s" % info
    record("head", started, queries, segments=int(segments))
    np.savez(net_file, R=R, **{"L%d" % L: layer for L, layer in enumerate(layers)})


# ----------------------------------------------------------------------------------------------------------------------
def check_out_dir():
    """OUT caches every step; refuse to mix in the results of another model."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "model.json")
    this = dict(model=MODEL, architecture=ARCHITECTURE)
    if os.path.exists(path):
        previous = json.load(open(path))
        assert previous == this, "%s holds the results of %s; pass another --out" % (OUT, previous["model"])
    else:
        json.dump(this, open(path, "w"), indent=1)


def main():
    assert all(kind in ("relu", "leaky_relu") for kind in ARCHITECTURE["activations"]),         "the attack targets (leaky) ReLU networks; %s has hidden activations %s" % (MODEL, ARCHITECTURE["activations"])
    print("== target %s: %d inputs, hidden widths %s, %d classes, %s -> %s" % (
        MODEL, ARCHITECTURE["input_dim"], WIDTHS, CLASSES,
        "LeakyReLU(%g)" % activation.LEAKY_COEFF if activation.LEAKY_COEFF else "ReLU", OUT), flush=True)
    check_out_dir()
    duals_dir = walk()
    for L in range(HIDDEN_LAYERS):
        hidden_layer(L, duals_dir)
    head()

    stats = json.load(open(os.path.join(OUT, "stats.json")))
    total = sum(s["queries"] for s in stats.values())
    print("== done: {:,} oracle queries; the network is in {}".format(total, os.path.join(OUT, "net.npz")), flush=True)

    print("== validation (this part reads the true weights)", flush=True)
    return subprocess.call([sys.executable, os.path.join(HERE, "validate.py"), OUT, MODEL])


if __name__ == "__main__":
    sys.exit(main())
