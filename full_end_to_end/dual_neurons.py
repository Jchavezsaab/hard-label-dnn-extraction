#!/usr/bin/env python3
# Which hidden neuron each dual point found by run.py's walk belongs to, and how many duals every neuron got.
# This reads the true weights (through oracle._net()), so it is a diagnostic, not part of the attack.
#
# A dual is a point where the decision boundary bends because some neuron's pre-activation is zero.  The walk only
# finds it approximately, so each dual is assigned to the neuron whose (locally linear) zero set is closest to it:
# distance |z_j(x)| / |grad_x z_j(x)|, over every neuron of every hidden layer.  Duals farther than --tolerance from
# every neuron are reported as unassigned.
#
# Usage: dual_neurons.py [OUT_DIR] [--model MODEL.keras] [--tolerance T]
#        OUT_DIR defaults to out/<model name>; the duals are read from OUT_DIR/duals/walk*.p.

import argparse
import glob
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oracle  # noqa: E402

parser = argparse.ArgumentParser(description="Count the walk's dual points per hidden neuron.")
parser.add_argument("out", nargs="?", help="run.py's output directory (default out/<model name>)")
parser.add_argument("--model", default=oracle.model_path(), help="the target .keras file (default %(default)s)")
parser.add_argument("--tolerance", type=float, default=1e-4,
                    help="largest distance from a neuron's zero set to count a dual as on it (default %(default)s)")
ARGS = parser.parse_args()


def load_duals(duals_dir):
    """Every dual point of every walk, and the walk it came from."""
    files = sorted(glob.glob(os.path.join(duals_dir, "walk*.p")), key=lambda f: int(os.path.basename(f)[4:-2]))
    assert files, "no walk*.p in %s; run run.py first" % duals_dir
    points, walks = [], []
    for f in files:
        duals = pickle.load(open(f, "rb"))
        points += [dual for _, dual, _, _, _ in duals]
        walks += [int(os.path.basename(f)[4:-2])] * len(duals)
    return np.array(points), np.array(walks)


def distances(x, hidden, activations):
    """[(n, width) distance of every point to every neuron's zero set] per hidden layer."""
    n, input_dim = x.shape
    h = x
    J = np.broadcast_to(np.eye(input_dim), (n, input_dim, input_dim))     # dh/dx, (n, units, input_dim)
    out = []
    for (W, b), (kind, slope) in zip(hidden, activations):
        z = h @ W + b
        Jz = np.einsum("ij,nik->njk", W, J)                                 # dz/dx
        out.append(np.abs(z) / np.maximum(np.linalg.norm(Jz, axis=2), 1e-300))
        if kind == "relu":
            d = (z > 0).astype(np.float64)
        elif kind == "leaky_relu":
            d = np.where(z > 0, 1.0, slope)
        else:
            d = np.ones_like(z)
        h = z * d
        J = Jz * d[:, :, None]
    return out


def main():
    oracle.set_model(os.path.abspath(ARGS.model))
    out_dir = os.path.abspath(ARGS.out or os.path.join(HERE, "out", os.path.splitext(os.path.basename(oracle.model_path()))[0]))
    layers, activations = oracle._net()
    hidden = layers[:-1]
    widths = [W.shape[1] for W, _ in hidden]

    points, walks = load_duals(os.path.join(out_dir, "duals"))
    print("== target %s: hidden widths %s" % (oracle.model_path(), widths), flush=True)
    print("== %d duals from %d walks in %s" % (len(points), len(set(walks)), os.path.join(out_dir, "duals")), flush=True)

    dist = np.concatenate(distances(points, hidden, activations), axis=1)  # (n, total neurons)
    offsets = np.cumsum([0] + widths)
    order = np.argsort(dist, axis=1)
    best = order[:, 0]
    nearest = dist[np.arange(len(points)), best]
    second = dist[np.arange(len(points)), order[:, 1]]
    assigned = nearest <= ARGS.tolerance
    # a dual within tolerance of two neurons sits where two zero sets cross: we cannot tell which bend the walk saw
    ambiguous = assigned & (second <= ARGS.tolerance)

    counts = np.bincount(best[assigned], minlength=offsets[-1])
    for L, w in enumerate(widths):
        c = counts[offsets[L]:offsets[L + 1]]
        print("\n-- layer %d (%d neurons): %d duals, %d neurons with none, min %d / median %d / max %d per neuron" % (
            L, w, c.sum(), (c == 0).sum(), c.min(), np.median(c), c.max()))
        for j in range(w):
            print("   neuron %3d  %6d%s" % (j, c[j], "   <-- none" if c[j] == 0 else ""))

    print("\n== %d duals assigned (%d of them within %g of two neurons), %d unassigned (nearest neuron farther than %g)" % (
        assigned.sum(), ambiguous.sum(), ARGS.tolerance, (~assigned).sum(), ARGS.tolerance))
    if (~assigned).any():
        print("   unassigned duals: nearest-neuron distance min %.3g / median %.3g / max %.3g" % (
            nearest[~assigned].min(), np.median(nearest[~assigned]), nearest[~assigned].max()))
    print("   assigned duals: nearest-neuron distance median %.3g / max %.3g" % (
        np.median(nearest[assigned]), nearest[assigned].max()) if assigned.any() else "")


if __name__ == "__main__":
    main()
