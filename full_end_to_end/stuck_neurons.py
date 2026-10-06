#!/usr/bin/env python3
# Which hidden neurons of the target never change state: always on (pre-activation > 0) or always off (<= 0) on every
# one of many random inputs.  Such neurons put no kink in the decision boundary that a walk is likely to find.
# This reads the true weights (through oracle._net()), so it is a diagnostic, not part of the attack.
#
# Inputs are drawn like the attack draws them: x ~ N(0, scale^2 I).
#
# Usage: stuck_neurons.py [--model MODEL.keras] [--samples N] [--scale S] [--batch B] [--seed SEED]

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oracle  # noqa: E402

parser = argparse.ArgumentParser(description="Find hidden neurons that never change state on random inputs.")
parser.add_argument("--model", default=oracle.model_path(), help="the target .keras file (default %(default)s)")
parser.add_argument("--samples", type=int, default=1_000_000, help="random inputs (default %(default)s)")
parser.add_argument("--scale", type=float, default=1.0, help="standard deviation of the inputs (default %(default)s)")
parser.add_argument("--batch", type=int, default=20_000, help="inputs per batch (default %(default)s)")
parser.add_argument("--seed", type=int, default=0)
ARGS = parser.parse_args()


def main():
    oracle.set_model(os.path.abspath(ARGS.model))
    layers, activations = oracle._net()
    hidden = layers[:-1]
    widths = [W.shape[1] for W, _ in hidden]
    input_dim = layers[0][0].shape[0]
    print("== target %s: %d inputs, hidden widths %s, activations %s" % (
        oracle.model_path(), input_dim, widths, [kind for kind, _ in activations]), flush=True)
    print("== %d samples, x ~ N(0, %g^2 I)" % (ARGS.samples, ARGS.scale), flush=True)

    rng = np.random.default_rng(ARGS.seed)
    on_count = [np.zeros(w, dtype=np.int64) for w in widths]
    lowest = [np.full(w, np.inf) for w in widths]       # smallest pre-activation seen
    highest = [np.full(w, -np.inf) for w in widths]     # largest pre-activation seen

    done = 0
    while done < ARGS.samples:
        n = min(ARGS.batch, ARGS.samples - done)
        h = ARGS.scale * rng.standard_normal((n, input_dim))
        for L, (W, b) in enumerate(hidden):
            z = h @ W + b
            on_count[L] += (z > 0).sum(axis=0)
            lowest[L] = np.minimum(lowest[L], z.min(axis=0))
            highest[L] = np.maximum(highest[L], z.max(axis=0))
            kind, slope = activations[L]
            if kind == "relu":
                h = np.maximum(z, 0)
            elif kind == "leaky_relu":
                h = np.where(z > 0, z, slope * z)
            else:
                h = z
        done += n

    total_stuck = 0
    for L, w in enumerate(widths):
        always_on = np.flatnonzero(on_count[L] == ARGS.samples)
        always_off = np.flatnonzero(on_count[L] == 0)
        total_stuck += len(always_on) + len(always_off)
        print("\n-- layer %d (%d neurons): %d always on, %d always off" % (L, w, len(always_on), len(always_off)))
        for name, neurons in (("always on ", always_on), ("always off", always_off)):
            for j in neurons:
                print("   %s  neuron %3d   pre-activation in [% .4g, % .4g]" % (name, j, lowest[L][j], highest[L][j]))
        # The nearly stuck ones are hard to find too: list the neurons that are on less than 0.1% or more than 99.9% of the time.
        fraction = on_count[L] / ARGS.samples
        rare = np.flatnonzero(((fraction > 0) & (fraction < 1e-3)) | ((fraction < 1) & (fraction > 1 - 1e-3)))
        for j in rare:
            print("   nearly stuck neuron %3d   on %.5f%% of the time" % (j, 100 * fraction[j]))

    print("\n== %d of %d hidden neurons never changed state" % (total_stuck, sum(widths)))


if __name__ == "__main__":
    main()
