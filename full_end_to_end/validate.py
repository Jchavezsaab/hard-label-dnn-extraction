#!/usr/bin/env python3
# VALIDATION ONLY -- this is the one file that reads the true weights.  run.py calls it after the attack is finished.
#
# Compares out/net.npz with the truth.  A stolen ReLU network can only match the truth up to (a) the order of the
# neurons in each hidden layer, (b) a positive scale per neuron, and (c) for the head, a common vector added to all
# ten rows plus one global positive scale (neither changes any argmax).  So every layer is matched to the truth up to
# exactly those freedoms, and finally the labels of the stolen and the true network are compared directly.
#
# Usage: validate.py OUT_DIR
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oracle  # noqa: E402
from oracle import label  # noqa: E402
from recover_head import forward_logits, canonical  # noqa: E402

RANDOM_INPUTS = 100000
BOUNDARY_SEGMENTS = 2000
BOUNDARY_OFFSETS = (1e-3, 1e-6, 1e-9)


def true_layers():
    """[W | b] with one row per neuron, for the four Dense layers.  (Validation is the only place this is allowed.)"""
    return [np.concatenate([W.T, b[:, None]], axis=1) for W, b in oracle._load_weights()]


def match_layer(ours, truth):
    """Match our rows to the true rows of one hidden layer, given how the layer below was matched.

    `truth` must already be expressed in our coordinates of the layer below (see main).  Returns (worst relative error,
    number of rows whose sign is wrong, the true row index of each of our rows, the scale of each of our rows)."""
    n = len(ours)
    cos = (ours / np.linalg.norm(ours, axis=1, keepdims=True)) @ (truth / np.linalg.norm(truth, axis=1, keepdims=True)).T
    which = np.argmax(np.abs(cos), axis=1)
    assert sorted(which) == list(range(n)), "our rows do not match the true neurons one to one"
    scales = np.zeros(n)
    errors = np.zeros(n)
    for i, j in enumerate(which):
        scales[i] = np.linalg.norm(ours[i]) / np.linalg.norm(truth[j]) * np.sign(cos[i, j])
        errors[i] = np.linalg.norm(ours[i] / scales[i] - truth[j]) / np.linalg.norm(truth[j])
    return errors.max(), int((scales < 0).sum()), which, np.abs(scales)


def truth_in_our_coordinates(true_layer, which, scales):
    """Rewrite a true layer's input columns in our coordinates: our unit i is true unit which[i] times scales[i]."""
    W = true_layer[:, :-1][:, which] / scales[None, :]
    return np.concatenate([W, true_layer[:, -1:]], axis=1)


def boundary_points(rng):
    """Points bisected onto the true decision boundary, and a unit vector across it at each of them."""
    a = rng.standard_normal((BOUNDARY_SEGMENTS, 32))
    b = rng.standard_normal((BOUNDARY_SEGMENTS, 32))
    keep = label(a) != label(b)
    lo, hi = a[keep], b[keep]
    label_lo = label(lo)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        same = label(mid) == label_lo
        lo[same] = mid[same]
        hi[~same] = mid[~same]
    across = hi - a[keep]
    across /= np.linalg.norm(across, axis=1, keepdims=True)
    return 0.5 * (lo + hi), across


def main(out):
    net = np.load(os.path.join(out, "net.npz"))
    ours = [net["L0"], net["L1"], net["L2"]]
    truth = true_layers()
    ok = True

    # hidden layers, bottom up: each one is compared in the coordinates our layer below defines
    which, scales = np.arange(32), np.ones(32)
    for L in range(3):
        comparable = truth[L] if L == 0 else truth_in_our_coordinates(truth[L], which, scales)
        error, wrong_signs, which, scales = match_layer(ours[L], comparable)
        print("layer %d: worst relative error %.1e, %d wrong signs" % (L, error, wrong_signs))
        ok = ok and error < 1e-6 and wrong_signs == 0

    # head: compared after removing the freedoms an argmax cannot see
    true_head = canonical(truth_in_our_coordinates(truth[3], which, scales))
    head_error = np.linalg.norm(net["R"] - true_head) / np.linalg.norm(true_head)
    print("head:    relative error %.1e" % head_error)
    ok = ok and head_error < 1e-6

    # and finally the only thing an attacker could check: do the labels agree?
    rng = np.random.default_rng(7)
    x = rng.standard_normal((RANDOM_INPUTS, 32))
    agree = int((np.argmax(forward_logits(ours, net["R"], x), axis=1) == label(x)).sum())
    print("labels:  %d / %d random inputs agree" % (agree, RANDOM_INPUTS))
    ok = ok and agree == RANDOM_INPUTS
    points, across = boundary_points(rng)
    for offset in BOUNDARY_OFFSETS:
        x = np.concatenate([points + offset * across, points - offset * across])
        agree = int((np.argmax(forward_logits(ours, net["R"], x), axis=1) == label(x)).sum())
        print("labels:  %d / %d points %g off the decision boundary agree" % (agree, len(x), offset))
        ok = ok and agree == len(x)

    print("RESULT:  %s" % ("the network was stolen" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out")))
