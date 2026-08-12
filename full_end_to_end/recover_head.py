# Recovers the head (the final Dense(10)) of the extracted network from labels only, and holds the
# numpy forward pass of the assembled network (used by validate/verify_net.py).
#
# recover() gets OUR three hidden layers plus the oracle's label()/query_count() and reads nothing else.
# Method:
#   1. label 2N random inputs, pair them, keep the pairs with different labels
#   2. bisect each kept segment H times -> a point m on the boundary between classes (a, b), where
#      [h3(m), 1] . (r_a - r_b) = 0 with r_k = head row k minus the row of a reference class
#      (a common vector added to every row never changes the argmax, so r_ref = 0 is a free choice)
#   3. the head is the null vector of those equations (one SVD); self-check: exactly one null direction
#   4. the sign of the null vector is fixed by re-predicting the 2N labelled inputs (no queries)
# The result is reported with every column centred over the classes and unit Frobenius norm (canonical()).
# Queries: exactly 2N + kept * H.
import time
import numpy as np

NCLASS = 10
DIM = 32


# ----------------------------------------------------------------------------------------------------------------------------
# the extracted network, as numpy
# ----------------------------------------------------------------------------------------------------------------------------
def forward_hidden(layers, X):
    """Hidden activations of a stack of signed [W | b] blocks (one row per neuron): X (N, 32) -> (N, 32)."""
    H = np.asarray(X, dtype=np.float64)
    for layer in layers:
        W = layer[:, :-1]
        b = layer[:, -1]
        H = np.maximum(H @ W.T + b, 0.0)
    return H


def forward_logits(layers, R, X):
    """Logits of the hidden stack `layers` followed by the head R (10, 33)."""
    H = forward_hidden(layers, X)
    return H @ R[:, :-1].T + R[:, -1]


def load_net(path):
    """net.npz as written by pipeline.py -> (layers, R), the arguments of forward()."""
    z = np.load(path)
    layers = [z["L0"], z["L1"], z["L2"]]
    return layers, z["R"]


def forward(net, X):
    """net = (layers, R) as returned by load_net -> logits (N, 10) in the extracted gauge (argmax-equivalent to the target's)."""
    layers, R = net
    return forward_logits(layers, R, X)


def predict(net, X):
    return np.argmax(forward(net, X), axis=1)


def canonical(R):
    """Gauge fix of a head: centre every column over the classes (common-vector invariance), then unit Frobenius norm (scale)."""
    R = R - R.mean(axis=0, keepdims=True)
    return R / np.linalg.norm(R)


# ----------------------------------------------------------------------------------------------------------------------------
# the attack
# ----------------------------------------------------------------------------------------------------------------------------
def mint_points(label, rng, n_segments, halvings):
    """Steps 1 and 2 of the recipe.

    Returns (X, y, points, pairs, bracket_width): the 2N labelled endpoints (reused later to fix the sign), one boundary point per
    kept segment, the (y_lo, y_hi) class pair of every point, and the width of the widest final bracket."""
    X = rng.standard_normal((2 * n_segments, DIM))
    y = label(X)
    lo = X[:n_segments].copy()
    hi = X[n_segments:].copy()
    y_lo = y[:n_segments].copy()
    y_hi = y[n_segments:].copy()

    keep = y_lo != y_hi
    lo = lo[keep]
    hi = hi[keep]
    y_lo = y_lo[keep]
    y_hi = y_hi[keep]

    for _ in range(halvings):
        mid = 0.5 * (lo + hi)
        y_mid = label(mid)
        same_as_lo = y_mid == y_lo
        moved_hi = ~same_as_lo
        lo[same_as_lo] = mid[same_as_lo]
        hi[moved_hi] = mid[moved_hi]
        y_hi[moved_hi] = y_mid[moved_hi]

    points = 0.5 * (lo + hi)
    pairs = np.stack([y_lo, y_hi], axis=1)
    width = float(np.linalg.norm(hi - lo, axis=1).max())
    return X, y, points, pairs, width


def build_system(H3, pairs, ref):
    """The N x 297 matrix Phi of step 3: row i encodes [h3_i, 1] . (r_a - r_b) with the block of class `ref` left out."""
    N = len(H3)
    feat = np.concatenate([H3, np.ones((N, 1))], axis=1)
    others = [c for c in range(NCLASS) if c != ref]
    column = {c: i for i, c in enumerate(others)}
    W = DIM + 1
    Phi = np.zeros((N, W * (NCLASS - 1)))
    for i, (a, b) in enumerate(pairs):
        if a != ref:
            start = W * column[a]
            Phi[i, start:start + W] += feat[i]
        if b != ref:
            start = W * column[b]
            Phi[i, start:start + W] -= feat[i]
    return Phi, others, column


def solve_head(H3, pairs, ref=None):
    """Step 3: the null vector of the system over all points, with r_ref = 0.

    Returns R (10, 33) with row `ref` all zero, and a dict of label-free diagnostics."""
    classes, counts = np.unique(pairs, return_counts=True)
    if ref is None:
        ref = int(classes[np.argmax(counts)])
    Phi, others, column = build_system(H3, pairs, ref)
    W = DIM + 1

    Phi /= np.linalg.norm(Phi, axis=1, keepdims=True)
    observations = (Phi != 0).sum(axis=0)          # equations touching each of the 297 coefficients
    column_norm = np.linalg.norm(Phi, axis=0)
    column_norm[column_norm == 0] = 1.0
    _, sv, Vt = np.linalg.svd(Phi / column_norm, full_matrices=True)
    # Vt is 297 x 297 even when there are fewer than 297 equations; the missing singular values are exact zeros.
    sv = np.concatenate([sv, np.zeros(Phi.shape[1] - len(sv))])
    u = Vt[-1] / column_norm

    R = np.zeros((NCLASS, W))
    for c in others:
        start = W * column[c]
        R[c] = u[start:start + W]

    residual = np.abs(Phi @ u).max() / np.linalg.norm(u)          # per-equation misfit of the unit null vector
    if sv[-1] > 0:
        gap = float(sv[-2] / sv[-1])
    else:
        gap = float("inf")
    # rank_ok = exactly ONE null direction.  It is False when there are fewer than 297 equations, when a class was never
    # reached, or when some (class, unit) coefficient was never observed (the unit never active at that class's points);
    # `gap` is meaningless in those cases and the stage must refuse.
    rank_ok = bool(sv[-2] / sv[0] > 1e-6)
    info = dict(
        ref=ref,
        s_min=float(sv[-1]),
        s_2nd=float(sv[-2]),
        s_max=float(sv[0]),
        gap=gap,
        rank_ok=rank_ok,
        min_coeff_observations=int(observations.min()),
        resid_max=float(residual),
        classes_reached=sorted(int(c) for c in classes),
        equations=int(len(H3)),
    )
    return R, info


def endpoint_agreement(layers, R, X, y):
    """Fraction of the already-labelled endpoints that the assembled network (layers + R) re-predicts.  Makes no queries."""
    predicted = np.argmax(forward_logits(layers, R, X), axis=1)
    return float((predicted == y).mean())


def recover(layers, label, query_count, n_segments=2000, halvings=30, seed=1):
    """The whole recipe.  Returns (R in canonical gauge, info dict, (boundary points, their class pairs))."""
    t0 = time.time()
    queries_before = query_count()
    rng = np.random.default_rng(seed)

    X, y, points, pairs, width = mint_points(label, rng, n_segments, halvings)
    R, info = solve_head(forward_hidden(layers, points), pairs)

    # Step 4: pick the sign under which the net reproduces the endpoint labels.
    agree_plus = endpoint_agreement(layers, R, X, y)
    agree_minus = endpoint_agreement(layers, -R, X, y)
    if agree_plus >= agree_minus:
        R = canonical(R)
    else:
        R = canonical(-R)
    agreement = max(agree_plus, agree_minus)

    self_check = (
        info["rank_ok"]
        and info["gap"] > 1e4
        and agreement > 0.999
        and len(info["classes_reached"]) == NCLASS
    )
    info.update(
        segments=n_segments,
        kept=int(len(points)),
        halvings=halvings,
        bracket_max=width,
        seed=seed,
        queries=int(query_count() - queries_before),
        wall_seconds=round(time.time() - t0, 3),
        endpoint_agreement=agreement,
        endpoint_agreement_wrong_sign=min(agree_plus, agree_minus),
        self_check=bool(self_check),
    )
    assert info["queries"] == 2 * n_segments + info["kept"] * halvings, info["queries"]
    return R, info, (points, pairs)
