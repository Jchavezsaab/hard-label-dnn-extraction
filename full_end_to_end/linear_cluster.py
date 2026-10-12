# Groups the dual points of one layer by neuron, as exact linear algebra (an alternative to par_cluster).
#
# At a dual point x0 the walk measured the decision boundary's normals nL and nR on either side of the bend.  The neuron
# that bends there has its hyperplane through x0, containing the (n-2)-dimensional subspace N orthogonal to both
# normals.  Behind the prefix (our recovered layers below this one; h = prefix output, A = its Jacobian at x0, taken in
# x0's on/off pattern) the neuron's row c = [w | b] therefore satisfies
#
#       (A N)^T w = 0          the hyperplane contains N
#       h . w + b = 0          it passes through x0
#
# which is R c = 0 for a small matrix R per dual (computed once; each row weighted by how firmly it constrains c).  A
# neuron's row satisfies the R of every one of its duals, so:
#
#   1. for a seed dual, every other dual j gives a candidate row: the solution of [R_seed; R_j] c = 0, when it is
#      unique; all partners of the seed's own neuron give the same candidate;
#   2. a candidate's support is every dual with ||R c|| ~ 0 (one matrix product over all duals) at which the row is
#      engaged, i.e. has weight on units that are on there (any row satisfies R c = 0 trivially at a dual where all the
#      units it weighs are off): a real neuron is supported by its duals, a coincidental fit by next to none;
#   3. the best supported candidate is refitted on its supporters and accepted, its supporters are explained, and the
#      next unexplained dual is the next seed;
#   4. at the end: rows that agree on the weights both know are one neuron (merged), and deeper neurons in disguise are
#      dropped (a deeper neuron behaves like a neuron of this layer within one on/off region of this layer, so its row
#      is a combination of the rows that keep one sign over its supporters).
#
# A prefix unit that is off at every supporter leaves its weight undetermined: it comes back as exactly 0 (refine fills
# it in).  cluster_layer() returns the same {"cluster": [duals], "weight": w, "bias": b} as par_cluster.cluster_layer.
import os
import pickle

import numpy as np

PAIR_TOL = 1e-6        # a pair's common solution: smallest singular value at most this ...
PAIR_GAP = 1e-3        # ... and the next one at least this (the solution is unique)
SUPPORT_TOL = 1e-6     # a dual supports a row if ||R c|| is below this (c of unit norm)
MIN_SUPPORT = 3        # duals a row needs to be accepted
PARTNERS = 3000        # a seed is paired with at most this many unexplained duals (at random)
IDLE_SEEDS = 300       # once there are `width` rows, seeding stops after this many seeds in a row add none
GROUP_COS = 1 - 1e-6   # candidates this parallel are one
RANK_TOL = 1e-9        # relative singular value below which a direction does not count (rank of A N, free columns)
ENGAGED_MIN = 1e-3     # a dual only supports a row with at least this share of its weight norm on units on at the dual
DEEPER_TOL = 1e-6      # a row this close to the span of the rows that keep one sign over its supporters is deeper


def load_duals(duals_dir):
    duals = []
    for f in sorted(os.listdir(duals_dir)):
        duals.extend(pickle.load(open(os.path.join(duals_dir, f), "rb")))
    return duals


def prefix_at(prefix, x):
    """Prefix output h, its Jacobian A (d x n) in x's on/off pattern, and the smallest |pre-activation| / ||gradient||
    over every prefix unit (how far x is from the nearest prefix hyperplane, inf for layer 0)."""
    h = np.array(x, dtype=np.float64)
    A = np.eye(len(h))
    margin = np.inf
    for rb in prefix:
        W, b = rb[:, :-1], rb[:, -1]
        z = W @ h + b
        G = W @ A
        margin = min(margin, float(np.min(np.abs(z) / np.maximum(np.linalg.norm(G, axis=1), 1e-300))))
        on = z > 0
        h = np.where(on, z, 0.0)
        A = G * on[:, None]
    return h, A, margin


def constraints(dual, prefix):
    """R (width d + 1, rows weighted by their singular values) with R [w | b] = 0 for the row of the neuron that bends at
    this dual, or None if the dual is unusable (on a prefix hyperplane, or its two normals are parallel)."""
    _, x0, _, nL, nR = dual
    x0 = np.asarray(x0, dtype=np.float64)
    h, A, margin = prefix_at(prefix, x0)
    if margin < 1e-5 * max(1.0, float(np.max(np.abs(x0)))):
        return None                                   # the dual of a prefix neuron, or too close to one to tell
    U, S, Vt = np.linalg.svd(np.vstack([nL, nR]))
    if S[1] < 1e-3 * S[0]:
        return None                                   # no bend: the two sides' normals agree
    N = Vt[2:].T                                      # n x (n-2): the directions the hyperplane contains
    G = (A @ N).T                                     # (n-2) x d: G w = 0
    _, Sg, Vg = np.linalg.svd(G, full_matrices=False)
    keep = Sg > RANK_TOL * max(Sg[0], 1e-300)
    # Each direction keeps its weight (singular value, relative to the largest): the weak ones are mostly the error of
    # the measured normals, and an orthonormal basis would blow that error up.
    Q = Vg[keep] * (Sg[keep] / Sg[0])[:, None]
    d = len(h)
    rows = np.zeros((len(Q) + 1, d + 1))
    rows[:len(Q), :d] = Q
    a = np.r_[h, 1.0]
    rows[len(Q)] = a / np.linalg.norm(a)
    return rows


class Duals:
    """Every usable dual's R, padded to one height for batched work, and stacked for support counts."""

    def __init__(self, duals, prefix):
        self.duals, Rs, on = [], [], []
        for dual in duals:
            R = constraints(dual, prefix)
            if R is not None:
                self.duals.append(dual)
                Rs.append(R)
                on.append(prefix_at(prefix, dual[1])[0] > 0)
        self.on = np.array(on, dtype=np.float64)       # units of the prefix that are on at each dual
        self.width = Rs[0].shape[1]
        self.height = max(len(R) for R in Rs)
        self.padded = np.zeros((len(Rs), self.height, self.width))
        for i, R in enumerate(Rs):
            self.padded[i, :len(R)] = R
        self.R = Rs

    def __len__(self):
        return len(self.R)

    def residuals(self, c):
        """||R_m c|| for every dual m (c of unit norm)."""
        return np.linalg.norm(self.padded @ c, axis=1)

    def engaged(self, c):
        """Whether the row has a real share of its weight on units that are on, at every dual."""
        w = c[:-1]
        return np.sqrt(self.on @ (w * w)) >= ENGAGED_MIN * np.linalg.norm(w)

    def support(self, c):
        return np.flatnonzero((self.residuals(c) <= SUPPORT_TOL) & self.engaged(c))


def null_basis(R):
    _, S, Vt = np.linalg.svd(R)
    rank = int(np.sum(S > RANK_TOL * S[0]))
    return Vt[rank:].T


def candidates(D, seed, partners):
    """Unit candidate rows from the seed paired with each partner (unique common solutions only)."""
    B = null_basis(D.R[seed])                         # (d+1) x k
    k = B.shape[1]
    if k == 1:
        return B.T                                    # the seed alone pins its row down
    P = D.padded[partners] @ B                        # (m, height, k)
    _, S, Vt = np.linalg.svd(P, full_matrices=False)
    ok = (S[:, -1] <= PAIR_TOL) & (S[:, -2] >= PAIR_GAP)
    return Vt[ok, -1, :] @ B.T


def group(cands):
    """Representatives of the groups of (anti)parallel candidates, largest groups first."""
    left = list(range(len(cands)))
    groups = []
    while left:
        ref = cands[left[0]]
        cos = cands[left] @ ref
        same = np.abs(cos) >= GROUP_COS
        members = np.array(left)[same]
        mean = (cands[members] * np.sign(cos[same])[:, None]).sum(axis=0)
        groups.append((len(members), mean / np.linalg.norm(mean)))
        left = [i for i, s in zip(left, same) if not s]
    groups.sort(key=lambda g: -g[0])
    return [c for _, c in groups]


def refit(D, supporters):
    """The row that best satisfies every supporter's R; weights on columns no supporter constrains come back as 0.
    Returns (c, smallest singular value, next one) or None."""
    stacked = np.vstack([D.R[m] for m in supporters])
    free = np.any(np.abs(stacked) > 0, axis=0)
    free[-1] = True
    _, S, Vt = np.linalg.svd(stacked[:, free], full_matrices=False)
    if len(S) < 2:
        return None
    c = np.zeros(D.width)
    c[free] = Vt[-1]
    return c / np.linalg.norm(c), S[-1], S[-2]


def find_rows(D, width, rng, log):
    """Steps 1-3: every accepted row with its supporters."""
    explained = np.zeros(len(D), bool)
    found = []
    idle = 0
    for seed in rng.permutation(len(D)):
        if len(found) >= width and idle >= IDLE_SEEDS:
            break
        if explained[seed]:
            continue
        idle += 1
        partners = np.flatnonzero(~explained)
        partners = partners[partners != seed]
        if len(partners) > PARTNERS:
            partners = rng.choice(partners, PARTNERS, replace=False)
        cands = candidates(D, seed, partners)
        if not len(cands):
            continue
        best = None
        for c in group(cands)[:3]:
            support = D.support(c)
            if seed in support and (best is None or len(support) > len(best[1])):
                best = (c, support)
        if best is None or len(best[1]) < MIN_SUPPORT:
            continue
        c, support = best
        for _ in range(2):                            # refit on the supporters, which may pick up a few more
            fit = refit(D, support)
            if fit is None:
                break
            c = fit[0]
            new = D.support(c)
            if len(new) < MIN_SUPPORT or np.array_equal(new, support):
                break
            support = new
        explained[support] = True
        found.append({"row": c, "support": support})
        idle = 0
        log("   row %d: %d duals (seed %d), %d unknown weights" % (len(found), len(support), seed,
            int(np.sum(c[:-1] == 0))))
    return found


def merge_duplicates(D, found):
    """Rows that agree on every weight both know are one neuron: their supporters are pooled and refitted."""
    merged = True
    while merged:
        merged = False
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                a, b = found[i]["row"], found[j]["row"]
                known = np.r_[(a[:-1] != 0) & (b[:-1] != 0), True]
                if abs(a[known] @ b[known]) >= GROUP_COS * np.linalg.norm(a[known]) * np.linalg.norm(b[known]):
                    support = np.union1d(found[i]["support"], found[j]["support"])
                    fit = refit(D, support)
                    if fit is not None:
                        found[i] = {"row": fit[0], "support": support}
                        del found[j]
                        merged = True
                        break
            if merged:
                break
    return found


def deeper(D, found):
    """Indices of rows that are deeper neurons in disguise: the row lies in the span of the other rows that keep one
    sign over all of its supporters (within one on/off region of this layer, a deeper neuron's zero set is such a
    combination; a neuron of this layer has supporters on both sides of most other hyperplanes)."""
    out = []
    for i, f in enumerate(found):
        H = np.array([np.r_[prefix_h, 1.0] for prefix_h in f["h"]])
        steady = []
        for j, g in enumerate(found):
            if j != i:
                s = H @ g["row"]
                if np.all(s > 0) or np.all(s < 0):
                    steady.append(g["row"])
        if not steady:
            continue
        basis = np.linalg.svd(np.array(steady), full_matrices=False)[2]
        basis = basis[: np.linalg.matrix_rank(np.array(steady), 1e-9)]
        rest = f["row"] - basis.T @ (basis @ f["row"])
        if np.linalg.norm(rest) <= DEEPER_TOL:
            out.append(i)
    return out


def cluster_layer(layer, duals_dir, prefix_files, width, seed=0, log=print):
    """The clusters of layer `layer` (dicts like par_cluster's, plus "support"), at most `width` of them; fewer if some
    neurons have no usable duals.  prefix_files: our recovered layers below this one."""
    prefix = [np.load(f) for f in prefix_files]
    D = Duals(load_duals(duals_dir), prefix)
    log("   %d usable duals" % len(D))
    found = find_rows(D, width, np.random.default_rng(seed), log)
    found = merge_duplicates(D, found)
    for f in found:
        f["h"] = [prefix_at(prefix, D.duals[m][1])[0] for m in f["support"]]
    dropped = 0
    while True:
        bad = deeper(D, found)
        if not bad:
            break
        # drop the least supported impostor first: the others' test changes once it is gone
        worst = min(bad, key=lambda i: len(found[i]["support"]))
        found.pop(worst)
        dropped += 1
    found.sort(key=lambda f: -len(f["support"]))
    if len(found) > width:
        log("   %d rows left for %d neurons: keeping the %d best supported" % (len(found), width, width))
        found = found[:width]
    log("   %d rows (%d dropped as deeper neurons)" % (len(found), dropped))
    return [{"cluster": [D.duals[m] for m in f["support"]], "weight": f["row"][:-1], "bias": f["row"][-1],
             "support": f["support"]} for f in found]
