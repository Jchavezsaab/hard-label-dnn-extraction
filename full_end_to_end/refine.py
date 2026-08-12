#!/usr/bin/env python3
import json
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import numpy as np  # noqa: E402

from oracle import label, query_count  # noqa: E402  -- the only access to the target

EPS = np.finfo(np.float64).eps
BAD_FIT = 1e-10
BAD_RANK = 1e-4
COLLINEAR = 256
MIN_TANGENT = 0.1
NO_BEND = 1e-6
SECOND_KINK_STATIONS = 8


def extra_points(n_unknowns):
    """A fit is only attempted / kept with at least this many kinks."""
    return n_unknowns + 2


def load_unit(path):
    """A [row | bias] file with every row scaled to unit norm (the scale of a ReLU unit is free)."""
    rb = np.array(np.load(path), dtype=np.float64)
    return rb / np.linalg.norm(rb[:, :-1], axis=1, keepdims=True)


def prefix_forward(prefix, x):
    """Activations h at the top of the prefix (layer 0: x itself) and the local affine map h = A x + c around x.

    Also returns the distance from x to the nearest prefix hyperplane, i.e. how far H_j is flat around x (inf for layer 0)."""
    A = np.eye(len(x))
    c = np.zeros(len(x))
    h = np.array(x, dtype=np.float64)
    margin = np.inf
    for rb in prefix:
        W = rb[:, :-1]
        b = rb[:, -1]
        z = W @ h + b
        G = W @ A                                       # input-space gradient of every unit's pre-activation
        grad_norm = np.maximum(np.linalg.norm(G, axis=1), 1e-300)
        margin = min(margin, float(np.min(np.abs(z) / grad_norm)))
        on = (z > 0).astype(np.float64)
        A = on[:, None] * G
        c = on * (W @ c + b)
        h = on * z
    return h, A, c, margin


def pattern(prefix, x):
    """The on/off pattern of every prefix unit at x, as one boolean vector."""
    x = np.array(x, dtype=np.float64)
    pat = []
    for rb in prefix:
        z = rb[:, :-1] @ x + rb[:, -1]
        pat.append(z > 0)
        x = np.maximum(z, 0)
    if not pat:
        return np.zeros(0, bool)
    return np.concatenate(pat)


def project(prefix, rows, j, x, iters=30):
    """Newton-project x onto H_j of neuron j (re-deriving the prefix pattern at every step).

    Returns (x, nu, piece, local) or None if it did not converge: nu = unit input-space normal of H_j at x, piece = distance to
    the nearest prefix hyperplane, local = min(piece, distance to the other neurons' hyperplanes of this layer)."""
    w = rows[j, :-1]
    b = rows[j, -1]
    converged = False
    for _ in range(iters):
        h, A, c, piece = prefix_forward(prefix, x)
        g = w @ h + b
        nu = A.T @ w
        nn = nu @ nu
        if nn < 1e-30:
            return None
        if abs(g) <= 1e-12 * np.sqrt(nn) * (1 + np.linalg.norm(x)):
            converged = True
            break
        x = x - g / nn * nu
    if not converged:
        return None
    nu = nu / np.sqrt(nn)
    Z = rows[:, :-1] @ h + rows[:, -1]
    G = rows[:, :-1] @ A
    others = np.abs(Z) / np.maximum(np.linalg.norm(G, axis=1), 1e-300)
    others[j] = np.inf
    local = min(piece, float(others.min()))
    return x, nu, piece, local


def ray_to_border(prefix, x, d):
    """Distance along d from x to the nearest prefix hyperplane (exact: the prefix is affine in between); inf at layer 0."""
    A = np.eye(len(x))
    h = np.array(x, dtype=np.float64)
    t = np.inf
    for rb in prefix:
        W = rb[:, :-1]
        b = rb[:, -1]
        z = W @ h + b
        G = W @ A
        gd = G @ d
        with np.errstate(divide="ignore", invalid="ignore"):
            ti = -z / gd
        ahead = ti[(ti > 0) & np.isfinite(ti)]
        if len(ahead):
            t = min(t, float(ahead.min()))
        on = (z > 0).astype(np.float64)
        A = on[:, None] * G
        h = on * z
    return t


def bisect(p, u, ta, tb, la):
    """label(p + ta u) == la != label(p + tb u): shrink the bracket to ~2 ulp of the coordinates, return its midpoint."""
    limit = 4 * EPS * max(1.0, float(np.max(np.abs(p))))
    while abs(tb - ta) > limit:
        tm = 0.5 * (ta + tb)
        if tm == ta or tm == tb:
            break
        if label(p + tm * u) == la:
            ta = tm
        else:
            tb = tm
    return 0.5 * (ta + tb)


def walk_to_boundary(prefix, rows, j, x, nu, d, la, reach, max_pieces=64):
    left = reach
    for _ in range(max_pieces):
        d = d - (d @ nu) * nu
        n = np.linalg.norm(d)
        if n < MIN_TANGENT:
            return None
        d = d / n
        t_border = ray_to_border(prefix, x, d)
        t_end = min(0.999 * t_border, left)
        if t_end >= 1e-3 and label(x + t_end * d) != la:
            t_hit = bisect(x, d, 0.0, t_end, la)
            return x + t_hit * d, d
        if t_border >= left:
            return None                                 # reach exhausted inside this piece (layer 0 always ends here)
        hop = t_border + 1e-4 * (1.0 + float(np.max(np.abs(x))))      # clearly into the next piece
        projected = project(prefix, rows, j, x + hop * d)
        if projected is None:
            return None
        x = projected[0]
        nu = projected[1]
        left = left - hop
        if label(x) != la:
            return None                                 # the label changed inside the hop itself: cannot be bisected in one piece
    return None


def boundary_in_plane(prefix, rows, j, x, nu, rng, reach, directions=3):
    """Look for a decision boundary on H_j within `reach` of x along a few random tangents (both ways).  (x_B, d) or None."""
    la = label(x)
    for _ in range(directions):
        d0 = rng.normal(size=len(x))
        for sign in (1.0, -1.0):
            result = walk_to_boundary(prefix, rows, j, x, nu, sign * d0, la, reach)
            if result is not None:
                return result
    return None


def predict_station(stations, s, e):
    """Where to expect the boundary at station s, and how wide a bracket to start with, from the stations measured so far."""
    if len(stations) >= 2:
        (s1, t1), (s2, t2) = stations[-2], stations[-1]
        t_guess = t2 + (t2 - t1) * (s - s2) / (s2 - s1)
        width = max(abs(t2 - t1), e) / 8
        return t_guess, width
    if stations:
        return stations[-1][1], e / 4
    return 0.0, e / 4


def measure_station(p, d, t_guess, width, reach_t):
    """Offset t along d of the boundary through the line p + t d, bisected to round-off; None if none is found within reach_t."""
    lc = label(p + t_guess * d)
    w = width
    while w <= reach_t:
        for sign in (+1, -1):
            if label(p + (t_guess + sign * w) * d) != lc:
                return bisect(p, d, t_guess, t_guess + sign * w, lc)
        w *= 4
    return None


def fit_line(stations):
    """Least-squares line t = alpha + beta s through the stations.  Returns (alpha, beta, max residual, max |t|)."""
    S = np.array([s for s, _ in stations])
    T = np.array([t for _, t in stations])
    beta, alpha = np.polyfit(S, T, 1)
    resid = float(np.max(np.abs(T - (alpha + beta * S))))
    return alpha, beta, resid, float(np.max(np.abs(T)))


def two_line(x_B, nu, d, e, reach_t):
    lines = {}
    for side in (-1, +1):
        stations = []
        for k in (1, 2, 3):
            s = side * k * e
            p = x_B + s * nu
            t_guess, width = predict_station(stations, s, e)
            t = measure_station(p, d, t_guess, width, reach_t)
            if t is None:
                return None, "no boundary at station"
            stations.append((s, t))
        lines[side] = fit_line(stations)

    alpha_m, beta_m, resid_m, _ = lines[-1]
    alpha_p, beta_p, resid_p, _ = lines[+1]
    tol = COLLINEAR * EPS * max(1.0, float(np.max(np.abs(x_B)))) * (1 + abs(beta_m) + abs(beta_p))
    resid = max(resid_m, resid_p)
    if resid > tol:
        return None, "stations not collinear"
    bend = beta_p - beta_m
    if abs(bend) < NO_BEND:
        return None, "no bend"
    s_star = (alpha_m - alpha_p) / bend
    if abs(s_star) > 0.5 * e:
        return None, "kink outside window"
    if s_star > 0:
        own_intercept = alpha_m
    else:
        own_intercept = alpha_p
    if abs(own_intercept) > tol:
        return None, "second kink in the gap"
    t_star = alpha_m + beta_m * s_star
    return (s_star, t_star, abs(bend), resid), None


def mint_kinks(prefix, rows, j, K, rng, radii, e, max_tries):
    """Mint up to K kinks on H_j.  Returns (H = prefix activations at the kinks, their bends, stats)."""
    H = []
    bends = []
    rejected = {}
    tried = 0
    dim = rows.shape[1] - 1

    def reject(why):
        rejected[why] = rejected.get(why, 0) + 1

    while len(H) < K and tried < max_tries:
        tried += 1
        radius = radii[tried % len(radii)]
        x0 = rng.normal(size=dim) * radius
        projected = project(prefix, rows, j, x0)
        if projected is None:
            reject("projection failed")
            continue
        x = projected[0]
        nu = projected[1]

        found = boundary_in_plane(prefix, rows, j, x, nu, rng, reach=4.0 * radius)
        if found is None:
            reject("no boundary in reach")
            continue
        x_B, d = found

        # Re-project x_B (moves it by round-off only) and make sure it stayed on the same piece of the prefix.
        projected = project(prefix, rows, j, x_B)
        if projected is None or not np.array_equal(pattern(prefix, projected[0]), pattern(prefix, x_B)):
            reject("boundary point off the piece")
            continue
        x_B, nu, piece, local = projected
        if local < SECOND_KINK_STATIONS * e:
            reject("another hyperplane within 8 stations")
            continue

        result, why = two_line(x_B, nu, d, e, reach_t=min(4 * e, local / 3))
        if result is None:
            reject(why)
            continue
        s_star, t_star, bend, resid = result
        x_star = x_B + s_star * nu + t_star * d
        h_star = prefix_forward(prefix, x_star)[0]         # layer 0: x_star itself
        H.append(h_star)
        bends.append(bend)

    info = dict(tried=tried, rejected=rejected)
    if bends:
        info["bend_median"] = float(np.median(bends))
        info["bend_min"] = float(np.min(bends))
    else:
        info["bend_median"] = None
        info["bend_min"] = None
    return np.array(H), np.array(bends), info


def weighted_null(M, weights, keep):
    """Null vector of the kept rows of M, each row weighted."""
    _, _, Vt = np.linalg.svd(M[keep] * weights[keep, None], full_matrices=False)
    return Vt[-1]


def residuals(M, v):
    """Distance of every kink from the plane v (v = [normal | offset])."""
    return np.abs(M @ v) / np.linalg.norm(v[:-1])


def trim_against_old_row(M, old, floor):
    """First trim: drop the kinks far (20x the median) from the plane of the row we started from."""
    r_old = np.abs(M @ old)
    keep = r_old <= max(20 * float(np.median(r_old)), floor)
    if keep.sum() < extra_points(M.shape[1]):
        keep = np.ones(len(M), bool)
    return keep


def best_subset_residuals(M, keep):
    """Second trim (least median of squares): fit 64 random subsets of the kept kinks, unweighted, and return the residuals of
    every kink under the subset fit whose median residual over the kinks OUTSIDE the subset is smallest.  A subset free of the
    (at most few) bad kinks fits to round-off and shows every bad kink at its full offset.  Returns (median, residuals) or None."""
    n_unknowns = M.shape[1]
    idx = np.flatnonzero(keep)
    m = int(np.clip(len(idx) - 16, n_unknowns, n_unknowns + 8))
    rng = np.random.default_rng(0)
    # A subset on which one prefix unit is off everywhere has a 2-dim null space; such fits are recognised by their second-smallest
    # singular value being far below that of the whole set, and skipped.
    rank_floor = 0.1 * np.linalg.svd(M[keep], compute_uv=False)[-2]
    best = None
    for _ in range(64):
        subset = rng.choice(idx, m, replace=False)
        S, Vt = np.linalg.svd(M[subset], full_matrices=False)[1:]
        if S[-2] < rank_floor:
            continue
        res = residuals(M, Vt[-1])
        outside = keep.copy()
        outside[subset] = False
        if outside.any():
            median = float(np.median(res[outside]))
        else:
            median = np.inf
        if best is None or median < best[0]:
            best = (median, res)
    return best


def fit_row(H, bends, old):
    """[row | bias] = null vector of [H | 1], rows weighted by bend, after trimming outliers (see the trim_* helpers).

    Returns (row scaled to unit norm and oriented like `old`, label-free fit statistics)."""
    M = np.concatenate([H, np.ones((len(H), 1))], axis=1)
    weights = np.minimum(np.asarray(bends), 1.0)
    minimum = extra_points(M.shape[1])
    floor = 64 * EPS * float(np.max(np.abs(M)))
    old = np.asarray(old, dtype=np.float64) / np.linalg.norm(old[:-1])

    keep = trim_against_old_row(M, old, floor)
    best = best_subset_residuals(M, keep)
    if best is not None:
        median, res = best
        candidate = keep & (res <= max(20 * median, floor))
        if candidate.sum() >= minimum:
            keep = candidate
    # Finally settle on the weighted fit's own residuals.
    for _ in range(2):
        res = residuals(M, weighted_null(M, weights, keep))
        candidate = res <= max(20 * float(np.median(res[keep])), floor)
        if candidate.sum() < minimum or (candidate == keep).all():
            break
        keep = candidate

    v = weighted_null(M, weights, keep)
    S = np.linalg.svd(M[keep], compute_uv=False)
    v = v / np.linalg.norm(v[:-1])
    if v[:-1] @ old[:-1] < 0:
        v = -v
    stats = dict(
        points_used=int(keep.sum()),
        resid_rms=float(np.sqrt(np.mean((M[keep] @ v) ** 2))),
        sv_gap=float(S[-2] / np.sqrt(keep.sum())),
    )
    return v, stats


def refine_neuron(args):
    """One neuron (runs in a worker process).  Returns (j, new or old row, stats)."""
    j, prefix, rows, K, seed, radii, e = args
    queries_before = query_count()
    t0 = time.time()
    rng = np.random.default_rng([seed, j])
    H, bends, info = mint_kinks(prefix, rows, j, K, rng, radii, e, max_tries=40 * K)
    info["kinks"] = len(H)

    def finish(stats):
        stats["queries"] = query_count() - queries_before
        stats["seconds"] = round(time.time() - t0, 2)
        return stats

    if len(H) < extra_points(rows.shape[1]):
        info["kept_old"] = True
        return j, rows[j], finish(info)
    v, fit = fit_row(H, bends, rows[j])
    stats = dict(info, **fit)
    stats["moved"] = float(np.linalg.norm(v - rows[j]))
    stats = finish(stats)
    if fit["resid_rms"] > BAD_FIT or fit["sv_gap"] < BAD_RANK:
        # A fit this far from round-off is contaminated, or the kept kinks span no plane: the input row is the better answer.
        stats["kept_old"] = True
        stats["bad_fit"] = True
        return j, rows[j], stats
    return j, v, stats


def run_jobs(jobs, procs):
    if procs <= 1:
        return [refine_neuron(job) for job in jobs]
    import multiprocessing as mp
    context = mp.get_context("fork")
    with context.Pool(min(procs, len(jobs))) as pool:
        return pool.map(refine_neuron, jobs, chunksize=1)


def pass_summary(per_neuron):
    values = list(per_neuron.values())
    return dict(
        queries=sum(s["queries"] for s in values),
        kinks=sum(s["kinks"] for s in values),
        tried=sum(s["tried"] for s in values),
        kept_old=[j for j, s in per_neuron.items() if s.get("kept_old")],
        # Over every refit that was attempted, the bad ones included.
        resid_rms_max=max([s["resid_rms"] for s in values if "resid_rms" in s], default=float("nan")),
        sv_gap_min=min([s["sv_gap"] for s in values if "sv_gap" in s], default=float("nan")),
        neurons={int(j): per_neuron[j] for j in sorted(per_neuron)},
    )


def refine_layer(rows, prefix, K=64, procs=8, seed=0, radii=(1.0, 2.0, 4.0), e=1e-3, passes=1, neurons=None):
    """Refine the given rows (any scale) behind the prefix.  Returns (refined rows with unit norm, stats)."""
    rows = np.array(rows, dtype=np.float64)
    rows /= np.linalg.norm(rows[:, :-1], axis=1, keepdims=True)
    if neurons is None:
        neurons = list(range(len(rows)))
    else:
        neurons = list(neurons)
    t0 = time.time()

    # Load the oracle once, in the parent, before the workers are forked.
    queries_before = query_count()
    label(np.zeros(rows.shape[1] - 1))
    stats = dict(passes=[], warmup_queries=query_count() - queries_before)

    for p in range(passes):
        jobs = [(j, prefix, rows, K, seed + 1000 * p, radii, e) for j in neurons]
        results = run_jobs(jobs, procs)
        new_rows = rows.copy()
        per_neuron = {}
        for j, v, neuron_stats in results:
            new_rows[j] = v
            per_neuron[j] = neuron_stats
        rows = new_rows
        stats["passes"].append(pass_summary(per_neuron))

    stats.update(
        queries=sum(p["queries"] for p in stats["passes"]) + stats["warmup_queries"],
        wall_seconds=round(time.time() - t0, 1),
        kinks_per_neuron=K,
        procs=procs,
        radii=list(radii),
        station=e,
        seed=seed,
    )
    return rows, stats


USAGE = """refine.py LAYER TARGET.npy [--prefix P0.npy ..] [--out OUT.npy] [--kinks 64] [--procs 8] [--passes 1] [--seed 0]
          [--radius 1,2,4] [--station 1e-3] [--neurons 0,1,..]"""


def option(argv, key, default=None):
    if key in argv:
        return argv[argv.index(key) + 1]
    return default


def prefix_files(argv):
    files = []
    if "--prefix" not in argv:
        return files
    i = argv.index("--prefix") + 1
    while i < len(argv) and not argv[i].startswith("--"):
        files.append(os.path.abspath(argv[i]))
        i += 1
    return files


def rejection_summary(last_pass):
    counts = {}
    for neuron_stats in last_pass["neurons"].values():
        for why, n in neuron_stats["rejected"].items():
            counts[why] = counts.get(why, 0) + n
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    text = ", ".join("%s: %d" % item for item in ordered)
    return text or "none"


def main(argv):
    if len(argv) < 2:
        print(USAGE)
        return 2
    layer = int(argv[0])
    target = os.path.abspath(argv[1])
    files = prefix_files(argv)
    assert len(files) == layer, "layer %d needs %d prefix files, got %d" % (layer, layer, len(files))
    prefix = [load_unit(path) for path in files]

    default_out = os.path.splitext(target)[0] + "_refined.npy"
    out = os.path.abspath(option(argv, "--out", default_out))
    neurons = None
    if "--neurons" in argv:
        neurons = [int(n) for n in option(argv, "--neurons").split(",")]
    radii = tuple(float(r) for r in option(argv, "--radius", "1,2,4").split(","))

    rows, stats = refine_layer(
        np.load(target),
        prefix,
        K=int(option(argv, "--kinks", 64)),
        procs=int(option(argv, "--procs", 8)),
        seed=int(option(argv, "--seed", 0)),
        radii=radii,
        e=float(option(argv, "--station", 1e-3)),
        passes=int(option(argv, "--passes", 1)),
        neurons=neurons,
    )
    stats.update(
        layer=layer,
        target=target,
        prefix=files,
        note="labels only (oracle.label) + the recovered prefix files above; queries = sum of per-neuron query_count() deltas",
    )
    np.save(out, rows)
    with open(os.path.splitext(out)[0] + ".json", "w") as fh:
        json.dump(stats, fh, indent=1)

    last = stats["passes"][-1]
    print("layer %d: %d neurons refined, %d kinks (%d tries), queries %s, wall %.1f s (%d procs); "
          "label-free: resid rms max %.1e, sv gap min %.2f, kept old %s"
          % (layer, len(last["neurons"]), last["kinks"], last["tried"], "{:,}".format(stats["queries"]), stats["wall_seconds"],
             stats["procs"], last["resid_rms_max"], last["sv_gap_min"], last["kept_old"]))
    print("   rejected starts: %s" % rejection_summary(last))
    print("   wrote %s (+ .json)" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
