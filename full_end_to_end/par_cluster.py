# Groups the dual points of one layer by neuron.  Every dual point is tried as the seed of a cluster: the other duals
# are scanned and those consistent with it (recover_weights.is_consistent) are added; a cluster is accepted once the
# whole set solves to one hyperplane with a common bias.  Seeds scan in parallel, each worker with its own copy of the
# prefix.  cluster_layer() returns a list of {"cluster": [duals], "weight": w, "bias": b}.
import logging
import os
import pickle
import hashlib
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import torch
from utils import DEVICE
from recover_weights import is_consistent, Prefix

multiprocessing.set_start_method('spawn', force=True)

BASE_SEED = 10000
MIN_CLUSTER = 3
MIN_CLUSTER_ZEROED = 8    # a cluster whose solution sets weights to 0 must grow to this many duals before it counts
SPLIT_MIN = 0.25           # a cluster split by fewer than this fraction of the others' hyperplanes is a deeper neuron (disguised)
PRUNE_SIZES = (4, 6)       # a cluster whose members have no common hyperplane: subsets of these sizes are tried (prune)
PRUNE_TRIALS = 40          #   this many of each size
PRUNES_MAX = 3             #   at most this many times per seed
PRUNE_TOL = 1e-5           #   a member lies on a subset's hyperplane if its activation gives the bias to this
FACE_TOL = 1e-6            # a row whose input normal is within this (1 - |cos|) of a boundary normal at a dual ...
FACE_SHARE = 0.5           # ... at this share of its duals is a face of the decision boundary (is_face)
FACE_PLANE_TOL = 1e-5      # a dual lies on a face if its activation gives the face's bias to this
FACES_MAX = 3              # a seed gives up after this many faces
GAP_MIN = 0.007            # a fit whose second singular value (relative) is below this is too poorly determined
IDENTITY_MAX = 1e-8        # a hyperplane that moves off its duals by less than this (relative) holds everywhere (prefix_identity)
IDENTITY_SHARE = 0.25      # ... and a cluster with this share of such identity duals is rejected (prefix_identity)
ENGAGED_MIN = 1e-3         # a row with less of its weight than this on the inputs on at one of its duals is rejected
SAME_ROW = 1 - 1e-6        # a cluster this parallel to a rejected one is the same hyperplane again (cluster_layer)
PROGRESS_EVERY = 100       # a progress line is printed after this many seeds
SAVE_EVERY = 200           # the clustering state is saved after this many seeds (and after every cluster)
SECOND_PASS_ID = 10 ** 6   # cluster ids (and so seeds of the random draws) of the second pass start here
SAME_NORMAL = 1 - 1e-6     # two duals whose boundary normals are this parallel are the same bend (not added twice)
FILES_AT_START = 5         # dual files loaded up front; one more per 1000 seeds tried

# set in every worker by init_worker
_worker_prefix = None
_worker_layer = None
_duals_dir = None


_rest = (None, None)      # (path, duals) of the leftover pool of the second pass, loaded once per worker


def load_rest(path):
    global _rest
    if _rest[0] != path:
        _rest = (path, pickle.load(open(path, "rb")))
    return _rest[1]


def init_worker(layer, prefix_files, duals_dir):
    global _worker_prefix, _worker_layer, _duals_dir
    _worker_layer = layer
    _worker_prefix = Prefix(prefix_files).to(DEVICE)
    _duals_dir = duals_dir


def reject_on_earlier_layer(x, prefix, layer):
    """Filter out points that are rejected by earlier layers"""
    if layer == 0: return x
    out = prefix.forward_nolastrelu(torch.tensor(np.stack([z[1] for z in x])).to(DEVICE)).cpu()
    reject = torch.any(torch.abs(out) < 1e-5, (0,2))
    x = [z for z, r in zip(x, reject) if not r]
    return x



_walks = None             # every walk file's usable duals and their prefix outputs, loaded once per worker


def walk_pool(dual_files):
    """[(duals, H)] per walk file, in dual_files' order: the duals that are not on a hyperplane of an earlier layer, and
    their outputs behind the prefix (to tell, without a pairwise test, which lie on a hyperplane already found)."""
    global _walks
    if _walks is None:
        _walks = []
        for f in dual_files:
            x = reject_on_earlier_layer(pickle.load(open(os.path.join(_duals_dir, f), "rb")), _worker_prefix, _worker_layer)
            H = (_worker_prefix.forward(torch.tensor(np.array([d[1] for d in x])).to(DEVICE)).cpu().numpy()
                 if x else np.zeros((0, 0)))
            _walks.append((x, H))
    return _walks


def partners(dual_files, home, known, rng):
    """The candidates for a seed's cluster, one walk at a time in turn: one from the seed's own walk (index `home` in
    dual_files), one from the next, and so on, cycling, each walk in its own random order.  Duals on a hyperplane in
    `known` (the clusters found or rejected so far) are left out: they belong to another neuron."""
    walks = walk_pool(dual_files)
    if known:
        W = np.array([c["weight"] for c in known])
        B = np.array([c["bias"] for c in known])
    start = home if home is not None else int(rng.integers(len(walks)))
    order = []
    for k in list(range(start, len(walks))) + list(range(start)):
        duals, H = walks[k]
        if not duals:
            continue
        keep = np.ones(len(duals), bool)
        if known:
            keep = ~np.any(np.abs(H @ W.T + B) < 1e-5, axis=1)
        idx = np.flatnonzero(keep)
        rng.shuffle(idx)
        if len(idx):
            order.append((duals, idx))
    position = 0
    while order:
        still = []
        for duals, idx in order:
            yield duals[idx[position]]
            if position + 1 < len(idx):
                still.append((duals, idx))
        order = still
        position += 1


def process_single_candidate(args):
    """Process a single candidate neuron to find a cluster"""
    cluster_id, a, found_clusters, dual_files, random_seed, worker_id = args[:6]
    # args[6]: the index in dual_files of the seed's own walk, where the scan starts (a walk tends to cross a neuron's
    # hyperplane several times in a row; see partners), or (path, i): second pass, the seed is duals[i] of the leftover
    # pool in path, and only that pool is scanned, in full
    home, pool = (args[6] if len(args) > 6 else None), None
    if isinstance(home, tuple):
        pool = load_rest(home[0])
        a = pool[home[1]]
        home = None
    
    # Use global prefix initialized in worker
    prefix = _worker_prefix
    layer = _worker_layer

    logging.info(f"[W{worker_id}] Processing ID {cluster_id}, found clusters: {len(found_clusters)}")
    
    # Check if this neuron is already in a found cluster
    for j, previous in enumerate(found_clusters):
        prior_w = previous['weight']
        prior_b = previous['bias']
        
        if np.abs((prefix.forward(torch.tensor(np.array(a[1])).to(DEVICE)).cpu() @ prior_w)[0] + prior_b) < 1e-5:
            logging.info(f"[W{worker_id}] Skipped {cluster_id} because found this previously")
            return None
    
    # Search for cluster members
    maybe = [a]
    last_try_size = MIN_CLUSTER
    prunes = 0
    rng = np.random.default_rng(cluster_id)
    faces = []                # decision boundary faces (w, bias) the members turned out to fit: their duals stay out
    restart = 0               # the scan position the abort rules count from (moved on when a face is dropped)

    def on_face(dual, face):
        h = prefix.forward(torch.tensor(np.array(dual[1])).to(DEVICE)).cpu().numpy()[0]
        return abs(h @ face[0] - face[1]) < FACE_PLANE_TOL

    def prune_or_stop(why):
        """The members fail together: keep the ones on the best hyperplane through the seed and grow on, or give up."""
        nonlocal maybe, prunes
        logging.info(f"[W{worker_id}] {why}")
        if prunes >= PRUNES_MAX:
            return False
        prunes += 1
        kept = prune(a, maybe, prefix, layer, rng)
        if kept is None:
            return False
        logging.info(f"[W{worker_id}] Pruned {cluster_id} from {len(maybe)} to {len(kept)} duals")
        maybe = kept
        return True
    
    logging.info(f"Going {dual_files}")
    scan = pool if pool is not None else partners(dual_files, home, found_clusters,
                                                  np.random.default_rng([BASE_SEED, cluster_id]))
    for j, b in enumerate(scan):
        if b is a or np.array_equal(b[1], a[1]):
            logging.info(f"[W{worker_id}] Skip adding the same neuron twice")
            continue
        if pool is None and j - restart > 1000 and len(maybe) == 1:
            logging.info(f"[W{worker_id}] Abort {cluster_id} because <1000")
            break
        if len(maybe)**2/(j - restart + 1) < .0001:
            logging.info(f"[W{worker_id}] Abort {cluster_id} because rare")
            break

        S = is_consistent((a, b), prefix, layer=layer, do_return_soln=False)
        
        if type(S) == np.float64 and S < 1e-5 and not any(on_face(b, face) for face in faces):
            an1, an2 = a[3:]          # the boundary normals on either side of each dual, stored by the walk
            bn1, bn2 = b[3:]
            # the same bend found twice; genuine pairs of one neuron can have boundary normals far closer than an
            # intruder's (a neuron that hardly feeds the decision bends the boundary little), so only near 1
            if max(abs(an1 @ bn1), abs(an1 @ bn2), abs(an2 @ bn1), abs(an2 @ bn2)) > SAME_NORMAL:
                continue
            
            logging.info(f"[W{worker_id}] Added consistent neuron to {cluster_id} {j} {len(maybe)}")
            maybe.append(b)
        
        if len(maybe) >= last_try_size:
            if len(maybe) > 100:
                logging.info(f"[W{worker_id}] CONFUSED THAT ITS SO BIG")
                break
            
            S, soln = is_consistent(maybe, prefix, layer=layer, do_return_soln=True)
            logging.info(f"[W{worker_id}] Check consistent for {cluster_id} {S}")
            
            if type(S) == np.ndarray:
                if S[-1] > 1e-7 * S[0]:
                    if prune_or_stop("Invalid cluster"):
                        continue
                    break
                if S[-2] < GAP_MIN * S[0]:
                    # barely one null direction: the hyperplane is poorly determined by these duals (many weights
                    # unknown, few duals), and duals of several neurons can fit it by accident; grow on for more
                    logging.info(f"[W{worker_id}] {cluster_id}: hyperplane poorly determined ({S[-2] / S[0]:.1e})")
                    last_try_size = max(2 * last_try_size, len(maybe) + 1)
                    continue
                if is_face(maybe, soln, prefix):
                    # the members fit a face of the decision boundary, not a neuron: keep the seed and whatever is off
                    # the face, keep the face's duals out from now on, and grow on
                    H = prefix.forward(torch.tensor(np.array([m[1] for m in maybe])).to(DEVICE)).cpu().numpy()
                    faces.append((soln, float(np.median(H @ soln))))
                    logging.info(f"[W{worker_id}] {cluster_id}: decision boundary face of {len(maybe)} duals")
                    if len(faces) > FACES_MAX:
                        break
                    maybe = [a] + [m for m in maybe[1:] if not on_face(m, faces[-1])]
                    last_try_size = max(MIN_CLUSTER, len(maybe) + 1)
                    restart = j
                    continue
                if np.any(soln == 0) and len(maybe) < MIN_CLUSTER_ZEROED:
                    # weights set to 0 leave few unknowns: a few duals of different neurons fit them by accident
                    last_try_size = MIN_CLUSTER_ZEROED
                    continue
                
                biases = []
                for m in maybe:
                    bias = (prefix.forward(torch.tensor(np.array(m[1])).to(DEVICE)).cpu().numpy() @ soln)[0]
                    biases.append(bias)
                med_bias = np.median(biases)
                
                if np.all(np.abs(biases - med_bias) < 1e-4):
                    result = {"cluster": maybe, "weight": soln, "bias": -med_bias}
                    if prefix_identity(result, prefix):
                        logging.info(f"[W{worker_id}] Cluster {cluster_id} rejected: its hyperplane holds off the duals too")
                        break
                    if disengaged(result, prefix):
                        logging.info(f"[W{worker_id}] Cluster {cluster_id} rejected: it weighs only inputs that are off")
                        break
                    logging.info(f"[W{worker_id}] Found good cluster {cluster_id} of size {len(maybe)}")
                    return result
                else:
                    if prune_or_stop("Wrong bias"):
                        continue
                    break
            else:
                last_try_size *= 2
    
    return None


def input_normal(prefix, x0, w):
    """Unit input-space normal at x0 of the hyperplane w . h + b = 0 behind the prefix (in x0's on/off pattern)."""
    x = np.asarray(x0, dtype=np.float64)
    A = np.eye(len(x))
    for linear in prefix.fcs:
        W = linear.weight.detach().cpu().numpy()
        z = W @ x + linear.bias.detach().cpu().numpy()
        on = z > 0
        A = (W @ A) * on[:, None]
        x = np.where(on, z, 0.0)
    n = A.T @ w
    return n / max(np.linalg.norm(n), 1e-300)


def is_face(members, w, prefix):
    """Whether the hyperplane w is a face of the decision boundary rather than a neuron.

    Every dual lies on the decision boundary, and its bend line lies in both boundary pieces next to it.  Wherever the
    layers above keep their pattern, a piece of the boundary is flat behind the prefix, so all the bends on its edges
    (whatever neuron makes them) have their bend lines in it: their duals fit it exactly, like a cluster.  Such a face's
    normal is one of the two boundary normals measured at the dual; a neuron's is a combination of both."""
    face = 0
    for d in members:
        n = input_normal(prefix, d[1], w)
        face += max(abs(n @ d[3]), abs(n @ d[4])) > 1 - FACE_TOL
    return face >= FACE_SHARE * len(members)


def prune(a, maybe, prefix, layer, rng):
    """The members of maybe that lie on the best hyperplane through the seed a, or None.

    Deep in the network the pairwise test that grows a cluster lets duals of other neurons in (few earlier neurons are
    on at a dual, so two duals have few coordinates to disagree on), and then the whole set has no common hyperplane.
    Small random subsets with the seed are solved (PRUNE_TRIALS of each size in PRUNE_SIZES); a member lies on a subset's
    hyperplane if its activation there gives the subset's bias to PRUNE_TOL (tighter than the full cluster's 1e-4).  The
    hyperplane most members lie on wins, if it is clean enough for a cluster itself."""
    H = prefix.forward(torch.tensor(np.array([m[1] for m in maybe])).to(DEVICE)).cpu().numpy()
    others = np.arange(1, len(maybe))
    best = None
    for size in PRUNE_SIZES:
        if size > len(maybe):
            break
        for _ in range(PRUNE_TRIALS):
            pick = np.r_[0, rng.choice(others, size - 1, replace=False)]
            S, soln = is_consistent([maybe[i] for i in pick], prefix, layer=layer, do_return_soln=True)
            if type(S) != np.ndarray or S[-1] > 1e-7 * S[0]:
                continue
            z = H @ soln
            on = np.abs(z - np.median(z[pick])) < PRUNE_TOL
            if on[0] and (best is None or on.sum() > best.sum()):
                best = on
    if best is None or best.sum() < MIN_CLUSTER:
        return None
    return [m for m, keep in zip(maybe, best) if keep]


def prefix_identity(result, prefix, directions=4):
    """Whether the cluster's hyperplane holds around many of its duals in every direction, not only on them.

    Where the earlier-layer outputs that are on depend on fewer neurons further down than there are of them, they
    satisfy an exact linear relation everywhere in those regions.  Duals of any neurons there fit that relation exactly
    and with a common bias, so they pass as a cluster (most easily with weights set to 0, which leave few unknowns),
    and a few duals that fit it loosely can come along: a cluster is rejected once IDENTITY_SHARE of its duals are such
    identity duals.  A real neuron's hyperplane is left as soon as a point moves off it, at every one of its duals.
    Every move keeps the dual's own on/off pattern (forward_around), so its size does not matter."""
    w = result["weight"]
    rng = np.random.default_rng(0)
    identity = 0
    for d in result["cluster"]:
        x0 = np.asarray(d[1], dtype=np.float64)
        x = np.vstack([x0, x0 + rng.standard_normal((directions, x0.size))])
        h = prefix.forward_around(torch.tensor(x).to(DEVICE)).cpu().numpy()
        moved = h[1:] - h[0]
        size = np.linalg.norm(moved, axis=1) * np.linalg.norm(w)
        identity += not np.any(np.abs(moved @ w) > IDENTITY_MAX * size)
    return identity >= IDENTITY_SHARE * len(result["cluster"])


def disengaged(result, prefix):
    """Whether the cluster's row puts next to none of its weight on the inputs that are on at one of its duals.

    w . h + b = 0 holds trivially wherever every input the row weighs is off (and its bias is ~0): such a row fits the
    duals of any neurons there.  A real neuron is zero at its own duals through its weights on the inputs that are on."""
    w = result["weight"] / np.linalg.norm(result["weight"])
    H = prefix.forward(torch.tensor(np.array([d[1] for d in result["cluster"]])).to(DEVICE)).cpu().numpy()
    return bool(np.any(np.sqrt(((H > 1e-4) * w ** 2).sum(axis=1)) < ENGAGED_MIN))


def is_duplicate(found_clusters, result, cos=0.8):
    """Whether result is a neuron already found (|cos| above `cos`).  Weights set to 0 (exactly 0.0) are unknown, not
    zero, and two clusters of one neuron may zero different ones: they are compared on the weights both of them know."""
    w = result['weight']
    for other in found_clusters:
        known = (w != 0) & (other['weight'] != 0)
        a, b = w[known], other['weight'][known]
        if len(a) and abs(a @ b) > cos * np.linalg.norm(a) * np.linalg.norm(b):
            return True
    return False


def disguised(clusters, prefix):
    """Indices of the clusters whose duals are split by fewer than SPLIT_MIN of the other clusters' hyperplanes.

    Wherever this layer's on/off pattern stays the same, the layer is one linear map, so a deeper neuron's zero set is a
    flat hyperplane behind the prefix too: a cluster of its duals from a single region solves exactly like a neuron of
    this layer (and so, once weights are set to 0 and few unknowns are left, can duals of several neurons from few
    regions).  The duals of a real neuron of this layer are spread over many regions of its own layer, on both sides of
    most of the other neurons' hyperplanes.  Not all of them: the other clusters may include such impostors, or be
    inexact where they have weights set to 0, so a cluster is not required to lie in one region of all of them."""
    found = []
    for i, c in enumerate(clusters):
        others = [o for j, o in enumerate(clusters) if j != i]
        if len(c["cluster"]) < 2 or not others:
            continue
        H = prefix.forward(torch.tensor(np.array([d[1] for d in c["cluster"]])).to(DEVICE)).cpu().numpy()
        W = np.array([o["weight"] for o in others])
        b = np.array([o["bias"] for o in others])
        side = H @ W.T + b > 0
        split = np.sum(side.any(axis=0) & ~side.all(axis=0))
        if split < SPLIT_MIN * len(others):
            found.append(i)
    return found


def dual_key(dual):
    """A name for a dual point that stays the same from run to run (to remember which seeds were tried)."""
    return hashlib.sha1(np.asarray(dual[1], dtype=np.float64).tobytes()).hexdigest()[:16]


def save_clusters(path, clusters, rejected, complete, tried=(), tried_second=()):
    """Write the clustering state of a layer (atomically): the clusters accepted, those rejected, whether the
    clustering finished, and the seeds (dual_key) already tried in the first and in the second pass."""
    with open(path + ".partial", "wb") as fh:
        pickle.dump({"clusters": clusters, "rejected": rejected, "complete": complete, "tried": set(tried),
                     "tried_second": set(tried_second)}, fh)
    os.replace(path + ".partial", path)


def load_clusters(path):
    """The clustering state written by save_clusters (an older file holding only the list of clusters counts as
    complete)."""
    state = pickle.load(open(path, "rb"))
    if isinstance(state, list):
        state = {"clusters": state, "rejected": [], "complete": True}
    state.setdefault("tried", set())
    state.setdefault("tried_second", set())
    return state


def mark_complete(path):
    state = load_clusters(path)
    save_clusters(path, state["clusters"], state["rejected"], True, state["tried"], state["tried_second"])


def cluster_layer(layer, duals_dir, prefix_files, num_workers, width, checkpoint=None, previous=None):
    """Returns the clusters found; stops as soon as `width` (the number of neurons in the layer) have been found that
    are not deeper neurons in disguise (see disguised).

    checkpoint: a file the state is saved to (save_clusters, marked incomplete) whenever a cluster is accepted or
    dropped, and every SAVE_EVERY seeds, so that an interrupted run can go on; previous: such a state to go on from (its
    clusters are kept, the seeds on their hyperplanes are skipped, and so are the seeds already tried, in each pass).
    Marking the clustering complete is the caller's (mark_complete)."""
    prefix = Prefix(prefix_files).to(DEVICE)
    dual_files = sorted(os.listdir(duals_dir))
    random.Random(BASE_SEED).shuffle(dual_files)

    def load_seeds(f):
        x = pickle.load(open(os.path.join(duals_dir, f), "rb"))
        random.Random(BASE_SEED).shuffle(x)
        return reject_on_earlier_layer(x, prefix, layer)

    seeds, homes, seed_h = [], [], []   # the seeds, the index in dual_files of their walk, their prefix outputs

    def add_seeds(k):
        x = load_seeds(dual_files[k])
        seeds.extend(x)
        homes.extend([k] * len(x))
        if x:
            seed_h.extend(prefix.forward(torch.tensor(np.array([d[1] for d in x])).to(DEVICE)).cpu().numpy())

    for k in range(FILES_AT_START):
        add_seeds(k)
    files_loaded = FILES_AT_START

    found_clusters = list(previous["clusters"]) if previous else []
    # deeper neurons in disguise: their duals are skipped and they are not accepted again
    rejected = list(previous["rejected"]) if previous else []

    tried = set(previous.get("tried", ())) if previous else set()                  # seeds done, first pass
    tried_second = set(previous.get("tried_second", ())) if previous else set()    # seeds done, second pass
    finished = 0

    def save():
        if checkpoint is not None:
            save_clusters(checkpoint, found_clusters, rejected, False, tried, tried_second)

    started = time.time()

    def done_with(key, second=False):
        """Remember that a seed has been tried; save every SAVE_EVERY of them, report every PROGRESS_EVERY."""
        nonlocal finished
        (tried_second if second else tried).add(key)
        finished += 1
        if finished % SAVE_EVERY == 0:
            save()
        if finished % PROGRESS_EVERY == 0:
            rate = finished / max(time.time() - started, 1e-9)
            if second:
                where = "second pass: %d of %d seeds tried" % (len(tried_second), len(rest))
            else:
                where = "%d seeds tried of %d loaded (%d of %d walk files)" % (len(tried), len(seeds), files_loaded,
                                                                            len(dual_files))
            print("   %s; %d of %d clusters; %.1f seeds/s" % (where, len(found_clusters), width, rate), flush=True)

    if len(found_clusters) >= width:
        save()
        return found_clusters
    max_active = num_workers * 2
    next_seed = 0
    futures = {}

    def drop_disguised():
        dropped = disguised(found_clusters, prefix)
        for i in reversed(dropped):
            print("   dropped cluster of %d duals: on one side of most of the others' hyperplanes, a deeper neuron"
                  % len(found_clusters[i]['cluster']), flush=True)
            rejected.append(found_clusters.pop(i))
        if dropped:
            save()

    def submit(executor):
        nonlocal next_seed
        known = found_clusters + rejected
        if known:
            W = np.array([c["weight"] for c in known])
            B = np.array([c["bias"] for c in known])
        while len(futures) < max_active and next_seed < len(seeds):
            if dual_key(seeds[next_seed]) in tried:
                next_seed += 1                 # tried before this run was interrupted
                continue
            if known and np.any(np.abs(W @ seed_h[next_seed] + B) < 1e-5):
                next_seed += 1                 # on a hyperplane found (or rejected) already: its neuron is known
                continue
            worker = next_seed % num_workers
            args = (next_seed, seeds[next_seed], found_clusters + rejected, dual_files, worker, worker, homes[next_seed])
            futures[executor.submit(process_single_candidate, args)] = next_seed
            next_seed += 1

    def take(result):
        """Accept a worker's result; True once the layer is complete."""
        # A rejected cluster only stops the same hyperplane from coming back: a deeper neuron's row within one region is
        # a combination of this layer's rows, and may well be as close to a real neuron's as the loose test allows.
        if result is None or is_duplicate(found_clusters, result) or is_duplicate(rejected, result, cos=SAME_ROW):
            return False
        found_clusters.append(result)
        print("cluster %d of %d: %d duals" % (len(found_clusters), width, len(result['cluster'])), flush=True)
        save()
        if len(found_clusters) == width:
            # Checked once the layer looks complete: only then are the other clusters' hyperplanes all there.
            drop_disguised()
        return len(found_clusters) == width

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(layer, prefix_files, duals_dir)) as executor:
        submit(executor)
        while futures:
            done = next(as_completed(futures))
            done_with(dual_key(seeds[futures.pop(done)]))
            if take(done.result()):
                # Every neuron of the layer is accounted for: the remaining seeds can only be duals of deeper layers.
                for future in futures:
                    future.cancel()
                break
            # feed in another file of seeds as the scan gets deep into the current ones
            if next_seed >= len(seeds) - max_active and files_loaded < len(dual_files) and files_loaded <= FILES_AT_START + next_seed // 1000:
                add_seeds(files_loaded)
                files_loaded += 1
            submit(executor)

        if len(found_clusters) < width:
            # the seeds ran out first: the layer never looked complete, check it now
            drop_disguised()
        if len(found_clusters) < width:
            # Second pass, for neurons with few duals: every dual on none of the hyperplanes found (nor rejected) is a
            # seed once more, and scans all of those in full instead of giving up after 1000.  Most duals belong to the
            # neurons found, so a rare neuron's duals are far denser among the rest.
            rest = []
            known = found_clusters + rejected
            for f in dual_files:
                for d in load_seeds(f):
                    h = prefix.forward(torch.tensor(np.array(d[1])).to(DEVICE)).cpu().numpy()[0]
                    if not any(abs(h @ c['weight'] + c['bias']) < 1e-5 for c in known):
                        rest.append(d)
            fd, rest_path = tempfile.mkstemp(suffix=".p")
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(rest, fh)
            print("   second pass: %d neurons missing, %d duals on none of the hyperplanes found"
                  % (width - len(found_clusters), len(rest)), flush=True)
            futures = {}
            for i in range(len(rest)):
                if dual_key(rest[i]) in tried_second:
                    continue                           # tried before this run was interrupted
                args = (SECOND_PASS_ID + i, None, found_clusters + rejected, dual_files, i % num_workers, i % num_workers,
                        (rest_path, i))
                futures[executor.submit(process_single_candidate, args)] = i
            for done in as_completed(futures):
                done_with(dual_key(rest[futures[done]]), second=True)
                if take(done.result()):
                    for future in futures:
                        future.cancel()
                    break
            os.remove(rest_path)
    if rejected:
        print("   %d clusters dropped as deeper neurons" % len(rejected), flush=True)
    save()
    return found_clusters
