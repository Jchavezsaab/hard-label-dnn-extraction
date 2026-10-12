# Groups the dual points of one layer by neuron.  Every dual point is tried as the seed of a cluster: the other duals
# are scanned and those consistent with it (recover_weights.is_consistent) are added; a cluster is accepted once the
# whole set solves to one hyperplane with a common bias.  Seeds scan in parallel, each worker with its own copy of the
# prefix.  cluster_layer() returns a list of {"cluster": [duals], "weight": w, "bias": b}.
import logging
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import torch
from utils import DEVICE
from recover_weights import is_consistent, Prefix

multiprocessing.set_start_method('spawn', force=True)

BASE_SEED = 10000
MIN_CLUSTER = 3
MIN_CLUSTER_ZEROED = 12    # a cluster whose solution sets weights to 0 must grow to this many duals before it counts
SPLIT_MIN = 0.25           # a cluster split by fewer than this fraction of the others' hyperplanes is a deeper neuron (disguised)
IDENTITY_MAX = 1e-8        # a hyperplane that moves off its duals by less than this (relative) holds everywhere (prefix_identity)
FILES_AT_START = 5         # dual files loaded up front; one more per 1000 seeds tried

# set in every worker by init_worker
_worker_prefix = None
_worker_layer = None
_duals_dir = None


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



def load_dual_batch(dual_files, start_idx, random_seed, layer):
    """Generator that yields every dual once, starting at a file offset derived
    from start_idx (wrapping around so few files still get fully searched)"""
    first = (start_idx // 10000) % len(dual_files)  # Rough estimate of duals per file
    remaining = list(range(first, len(dual_files))) + list(range(first))
    current_file_duals = []
    dual_idx = 0

    while remaining or dual_idx < len(current_file_duals):
        # Load next file if we've exhausted current one
        if dual_idx >= len(current_file_duals):
            f = dual_files[remaining.pop(0)]
            x = pickle.load(open(os.path.join(_duals_dir, f), "rb"))
            random.Random(BASE_SEED + random_seed).shuffle(x)
            current_file_duals = reject_on_earlier_layer(x, _worker_prefix, layer)
            dual_idx = 0
            if not current_file_duals:  # Skip empty files
                continue
        
        # Yield duals from current file
        while dual_idx < len(current_file_duals):
            yield current_file_duals[dual_idx]
            dual_idx += 1



def process_single_candidate(args):
    """Process a single candidate neuron to find a cluster"""
    cluster_id, a, found_clusters, dual_files, random_seed, worker_id = args
    
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
    
    logging.info(f"Going {dual_files}")
    for j, b in enumerate(load_dual_batch(dual_files, cluster_id * 100, random_seed, layer)):
        if b is a:
            logging.info(f"[W{worker_id}] Skip adding the same neuron twice")
            continue
        if j > 1000 and len(maybe) == 1:
            logging.info(f"[W{worker_id}] Abort {cluster_id} because <1000")
            break
        if len(maybe)**2/(j+1) < .0001:
            logging.info(f"[W{worker_id}] Abort {cluster_id} because rare")
            break

        S = is_consistent((a, b), prefix, layer=layer, do_return_soln=False)
        
        if type(S) == np.float64 and S < 1e-5:
            an1, an2 = a[3:]          # the boundary normals on either side of each dual, stored by the walk
            bn1, bn2 = b[3:]
            if an1 @ bn1 > .8 or an1 @ bn2 > .8 or an2 @ bn1 > .8 or an2 @ bn2 > .8:
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
                    logging.info(f"[W{worker_id}] Invalid cluster")
                    break
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
                    logging.info(f"[W{worker_id}] Found good cluster {cluster_id} of size {len(maybe)}")
                    return result
                else:
                    logging.info(f"[W{worker_id}] Wrong bias. This is very weird.")
                    break
            else:
                last_try_size *= 2
    
    return None


def prefix_identity(result, prefix, directions=4):
    """Whether the cluster's hyperplane holds around its duals in every direction, not only on them.

    Where the earlier-layer outputs that are on depend on fewer neurons further down than there are of them, they
    satisfy an exact linear relation everywhere in those regions.  Duals of any neurons there fit that relation exactly
    and with a common bias, so they pass as a cluster (most easily with weights set to 0, which leave few unknowns); a
    real neuron's hyperplane is left as soon as a point moves off it.  Every move keeps the dual's own on/off pattern
    (forward_around), so its size does not matter."""
    w = result["weight"]
    rng = np.random.default_rng(0)
    for d in result["cluster"]:
        x0 = np.asarray(d[1], dtype=np.float64)
        x = np.vstack([x0, x0 + rng.standard_normal((directions, x0.size))])
        h = prefix.forward_around(torch.tensor(x).to(DEVICE)).cpu().numpy()
        moved = h[1:] - h[0]
        size = np.linalg.norm(moved, axis=1) * np.linalg.norm(w)
        if np.any(np.abs(moved @ w) > IDENTITY_MAX * size):
            return False
    return True


def is_duplicate(found_clusters, result):
    """Whether result is a neuron already found.  Weights set to 0 (exactly 0.0) are unknown, not zero, and two clusters
    of one neuron may zero different ones: they are compared on the weights both of them know."""
    w = result['weight']
    for other in found_clusters:
        known = (w != 0) & (other['weight'] != 0)
        a, b = w[known], other['weight'][known]
        if len(a) and abs(a @ b) > 0.8 * np.linalg.norm(a) * np.linalg.norm(b):
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


def cluster_layer(layer, duals_dir, prefix_files, num_workers, width):
    """Returns the clusters found; stops as soon as `width` (the number of neurons in the layer) have been found that
    are not deeper neurons in disguise (see disguised)."""
    prefix = Prefix(prefix_files).to(DEVICE)
    dual_files = sorted(os.listdir(duals_dir))
    random.Random(BASE_SEED).shuffle(dual_files)

    def load_seeds(f):
        x = pickle.load(open(os.path.join(duals_dir, f), "rb"))
        random.Random(BASE_SEED).shuffle(x)
        return reject_on_earlier_layer(x, prefix, layer)

    seeds = []
    for f in dual_files[:FILES_AT_START]:
        seeds.extend(load_seeds(f))
    files_loaded = FILES_AT_START

    found_clusters = []
    rejected = []              # deeper neurons in disguise: their duals are skipped and they are not accepted again
    max_active = num_workers * 2
    next_seed = 0
    futures = {}

    def submit(executor):
        nonlocal next_seed
        while len(futures) < max_active and next_seed < len(seeds):
            worker = next_seed % num_workers
            args = (next_seed, seeds[next_seed], found_clusters + rejected, dual_files, worker, worker)
            futures[executor.submit(process_single_candidate, args)] = next_seed
            next_seed += 1

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(layer, prefix_files, duals_dir)) as executor:
        submit(executor)
        while futures:
            done = next(as_completed(futures))
            futures.pop(done)
            result = done.result()
            if result is not None and not is_duplicate(found_clusters + rejected, result):
                found_clusters.append(result)
                print("cluster %d of %d: %d duals" % (len(found_clusters), width, len(result['cluster'])), flush=True)
                if len(found_clusters) == width:
                    # Checked once the layer looks complete: only then are the other clusters' hyperplanes all there.
                    for i in reversed(disguised(found_clusters, prefix)):
                        print("   dropped cluster of %d duals: all in one on/off region of this layer, a deeper neuron"
                              % len(found_clusters[i]['cluster']), flush=True)
                        rejected.append(found_clusters.pop(i))
                if len(found_clusters) == width:
                    # Every neuron of the layer is accounted for: the remaining seeds can only be duals of deeper layers.
                    for future in futures:
                        future.cancel()
                    break
            # feed in another file of seeds as the scan gets deep into the current ones
            if next_seed >= len(seeds) - max_active and files_loaded < len(dual_files) and files_loaded <= FILES_AT_START + next_seed // 1000:
                seeds.extend(load_seeds(dual_files[files_loaded]))
                files_loaded += 1
            submit(executor)
    if rejected:
        print("   %d clusters dropped as deeper neurons" % len(rejected), flush=True)
    return found_clusters
