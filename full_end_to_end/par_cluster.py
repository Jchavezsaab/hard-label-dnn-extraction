# Groups the dual points of one layer by neuron.  Every dual point is tried as the seed of a cluster: the other duals
# are scanned and those consistent with it (recover_weights.is_consistent) are added; a cluster is accepted once the
# whole set solves to one hyperplane with a common bias.  Seeds scan in parallel, each worker with its own copy of the
# prefix.  cluster_layer() returns a list of {"cluster": [duals], "weight": w, "bias": b} and how many clusters it dropped
# as deeper neurons in disguise.
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
FIT_MAX = 1e-6             # a cluster's hyperplane fits its duals if its smallest singular value is below FIT_MAX * the largest
FIT_MAX_ZEROED = 1e-7      # the same for a cluster with weights set to 0: fewer unknowns, so a stricter fit (settle_zeroed)
BIAS_MIN = 1e-5            # a cluster whose hyperplane passes this close to the origin is degenerate (common_bias)
EXTRA_PER_INPUT = 4        # duals added to a cluster per earlier-layer neuron it was missing (settle_zeroed)
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
    pre = prefix.forward_nolastrelu(torch.tensor(np.stack([z[1] for z in x])).to(DEVICE))
    reject = torch.stack([torch.any(torch.abs(z.cpu()) < 1e-5, 1) for z in pre]).any(0)
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
                if S[-1] > FIT_MAX * S[0]:
                    logging.info(f"[W{worker_id}] Invalid cluster")
                    break
                
                bias = common_bias(maybe, soln, prefix)
                if bias is not None:
                    logging.info(f"[W{worker_id}] Found good cluster {cluster_id} of size {len(maybe)}")
                    result = {"cluster": maybe, "weight": soln, "bias": bias}
                    if np.any(soln == 0):
                        result = settle_zeroed(result, dual_files, random_seed, layer, prefix)
                    return result
                else:
                    logging.info(f"[W{worker_id}] Wrong bias. This is very weird.")
                    break
            else:
                last_try_size *= 2
    
    return None


def common_bias(maybe, soln, prefix):
    """The bias every dual of the cluster agrees on (each lies on the hyperplane), or None if they disagree.

    Also None for a bias within BIAS_MIN of 0: a hyperplane through the origin (every earlier-layer neuron off) contains
    every point where the few inputs it weighs are off, so duals of many different neurons fit it without telling it
    anything; a real neuron's bias is not that close to 0."""
    biases = np.array([(prefix.forward(torch.tensor(np.array(m[1])).to(DEVICE)).cpu().numpy() @ soln)[0] for m in maybe])
    med_bias = np.median(biases)
    if np.all(np.abs(biases - med_bias) < 1e-4) and abs(med_bias) >= BIAS_MIN:
        return -med_bias
    return None


def settle_zeroed(result, dual_files, random_seed, layer, prefix):
    """The cluster's solution has weights set to 0: earlier-layer neurons that are off at every one of its duals.

    Goes over every dual of the pool and collects those consistent with the cluster: where every zeroed input is off, a
    dual is consistent iff it lies on the cluster's hyperplane; where one is on, the pairwise test that grows clusters
    decides.  Where a missing neuron is on at some of them, a few are added and the cluster is solved again; a weight
    stays 0 only if its neuron is off at every consistent dual.  Returns the cluster with "zeroed" (those inputs) and
    "consistent_duals" (how many were found), or None if the enlarged cluster no longer solves or if, with weights
    still zeroed, it fits worse than FIT_MAX_ZEROED or its hyperplane is not determined without each one of its duals
    (determined_without_each)."""
    maybe, soln = result["cluster"], result["weight"]
    zeroed = np.flatnonzero(soln == 0)
    extras = {int(k): [] for k in zeroed}
    consistent = 0
    for b in load_dual_batch(dual_files, 0, random_seed, layer):
        if any(b is m for m in maybe):
            continue
        h = prefix.forward(torch.tensor(np.array(b[1])).to(DEVICE)).cpu().numpy()[0]
        if not np.any(h[zeroed] > 1e-4):
            # every weight that matters at b is known: b is consistent iff it lies on the hyperplane (no costly test)
            if abs(h @ soln + result["bias"]) < 1e-4:
                consistent += 1
            continue
        S = None
        for m in maybe[:4]:                        # duals too close to m cannot be compared with it: try another member
            S = is_consistent((m, b), prefix, layer=layer, do_return_soln=False)
            if S is not None:
                break
        if type(S) != np.float64 or not S < 1e-5:
            continue
        consistent += 1
        for k in zeroed:
            if h[k] > 1e-4 and len(extras[int(k)]) < EXTRA_PER_INPUT:
                extras[int(k)].append(b)

    added = [b for k in extras for b in extras[k]]
    if added:
        maybe = maybe + [b for i, b in enumerate(added) if not any(b is c for c in added[:i])]
        S, soln = is_consistent(maybe, prefix, layer=layer, do_return_soln=True)
        if type(S) != np.ndarray or S[-1] > FIT_MAX * S[0]:
            return None
        bias = common_bias(maybe, soln, prefix)
        if bias is None:
            return None
        result = {"cluster": maybe, "weight": soln, "bias": bias}
    result["zeroed"] = np.flatnonzero(result["weight"] == 0).tolist()
    result["consistent_duals"] = consistent
    if result["zeroed"]:
        # Without those columns there are few unknowns, and a handful of duals of different neurons can share a
        # hyperplane by accident: ask for a tighter fit, and for a hyperplane that is overdetermined by its duals.
        S, _ = is_consistent(result["cluster"], prefix, layer=layer, do_return_soln=True)
        if type(S) != np.ndarray or S[-1] > FIT_MAX_ZEROED * S[0]:
            logging.info("Cluster with zeroed weights %s rejected: fit above FIT_MAX_ZEROED" % result["zeroed"])
            return None
        if not determined_without_each(result["cluster"], prefix, layer):
            logging.info("Cluster with zeroed weights %s rejected: its hyperplane hinges on single duals" % result["zeroed"])
            return None
    return result


def determined_without_each(maybe, prefix, layer):
    """Whether the cluster's hyperplane is still determined (a single null direction) with any one of its duals left out."""
    for i in range(len(maybe)):
        S, _ = is_consistent(maybe[:i] + maybe[i + 1:], prefix, layer=layer, do_return_soln=True)
        if type(S) != np.ndarray or S[-2] <= 1e-2:
            return False
    return True


def disguised(clusters, prefix):
    """Indices of the clusters whose duals all lie in one on/off region of the other clusters' hyperplanes.

    A neuron of a deeper layer has a flat zero set behind the prefix wherever this layer's on/off pattern stays the same,
    so a cluster of its duals from a single region looks exactly like a neuron of this layer.  The duals of a real neuron
    of this layer spread over many regions of its own layer."""
    found = []
    for i, c in enumerate(clusters):
        others = [o for j, o in enumerate(clusters) if j != i]
        if len(c["cluster"]) < 2 or not others:
            continue
        H = prefix.forward(torch.tensor(np.array([d[1] for d in c["cluster"]])).to(DEVICE)).cpu().numpy()
        W = np.array([o["weight"] for o in others])
        b = np.array([o["bias"] for o in others])
        if len({tuple(row) for row in (H @ W.T + b > 0)}) == 1:
            found.append(i)
    return found


def is_duplicate(found_clusters, result):
    return any(abs(np.dot(other['weight'], result['weight'])) > 0.8 for other in found_clusters)


def cluster_layer(layer, duals_dir, prefix_files, num_workers, width):
    """Returns (the clusters found, how many were dropped as deeper neurons in disguise); stops as soon as `width` (the
    number of neurons in the layer) clusters have been found that pass the disguise check (see disguised)."""
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
    dropped = []               # clusters of deeper neurons: their seeds are skipped and they are not accepted again
    max_active = num_workers * 2
    next_seed = 0
    futures = {}

    def submit(executor):
        nonlocal next_seed
        while len(futures) < max_active and next_seed < len(seeds):
            worker = next_seed % num_workers
            args = (next_seed, seeds[next_seed], found_clusters + dropped, dual_files, worker, worker)
            futures[executor.submit(process_single_candidate, args)] = next_seed
            next_seed += 1

    def drop_disguised():
        """Move the clusters of deeper neurons from found_clusters to dropped; whether there were any."""
        bad = disguised(found_clusters, prefix)
        for i in reversed(bad):
            c = found_clusters.pop(i)
            dropped.append(c)
            print("   note: dropped a cluster of %d duals: all of them lie in one on/off region of the other neurons of this"
                  " layer, so it is a deeper neuron (%d of %d clusters left)" % (len(c["cluster"]), len(found_clusters), width),
                  flush=True)
        return bool(bad)

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(layer, prefix_files, duals_dir)) as executor:
        submit(executor)
        while futures:
            done = next(as_completed(futures))
            futures.pop(done)
            result = done.result()
            if result is not None and not is_duplicate(found_clusters + dropped, result):
                found_clusters.append(result)
                print("cluster %d of %d: %d duals" % (len(found_clusters), width, len(result['cluster'])), flush=True)
                if result.get("zeroed"):
                    print("   note: weights on inputs %s set to 0: those earlier-layer neurons are off at every one of the %d"
                          " duals consistent with this neuron" % (result["zeroed"], result["consistent_duals"]), flush=True)
                if len(found_clusters) == width and not drop_disguised():
                    # Every neuron of the layer is accounted for: the remaining seeds can only be duals of deeper layers.
                    for future in futures:
                        future.cancel()
                    break
            # feed in another file of seeds as the scan gets deep into the current ones
            if next_seed >= len(seeds) - max_active and files_loaded < len(dual_files) and files_loaded <= FILES_AT_START + next_seed // 1000:
                seeds.extend(load_seeds(dual_files[files_loaded]))
                files_loaded += 1
            submit(executor)
    if len(found_clusters) < width:
        drop_disguised()           # out of seeds before the width was reached: check what was found all the same
    return found_clusters, len(dropped)
