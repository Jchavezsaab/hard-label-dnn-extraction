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
                if S[-1] > 1e-6:
                    logging.info(f"[W{worker_id}] Invalid cluster")
                    break
                
                biases = []
                for m in maybe:
                    bias = (prefix.forward(torch.tensor(np.array(m[1])).to(DEVICE)).cpu().numpy() @ soln)[0]
                    biases.append(bias)
                med_bias = np.median(biases)
                
                if np.all(np.abs(biases - med_bias) < 1e-4):
                    logging.info(f"[W{worker_id}] Found good cluster {cluster_id} of size {len(maybe)}")
                    return {"cluster": maybe, "weight": soln, "bias": -med_bias}
                else:
                    logging.info(f"[W{worker_id}] Wrong bias. This is very weird.")
                    break
            else:
                last_try_size *= 2
    
    return None


def is_duplicate(found_clusters, result):
    return any(abs(np.dot(other['weight'], result['weight'])) > 0.8 for other in found_clusters)


def cluster_layer(layer, duals_dir, prefix_files, num_workers, width):
    """Returns the clusters found; stops as soon as `width` (the number of neurons in the layer) have been found."""
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
    max_active = num_workers * 2
    next_seed = 0
    futures = {}

    def submit(executor):
        nonlocal next_seed
        while len(futures) < max_active and next_seed < len(seeds):
            worker = next_seed % num_workers
            args = (next_seed, seeds[next_seed], list(found_clusters), dual_files, worker, worker)
            futures[executor.submit(process_single_candidate, args)] = next_seed
            next_seed += 1

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(layer, prefix_files, duals_dir)) as executor:
        submit(executor)
        while futures:
            done = next(as_completed(futures))
            futures.pop(done)
            result = done.result()
            if result is not None and not is_duplicate(found_clusters, result):
                found_clusters.append(result)
                print("cluster %d of %d: %d duals" % (len(found_clusters), width, len(result['cluster'])), flush=True)
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
    return found_clusters
