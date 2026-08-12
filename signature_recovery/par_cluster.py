import re
import os
import sys
import pickle
from utils import *
from collections import defaultdict
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import multiprocessing
import sys
import logging
from multiprocessing import Queue
from logging.handlers import QueueHandler, QueueListener

from recover_weights import is_consistent, CIFAR10NetPrefix, transfer_weights

# Set spawn method for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)

PREFIX = '2g-cluster'
BASE_SEED = 10000

# Global variables for worker process initialization
_worker_prefix = None
_worker_layer = None
_log_queue = None

def init_worker(layer, log_queue):
    """Initialize CUDA model in each worker process"""
    global _worker_prefix, _worker_layer, _log_queue
    
    # Set up logging in worker
    _log_queue = log_queue
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(QueueHandler(log_queue))
    
    _worker_layer = layer
    _worker_prefix = CIFAR10NetPrefix(layer).to(DEVICE)
    transfer_weights(cheat_net_cpu, _worker_prefix)


def reject_on_earlier_layer(x, prefix, layer):
    """Filter out points that are rejected by earlier layers"""
    if layer == 0: return x
    out = prefix.forward_nolastrelu(torch.tensor(np.stack([z[1] for z in x])).to(DEVICE)).cpu()
    reject = torch.any(torch.abs(out) < 1e-5, (0,2))
    x = [z for z, r in zip(x, reject) if not r]
    return x


def load_dual_batch(dual_files, start_idx, root, random_seed, layer):
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
            x = pickle.load(open(os.path.join(root, f), "rb"))
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

    #if len(cheat_neuron_diff_cuda(a[0], a[2])) != 1 or cheat_neuron_diff_cuda(a[0], a[2]) >= 8:
    #    return None
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
    last_try_size = 2
    
    # Load duals for searching
    root = 'exp/1/'

    logging.info(f"Going {dual_files}")
    for j, b in enumerate(load_dual_batch(dual_files, cluster_id * 100, root, random_seed, layer)):
        #logging.info(f"J={j} diff {cheat_neuron_diff_cuda(b[0], b[2])}")
        if b is a:
            logging.info(f"[W{worker_id}] Skip adding the same neuron twice")
            continue
        if j > 1000 and len(maybe) == 1:
            logging.info(f"[W{worker_id}] Abort {cluster_id} because <1000")
            break
        if len(maybe)**2/(j+1) < .0001:
            logging.info(f"[W{worker_id}] Abort {cluster_id} because rare")
            break

        pickle.dump((a, b), open("/tmp/ab.p","wb"))
        S = is_consistent((a, b), prefix, layer=layer, do_return_soln=False)
        
        if type(S) == np.float64 and S < 1e-5:
            an1, an2 = a[3:] or (get_normal(a[0]), get_normal(a[2]))  # flank normals stored by the walk, if any
            bn1, bn2 = b[3:] or (get_normal(b[0]), get_normal(b[2]))
            if an1 @ bn1 > .8 or an1 @ bn2 > .8 or an2 @ bn1 > .8 or an2 @ bn2 > .8:
                continue
            
            logging.info(f"[W{worker_id}] Added consistent neuron to {cluster_id} {j} {len(maybe)}")
            maybe.append(b)
        
        if len(maybe) >= last_try_size:
            if len(maybe) > 100:
                logging.info(f"[W{worker_id}] CONFUSED THAT ITS SO BIG")
                pickle.dump(maybe, open(f"/tmp/weird-{cluster_id}.p", "wb"))
                break
            
            S, soln = is_consistent(maybe, prefix, layer=layer, do_return_soln=True)
            logging.info(f"[W{worker_id}] Check consistent for {cluster_id} {S}")
            
            if type(S) == np.ndarray:
                if S[-1] > 1e-6:  # 1e-5 upstream; one bisected dual ~1e-6 off its plane costs the row ~S[-1]/4, too coarse for sign recovery
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
                    pickle.dump(maybe, open(f"/tmp/badbias-{cluster_id}.p", "wb"))
                    break
            else:
                last_try_size *= 2
    
    return None


def cluster_single(layer, _):
    """Single-threaded version for debugging/testing"""
    global _worker_prefix, _worker_layer

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Initialize model
    prefix = CIFAR10NetPrefix(layer).to(DEVICE)
    transfer_weights(cheat_net_cpu, prefix)
    _worker_prefix = prefix
    _worker_layer = layer

    # Load existing clusters
    found_clusters = []
    print(f"Starting fresh for layer {layer}")

    # Load duals
    root = 'exp/1/'
    dual_files = sorted(os.listdir(root))
    random.Random(BASE_SEED).shuffle(dual_files)

    all_duals = []
    for f in dual_files[:5]:
        x = pickle.load(open(os.path.join(root, f), "rb"))
        random.Random(BASE_SEED).shuffle(x)
        x = reject_on_earlier_layer(x, prefix, layer)
        all_duals.extend(x)

    print(f"Processing {len(all_duals)} candidates")

    # Process each candidate sequentially
    for cluster_id, a in enumerate(all_duals):
        result = process_single_candidate((cluster_id, a, found_clusters.copy(), dual_files, 0, 0))
        if result is not None:
            found_clusters.append(result)
            print(f"Found cluster {len(found_clusters)}")
            pickle.dump(found_clusters, open(f"exp/{PREFIX}-{layer}.p", "wb"))

    print(f"Final: {len(found_clusters)} clusters")


def cluster_par(layer, num_workers):
    """Multi-process version using ProcessPoolExecutor"""
    # Set up logging queue
    log_queue = Queue()
    
    # Set up logging in main process
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    listener = QueueListener(log_queue, handler)
    listener.start()
    
    # Initialize prefix for main process (for loading duals)
    prefix = CIFAR10NetPrefix(layer).to(DEVICE)
    transfer_weights(cheat_net_cpu, prefix)
    
    # Load existing clusters
    try:
        found_clusters = pickle.load(open(f"exp/{PREFIX}-{layer}.p", "rb"))
        print(f"Loaded {len(found_clusters)} existing clusters for layer {layer}")
    except:
        found_clusters = []
        print(f"Starting fresh for layer {layer}")
    
    # Prepare dual files list
    root = 'exp/1/'
    dual_files = sorted(os.listdir(root))
    random.Random(BASE_SEED).shuffle(dual_files)
    
    # Load initial batch of duals for main process to distribute
    all_duals = []
    for i, f in enumerate(dual_files[:5]):  # Start with first 5 files
        x = pickle.load(open(os.path.join(root, f), "rb"))
        random.Random(BASE_SEED).shuffle(x)
        x = reject_on_earlier_layer(x, prefix, layer)
        all_duals.extend(x)
        print(f"Loaded {len(all_duals)} duals so far")
    
    # Create process pool
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(layer, log_queue)) as executor:
        # Submit initial batch of jobs
        futures = {}
        cluster_id = 0
        active_jobs = 0
        max_active = num_workers * 2  # Keep pipeline full
        
        # Submit initial jobs
        for i in range(min(max_active, len(all_duals))):
            if cluster_id < len(all_duals):
                a = all_duals[cluster_id]
                future = executor.submit(process_single_candidate, 
                                       (cluster_id, a, found_clusters.copy(), dual_files, cluster_id % num_workers, cluster_id % num_workers))
                futures[future] = cluster_id
                cluster_id += 1
                active_jobs += 1
        
        print(f"Submitted initial {active_jobs} jobs")
        
        # Process results as they complete
        try:
            while futures or cluster_id < len(all_duals):
                # Wait for at least one job to complete
                done_futures = []
                try:
                    for future in as_completed(futures, timeout=1):
                        done_futures.append(future)
                        break  # Process one at a time to keep pipeline full
                except TimeoutError:
                    # No jobs completed in this second, continue to check again
                    continue

                # Process completed futures
                for future in done_futures:
                    cid = futures.pop(future)
                    active_jobs -= 1
                    
                    try:
                        result = future.result()
                        if result is not None:
                            # Check if this is truly new (double-check against current list)
                            is_duplicate = False
                            for other in found_clusters:
                                if np.dot(other['weight'], result['weight']) > 0.8:
                                    print(f"Cluster {cid} was duplicate after all")
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                found_clusters.append(result)
                                print(f"Added new cluster from job {cid}, total: {len(found_clusters)}")
                                
                                # Save periodically
                                pickle.dump(found_clusters, open(f"exp/{PREFIX}-{layer}.p", "wb"))
                                print(f"Saved {len(found_clusters)} clusters to disk")

                    except Exception as e:
                        print(f"Job {cid} failed with error: {e}")
                        raise
                    
                
                # Submit new jobs to maintain active job count
                while active_jobs < max_active and cluster_id < len(all_duals):
                    a = all_duals[cluster_id]
                    future = executor.submit(process_single_candidate,
                                           (cluster_id, a, found_clusters.copy(), dual_files, cluster_id % num_workers, cluster_id % num_workers))
                    futures[future] = cluster_id
                    cluster_id += 1
                    active_jobs += 1
                
                # Load more duals if needed
                if cluster_id >= len(all_duals) - max_active and len(dual_files) > 5:
                    print(f"Loading more duals, currently at {cluster_id}/{len(all_duals)}")
                    next_file_idx = 5 + (cluster_id // 1000)
                    if next_file_idx < len(dual_files):
                        f = dual_files[next_file_idx]
                        x = pickle.load(open(os.path.join(root, f), "rb"))
                        random.Random(BASE_SEED).shuffle(x)
                        x = reject_on_earlier_layer(x, prefix, layer)
                        all_duals.extend(x)
                        print(f"Loaded more duals, total: {len(all_duals)}")
        
        except KeyboardInterrupt:
            print("\nInterrupted, cancelling remaining jobs...")
            for future in futures:
                future.cancel()
        
        # Final save
        pickle.dump(found_clusters, open(f"exp/{PREFIX}-{layer}.p", "wb"))
        print(f"Final save: {len(found_clusters)} clusters")
    
    # Stop the logging listener
    listener.stop()


if __name__ == "__main__":
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    print(f"Running cluster_par for layer {layer} with {num_workers} workers")
    cluster_par(layer, num_workers)
