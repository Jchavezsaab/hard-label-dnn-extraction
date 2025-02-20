import re
import os
import sys
import pickle
from utils import *
from collections import defaultdict

from recover_weights import is_consistent, CIFAR10NetPrefix, transfer_weights

def cheat_cluster(layer):
    duals = []
    root = 'exp/1/'
    for f in sorted(os.listdir(root)):
        print(f)
        x = pickle.load(open(os.path.join(root,f),"rb"))
        duals.extend(x)

    cheating = defaultdict(list)
    for idx,(left,middle,right) in enumerate(duals):
        if idx%1000 == 0:
            print(idx, '/', len(duals))
        diff = cheat_neuron_diff_cuda(left, right)
        if len(diff) == 1:
            if diff[0]//DIM == layer:
                cheating[diff[0]].append((left, middle, right))
    
    pickle.dump(cheating, open("exp/1-cluster-%d.p"%layer, "wb"))

def refine_cluster(maybe, layer, prefix):
    maybe = np.array(maybe)
    points = np.zeros(len(maybe))
    for _ in range(10):
        order = np.arange(len(maybe))
        random.shuffle(order)
        for i in range(0, len(order)-(len(order)%3), 3):
            ok = is_consistent([maybe[x] for x in order[i:i+3]], prefix, layer, False)
            print("ok?", ok)
            if ok is not None and ok > 1e-5:
                points[order[i:i+3]] += 1
    
    maybe = maybe[points < 6]
    return maybe

def cluster_slow(layer):
    prefix = CIFAR10NetPrefix(layer).cuda()
    transfer_weights(cheat_net_cpu, prefix)

    duals = []
    root = 'exp/1/'
    for f in sorted(os.listdir(root)):
        print(f)
        x = pickle.load(open(os.path.join(root,f),"rb"))
        duals.extend(x)

    output = {}
    print("LAYER", layer)
    for cluster_id,a in enumerate(duals[:1000]):
        print("idx", cluster_id)
        maybe = [a]
        for j,b in enumerate(duals):
            if j > 1000 and len(maybe) < 3000/j:
                print("Too low rate; break", j)
                break
            S = is_consistent((a,b), prefix, False)
            # Necessary to tune 1e-5 for the appropriate TPR/FPR tradeoff
            if type(S) == np.float64 and S < 1e-5:
                print("Got close")
                print(S, cheat_neuron_diff_cuda(a[0], a[2]), cheat_neuron_diff_cuda(b[0], b[2]))
                maybe.append(b)
        print("Found set size", len(maybe))

        print("Before refine")
        for i in range(len(maybe)):
            idx = cheat_neuron_diff_cuda(maybe[i][0], maybe[i][2])
            print(idx)
        
        # OPTIONAL: increase precision, reduce recall
        if len(maybe) > 2:
            maybe = refine_cluster(maybe, layer, prefix)
        else:
            continue

        print("After refine")
        for i in range(len(maybe)):
            idx = cheat_neuron_diff_cuda(maybe[i][0], maybe[i][2])
            print(idx)

        print("WRITING", cluster_id)
        output[cluster_id] = maybe
        pickle.dump(output, open("exp/1-cluster-%d.p"%layer, "wb"))

if len(sys.argv) > 2 and sys.argv[2] == 'slow':
    cluster_slow(int(sys.argv[1]))
else:
    cheat_cluster(int(sys.argv[1]))
