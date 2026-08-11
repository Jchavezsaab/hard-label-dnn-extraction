# TODO: ostill sometimes get bad things but they're only small clusters
# TODO: figure out why that happens still, perhaps decision boundary normal?


import re
import os
import sys
import pickle
from utils import *
from collections import defaultdict
import random

from recover_weights import is_consistent, CIFAR10NetPrefix, transfer_weights

def exp0():
    dats = pickle.load(open("exp/1/duals_00793806.p","rb"))
    prefix = CIFAR10NetPrefix(0).cuda()
    transfer_weights(cheat_net_cpu, prefix)

    r = []
    for idx,(a,x,b) in enumerate(dats):
        d = cheat_neuron_diff_cuda(a, b)
        if len(d) > 0 and d == [2]:
            r.append((a,x,b))
            print(idx)

    S, soln = is_consistent(r, prefix, layer=0, do_return_soln=True)
    print(S)
    print(soln)

    data = cheat_net_cpu.fc2.weight[2,:].cpu().detach().numpy()
    print(soln/data)
          

def exp():
    N = 643
    L = N//64
    ID = N%64

    dats = []
    for f in os.listdir("exp/1"):
        dats.extend(pickle.load(open("exp/1/"+f,"rb")))
        break
        
    prefix = CIFAR10NetPrefix(L).cuda()
    transfer_weights(cheat_net_cpu, prefix)

    r = []
    for idx,(a,x,b) in enumerate(dats):
        d = cheat_neuron_diff_cuda(a, b)
        if len(d) == 1 and d == [64*L+ID]:
            r.append((a,x,b))
            print(idx)
    if len(r) == 0:
        exit(0)

    S, soln = is_consistent(r, prefix, layer=L, do_return_soln=True)
    print(S)
    print(soln)

    data = cheat_net_cpu.fcs[L].weight[ID,:].cpu().detach().numpy()
    print(soln/data)

def exp_valid():
    N = 643
    L = N//64
    ID = N%64

    dats = []
    for f in os.listdir("exp/1"):
        dats.extend(pickle.load(open("exp/1/"+f,"rb")))
        
    prefix = CIFAR10NetPrefix(L).cuda()
    transfer_weights(cheat_net_cpu, prefix)

    r = []
    reject = []
    for idx,(a,x,b) in enumerate(dats):
        d = cheat_neuron_diff_cuda(a, b)
        if len(d) == 1 and d == [64*L+ID]:
            r.append((a,x,b))
            reject.extend(range(idx-20,idx+20))
            print('Take', idx)
            if len(r) >= 3:
                break
    reject = set(reject)
    if len(r) == 0:
        exit(0)

    for idx,other in enumerate(dats):
        if idx in reject: continue
        d = cheat_neuron_diff_cuda(other[0], other[2])
        S = is_consistent(r[:2]+ [other], prefix, layer=L)
        if S < 1e-8:
            print('s', S, len(d) == 1 and d == [64*L+ID], 'at', idx)

def exp_nsquare():

    dats = []
    for f in os.listdir("exp/1"):
        dats.extend(pickle.load(open("exp/1/"+f,"rb")))
        break
        

    for _ in range(1000):
        N = random.randint(0, 15*64)
        print("Choosing neuron", N)
        L = N//64
        ID = N%64

        prefix = CIFAR10NetPrefix(L).cuda()
        transfer_weights(cheat_net_cpu, prefix)
        
        r = []
        reject = None
        candidate = None
        for idx,(a,x,b) in enumerate(dats):
            d = cheat_neuron_diff_cuda(a, b)
            if len(d) == 1 and d == [64*L+ID]:
                print("Found candidate at idx", idx)
                candidate = (a,x,b)
                reject = range(idx-20,idx+20)
                break
        if candidate is None:
            continue
        reject = set(reject)

        for idx,other in enumerate(dats):  #
            #if idx in reject: continue
            #print('d', d)
            S = is_consistent([candidate, other], prefix, layer=L, deltacheck=True)
            if S < -3:
                d = cheat_neuron_diff_cuda(other[0], other[2])
                print('s', S, len(d) == 1 and d == [64*L+ID], 'at', idx)
            
    
exp_nsquare()
exit(0)

is_adjacent = {}

class DualIter:
    def __init__(self, duals=None, idx=0, dual_fs=None,
                 prefix=None, layer=None):
        if duals is None:
            self.duals = []
        else:
            self.duals = duals

        self.prefix = prefix
        self.layer = layer

        self.idx = 0
        self.root = 'exp/1/'

        if dual_fs is None:
            self.dual_fs = sorted(os.listdir(self.root))
        else:
            self.dual_fs = dual_fs

    def reject_on_earlier_layer(self, x):
        for layer in range(self.layer):
            out = self.prefix.forward_nolastrelu(torch.tensor([z[1] for z in x]).cuda()).cpu()
            print(out.shape)
            reject = torch.any(torch.abs(out) < 1e-5, 1)
            
            return [z for z,r in zip(x, reject) if not r]
        raise
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < len(self.duals):
            out = self.duals[self.idx]
            self.idx += 1
            return out

        print("Empty after", self.idx, "now gathering more")
        f = self.dual_fs.pop()
        x = pickle.load(open(os.path.join(self.root,f),"rb"))
        random.seed(0)
        random.shuffle(x)
        prior = None
        x = self.reject_on_earlier_layer(x)
        for y in x:
            self.duals.append(y)
            is_adjacent[(id(y),id(x))] = True
            is_adjacent[(id(x),id(y))] = True
            prior = y
        print("Done gather")
        return self.__next__()

    def clone_reset(self):
        return DualIter(self.duals, 0, self.dual_fs, self.prefix, self.layer)
    

def cluster_slow(layer):
    prefix = CIFAR10NetPrefix(layer).cuda()
    transfer_weights(cheat_net_cpu, prefix)

    duals_iter = DualIter(prefix=prefix, layer=layer)

    #found_clusters = []
    found_clusters = pickle.load(open("exp/1d-cluster-%d.p"%layer, "rb"))
    print("LAYER", layer)
    for cluster_id,a in enumerate(duals_iter):
        if cluster_id < 356: continue
        print("On ID", cluster_id, 'diff', cheat_neuron_diff_cuda(a[0], a[2]))


        is_prior_found = False
        for j,previous in enumerate(found_clusters):
            prior_examples = previous['cluster']

            prior_w = previous['weight']
            prior_b = previous['bias']

            if np.abs((prefix.forward(torch.tensor(np.array(a[1])).cuda()).cpu() @ prior_w)[0] + prior_b) < 1e-5:
                is_prior_found = True
                break
        if is_prior_found:
            print("Skip prior found", j)
            continue
        
        #'''
        maybe = [a]
        last_try_size = 2
        for j,b in enumerate(duals_iter.clone_reset()):
            #if cheat_neuron_diff_cuda(b[0], b[2]) < 64:
            #    raise

            
            if j > 1000 and len(maybe) == 1:
                print("Explored far enough, probably skip")
                break
            if len(maybe)**2/(j+1) < .001:
                print("Rate too low, aborting at", j)
                break



            S = is_consistent((a,b), prefix, layer=layer, do_return_soln=False)

            # Necessary to tune 1e-5 for the appropriate TPR/FPR tradeoff
            if type(S) == np.float64 and S < 1e-5:
                an1 = get_normal(a[0])
                bn1 = get_normal(b[0])
                an2 = get_normal(a[1])
                bn2 = get_normal(b[1])
                if an1 @ bn1 > .8 or an1 @ bn2 > .8 or an2 @ bn1 > .8 or an2 @ bn2 > .8:
                    print(j, "Too similar, abort",
                          an1 @ bn1, an1 @ bn2, an2 @ bn1, an2 @ bn2,
                          'diff', cheat_neuron_diff_cuda(b[0], b[2]))
                    if (id(a),id(b)) in is_adjacent:
                        print("Also adjacent")
                        continue

                if (id(a),id(b)) in is_adjacent:
                    raise

                
                print("Got close", j)
                print(S, cheat_neuron_diff_cuda(a[0], a[2]), cheat_neuron_diff_cuda(b[0], b[2]))
                maybe.append(b)
                
            if len(maybe) >= last_try_size:
                print("Before refine")
                for i in range(len(maybe)):
                    idx = cheat_neuron_diff_cuda(maybe[i][0], maybe[i][2])
                    print(idx,end=' ')
                print()
        
                S, soln = is_consistent(maybe, prefix, layer=layer, do_return_soln=True)

                if type(S) == np.ndarray:
                    if S[-1] > 1e-5:
                        print("Not all good together; abort")
                        break
                    biases = []
                    for m in maybe:
                        bias = (prefix.forward(torch.tensor(np.array(m[1])).cuda()).cpu().numpy() @ soln)[0]
                        biases.append(bias)
                    med_bias = np.median(biases)
                    print(biases)
                    assert np.all(np.abs(biases - med_bias) < 1e-4)
                    print("It's a good cluster. Adding to found_clusters at index", len(found_clusters))
                    found_clusters.append({"cluster": maybe, "weight": soln, "bias": -bias})

                    pickle.dump(found_clusters, open("exp/1d-cluster-%d.p"%layer, "wb"))
                    break
                else:
                    last_try_size *= 2

cluster_slow(0)
exit(0)

def run(cluster):
    layer = 0
    prefix = CIFAR10NetPrefix(layer).cuda()
    transfer_weights(cheat_net_cpu, prefix)

    idx = cheat_neuron_diff_cuda(cluster[0][0], cluster[0][2])
    S, vt = is_consistent(cluster, prefix, do_return_soln=True)
    factor = np.median(vt/cheat_net_cpu.fc1.weight[idx,:].cpu().detach().numpy())

    assert len(idx) == 1
    err = np.max(np.abs(vt/factor - cheat_net_cpu.fc1.weight[idx,:].cpu().detach().numpy()))
    print('ERROR', idx[0], err < 1e7, err)

for index in 'abcd':
    clusters = pickle.load(open(f"exp/1{index}-cluster-0.p", "rb"))
    for x in clusters:
        run(x)
    print(len(clusters))
    #extract()
