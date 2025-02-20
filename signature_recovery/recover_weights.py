import re
import os
import pickle
from utils import *
from collections import defaultdict
import sys

def intersect(left, right, nleft, nright):
    A = np.vstack((nleft, nright))
    b = np.array([np.dot(nleft, left), np.dot(nright, right)])

    # Find a particular solution
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Find the null space of A
    N = scipy.linalg.null_space(A, 1e-5)
    #print(A.shape)
    #U,S,V = np.linalg.svd(A)
    #print(U.shape, S.shape, V.shape)
    #print("V", V.shape)
    #rank = sum(1 for x in S if abs(x) > 1e-5)
    #N = V.T[:, rank:]

    
    return x0, N

# Function to generate random points on the n-2 dimensional subspace
def generate_points_on_subspace(x0, N, num_points=10):
    random_vectors = np.random.randn(N.shape[1], num_points)
    subspace_points = x0[:, np.newaxis] + N @ random_vectors
    return subspace_points.T


def vectorized_check_closest_pair_distance(points):
    # Extract the second coordinate from each point
    coords = np.array([p[1] for p in points])
    
    # Calculate pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sum(np.square(diff), axis=-1)
    
    # Set diagonal to infinity to ignore self-distances
    np.fill_diagonal(distances, -np.inf)
    
    # Find the minimum distance
    min_distance = np.max(distances)
    
    if min_distance < 1:
        return True
    else:
        return False


class CIFAR10NetPrefix(nn.Module):
    def __init__(self, layers):
        super(CIFAR10NetPrefix, self).__init__()
        if layers == 0:
            self.fcs = nn.Sequential()
        else:
            h = [DIM, DIM, 64]
            layers = [nn.Linear(DIM, h[layer]) for layer in range(layers-1)]
            self.fcs = nn.Sequential(*([nn.Linear(IDIM, DIM)] + layers))
        self.double()

    def relu_around(self, x):
        mask = (x[:1]>=0).to(torch.float64)
        return x * mask
        
    @torch.no_grad
    def forward_around(self, x):
        x = x.view(-1, IDIM)
        if len(self.fcs) == 0: return x
        for layer in self.fcs:
            x = self.relu_around(layer(x))
        return x

    @torch.no_grad
    def forward(self, x):
        x = x.view(-1, IDIM)
        if len(self.fcs) == 0: return x
        for layer in self.fcs:
            x = nn.functional.relu(layer(x))
        return x

def transfer_weights(source_model, target_model, source_prefix='', target_prefix='fcs'):
    target_state_dict = {}
    source_state_dict = source_model.state_dict()
    
    layer_count = 0
    while True:
        source_weight_key = f'{source_prefix}fc{layer_count+1}.weight'
        source_bias_key = f'{source_prefix}fc{layer_count+1}.bias'
        
        if source_weight_key not in source_state_dict:
            break
        
        target_weight_key = f'{target_prefix}.{layer_count}.weight'
        target_bias_key = f'{target_prefix}.{layer_count}.bias'

        target_state_dict[target_weight_key] = source_state_dict[source_weight_key]
        target_state_dict[target_bias_key] = source_state_dict[source_bias_key]

        layer_count += 1

    target_model.load_state_dict(target_state_dict, strict=False)

    return layer_count
    
def is_consistent_help(points, prefix, layer=0, do_return_soln=False, allow_close=False):
    samples = []

    # The points need to be in different linear regions to try and compare them
    if vectorized_check_closest_pair_distance(points) and not allow_close:
        return None, None # rejected
    
    if do_return_soln:
        print("Num points", len(points))
        mid = np.stack([x[1] for x in points])
        hiddens = prefix(torch.tensor(mid).cuda()).cpu().numpy()
        hiddens = (hiddens>1e-4)
        hits = hiddens.sum(0)
        order = np.argsort(hits)
        print("Hits", hits.shape)

        if np.min(hits) == 0 and layer > 0:
            print("Hit some zero times. Mean OK", np.mean(hits!=0))
            print(list(hits))
            return None, None
        points_subset = []
        hits = np.zeros([IDIM, DIM, DIM, DIM, 64][layer])
        
        for coord in order:
            if hits[coord] >= 4:
                continue
            for entry in np.where(hiddens[:, coord])[0][:2]:
                points_subset.append(points[entry])
                hits += hiddens[entry]
                
        points = points_subset

    for i, (left, x0, right) in enumerate(points):
        left = np.array(left)
        right = np.array(right)
        x0 = np.array(x0)

        nleft = get_normal(left)
        nright = get_normal(right)

        _, N = intersect(left, right, nleft, nright)
        points = generate_points_on_subspace(x0, N, DIM*2).tolist()

        points = np.concatenate(([x0], points), 0)
        
        points = prefix.forward_around(torch.tensor(points).cuda()).cpu()

        samples.append(points)

    samples = np.concatenate(samples, 0)

    all_zero = np.sum(np.sum(np.abs(samples),0)<1e-5)

    # We need to share at least 3 coordinates in common to try and compare
    # If we only have two there are enough free variables for anything to happen.
    shared_coords = np.sum(np.sum(np.abs(samples[::DIM*2]) > 1e-5,0) >= 2)
    #print('shared',shared_coords)
    if shared_coords <= 3:
        print("Reject")
        return None # rejected

    mean_point = np.mean(samples, axis=0)
    
    centered_samples = samples - mean_point

    if do_return_soln:
        U, S, Vt = np.linalg.svd(centered_samples)

        ans = Vt[-1]
        ans = norm(ans)
        

        return S, Vt[-1]

    tt = torch.tensor(centered_samples).double()
    S = torch.linalg.svdvals(tt).cpu().numpy()
    print(S)

    return S[len(S)-all_zero-1]

def is_consistent(points, prefix, layer=0, do_return_soln=False):
    try:
        return is_consistent_help(points, prefix, layer, do_return_soln)
    except MathIsHard:
        return None
        


def extract_weights(maybe, prefix, layer):
    if True:
        if True:
            if DEBUG:
                for i in range(len(maybe[:10])):
                    idx = cheat_neuron_diff(maybe[i][0], maybe[i][2])
                    print(i,idx,end='  ')
                print()

            print("Size", len(maybe))
            S, soln = is_consistent(maybe, prefix, layer, True)

            print('Singular values', S)
            if S is not None and S[-2] > 1e-2 and S[-1] < 1e-4:
                return soln

        
def dosteal(LAYER, cluster):
    prefix = CIFAR10NetPrefix(LAYER).cuda()
    transfer_weights(cheat_net_cpu, prefix)

    
    for cluster_id, maybe in sorted(cluster.items(), key=lambda x: len(x[1])):
        maybe = np.array(maybe)

        if True:
            print()
            print()
            print()
            print("CLUSTER ID", cluster_id)
            print("Recovering weight given", len(maybe), "dual points")
            maybe = maybe[:1200]
            soln = extract_weights(maybe, prefix, layer)

            print('Extracted weight vector', soln)

            # Compute error in recovering this layer
            if soln is not None:

                errs = []
                for maybe_neuron in range(DIM):
                    factor = np.median(soln/cheat_solution[LAYER][maybe_neuron, :])
                    errs.append(np.sum(np.abs(soln / factor - cheat_solution[LAYER][maybe_neuron, :])))
                if min(errs) < 1e-3:
                    print("Successfully extracted neuron", np.argmin(errs),
                          'with abs err', np.min(errs))
                else:
                    print("Failed to identify recovered neuron")
            else:
                print("Not enough to fully extract")
            
if __name__ == '__main__':
    layer = int(sys.argv[1])
    dosteal(layer, pickle.load(open("exp/1-cluster-%d.p"%layer,"rb")))
