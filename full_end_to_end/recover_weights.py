# Turns a cluster of dual points of one neuron into that neuron's [weights, bias] (up to scale and sign), and
# decides whether two dual points are consistent (belong to the same neuron) -- the test par_cluster uses.
# All geometry is done at the top of the prefix: Prefix holds OUR recovered layers below the one being attacked.
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from utils import IDIM, DIM, DEVICE, MathIsHard, norm

def intersect(left, right, nleft, nright):
    A = np.vstack((nleft, nright))
    b = np.array([np.dot(nleft, left), np.dot(nright, right)])

    # Find a particular solution
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Find the null space of A
    N = scipy.linalg.null_space(A, 1e-5)

    
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


class Prefix(nn.Module):
    """The recovered layers below the target layer, as a torch module.  files = one [row | bias] .npy per layer."""
    def __init__(self, files):
        super(Prefix, self).__init__()
        linears = []
        for f in files:
            rb = np.load(f)
            linear = nn.Linear(rb.shape[1] - 1, rb.shape[0])
            linear.weight.data = torch.tensor(rb[:, :-1])
            linear.bias.data = torch.tensor(rb[:, -1])
            linears.append(linear)
        self.fcs = nn.Sequential(*linears)
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

    @torch.no_grad
    def forward_nolastrelu(self, x):
        # pre-relu activations of every prefix layer, stacked (L, N, D); used to
        # reject duals whose zero neuron lives on an earlier layer
        x = x.view(-1, IDIM)
        pre = []
        for layer in self.fcs:
            x = layer(x)
            pre.append(x)
            x = nn.functional.relu(x)
        return torch.stack(pre)

def is_consistent_help(points, prefix, layer=0, do_return_soln=False, allow_close=False):
    samples = []
    # callers unpack (S, soln) when do_return_soln, else expect a scalar/None
    rejected = (None, None) if do_return_soln else None

    # The points need to be in different linear regions to try and compare them
    if vectorized_check_closest_pair_distance(points) and not allow_close:
        return rejected
    
    if do_return_soln:
        mid = np.stack([x[1] for x in points])
        hiddens = prefix(torch.tensor(mid).to(DEVICE)).cpu().numpy()
        hiddens = (hiddens>1e-4)
        hits = hiddens.sum(0)
        order = np.argsort(hits)

        if np.min(hits) == 0 and layer > 0:
            return rejected
        points_subset = []
        hits = np.zeros([IDIM, DIM, DIM][layer])
        
        for coord in order:
            if hits[coord] >= 4:
                continue
            for entry in np.where(hiddens[:, coord])[0][:2]:
                points_subset.append(points[entry])
                hits += hiddens[entry]
                
        points = points_subset

    for i, (left, x0, right, *normals) in enumerate(points):
        left = np.array(left)
        right = np.array(right)
        x0 = np.array(x0)

        nleft, nright = normals        # the boundary normals the walk measured on either side of the dual

        _, N = intersect(left, right, nleft, nright)
        points = generate_points_on_subspace(x0, N, DIM*2).tolist()

        points = np.concatenate(([x0], points), 0)
        
        points = prefix.forward_around(torch.tensor(points).to(DEVICE)).cpu()

        samples.append(points)

    # Each point's samples pin down only (their affine rank + 1) unknowns; behind a deep
    # prefix with few earlier neurons active that leaves free coordinates: reject those too
    pinned = sum(np.linalg.matrix_rank(s - s.mean(0), 1e-4) + 1 for s in samples)
    samples = np.concatenate(samples, 0)

    all_zero = np.sum(np.sum(np.abs(samples),0)<1e-5)

    # We need to share at least 3 coordinates in common to try and compare
    # If we only have two there are enough free variables for anything to happen.
    shared_coords = np.sum(np.sum(np.abs(samples[::DIM*2]) > 1e-5,0) >= 2)
    if shared_coords <= 3 or pinned < samples.shape[1] - all_zero + 2:
        return rejected

    mean_point = np.mean(samples, axis=0)
    
    centered_samples = samples - mean_point

    if do_return_soln:
        U, S, Vt = np.linalg.svd(centered_samples)

        ans = Vt[-1]
        ans = norm(ans)
        

        return S, Vt[-1]

    tt = torch.tensor(centered_samples).double()
    S = torch.linalg.svdvals(tt).cpu().numpy()

    return S[len(S)-all_zero-1]

def is_consistent(points, prefix, layer=0, do_return_soln=False):
    try:
        return is_consistent_help(points, prefix, layer, do_return_soln)
    except MathIsHard:
        return (None, None) if do_return_soln else None
        


def extract_weights(maybe, prefix, layer):
    S, soln = is_consistent(maybe, prefix, layer, True)

    if S is not None and S[-2] > 1e-2 and S[-1] < 1e-4:
        return soln

        
def recover_layer(LAYER, clusters, prefix_files):
    """clusters: list of lists of dual points.  Returns (weights (DIM, in), biases (DIM,)); rows not found stay zero."""
    prefix = Prefix(prefix_files).to(DEVICE)

    extracted = np.zeros((DIM, [IDIM, DIM, DIM][LAYER]))
    biases = np.zeros(DIM)   # for the sign stage: each dual point lies on its neuron's hyperplane
    nfound = 0
    # smallest clusters first; a neuron found twice (|cos| > .9) overwrites its earlier row
    for ci, maybe in enumerate(sorted(clusters, key=len)):
        maybe = np.array(maybe)
        if len(maybe) < 2:
            print("cluster %d (%d duals): dropped, too small" % (ci, len(maybe)), flush=True)
            continue
        maybe = maybe[:1200]
        soln = extract_weights(maybe, prefix, LAYER)
        if soln is None:
            print("cluster %d (%d duals): dropped, no clean single null direction" % (ci, len(maybe)), flush=True)
            continue
        same = [i for i in range(nfound) if abs(extracted[i] @ soln) > .9]
        slot = same[0] if same else nfound
        if slot >= DIM:
            print("cluster %d (%d duals): dropped, already have %d neurons" % (ci, len(maybe), DIM), flush=True)
            continue
        if same:
            print("cluster %d (%d duals): duplicate of neuron %d (|cos| %.6f), overwrites it" % (ci, len(maybe), slot, abs(extracted[slot] @ soln)), flush=True)
        else:
            print("cluster %d (%d duals): neuron %d" % (ci, len(maybe), slot), flush=True)
        extracted[slot] = soln
        # every dual of the cluster lies on the hyperplane: the bias is minus the activation there
        biases[slot] = -np.median(prefix(torch.tensor(maybe[:, 1]).to(DEVICE)).cpu().numpy() @ soln)
        nfound = max(nfound, slot+1)

    return extracted, biases
