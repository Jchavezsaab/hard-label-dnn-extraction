import collections
import time
import os
import pickle
import torch
import torch.nn as nn
import random
import numpy as np
import sys
import scipy.linalg
import functools
import matplotlib.pyplot as plt

DEBUG = True
USE_GRADIENT = True

LAYERS = 5
TINY = True

if TINY:
    IDIM = 64
    DIM = 64
    SHRINK = 64
else:
    IDIM = 32*32*3
    DIM = 256
    SHRINK = 64

SEED = 1 if len(sys.argv) < 3 else int(sys.argv[1])

x_test = np.load("data/x_test.npy")
if TINY:
    x_test = x_test.mean(1)[:, ::4, ::4]
x_test = x_test.reshape((-1, IDIM))
x_test = (x_test*2-1)
x_test = np.array(x_test, dtype=np.float64)

random.seed(SEED)
np.random.seed(SEED)


class MathIsHard(Exception):
    pass

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.fc1 = nn.Linear(IDIM, DIM)
        self.fc2 = nn.Linear(DIM, DIM)
        self.fc3 = nn.Linear(DIM, DIM)
        self.fc4 = nn.Linear(DIM, SHRINK)
        self.fc5 = nn.Linear(SHRINK, 10)
        self.relu = nn.ReLU()
        self.double()

    @torch.no_grad
    def forward(self, x):
        relu = self.relu
        x = x.view(-1, IDIM)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = self.fc5(x)
        #print(x.device)
        #if str(x.device) == 'cpu':
        #    import traceback
        #    traceback.print_stack()
        return x

    def forward_grad(self, x):
        relu = self.relu
        x = x.view(-1, IDIM)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = self.fc5(x)
        return x

    @torch.no_grad
    def cheat(self, x):
        o = []
        def relu(x):
            if x.size(-1) < DIM:
                padding = DIM - x.size(-1)
                xx = torch.nn.functional.pad(x, (0, padding))
                xx[:, -padding:] = 1
                o.append(xx)
            else:
                o.append(x)
            return self.relu(x)

        x = x.view(-1, IDIM)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        return torch.stack(o)
    
cheat_net_cuda = CIFAR10Net()
if not TINY:
    cheat_net_cuda.load_state_dict(torch.load("models/final.pth"))
else:
    cheat_net_cuda.load_state_dict(torch.load("models/tiny.pth"))
cheat_net_cuda.double().cuda()

cheat_net_cpu = CIFAR10Net()
if not TINY:
    cheat_net_cpu.load_state_dict(torch.load("models/final.pth"))
else:
    cheat_net_cpu.load_state_dict(torch.load("models/tiny.pth"))
cheat_net_cpu.double()

cheat_solution = [x.cpu().detach().numpy() for x in cheat_net_cpu.parameters()][::2]


def model(x):
    assert len(x.shape) == 1
    return (cheat_net_cpu(torch.tensor(x).double()).numpy().argmax(1)).astype(np.int32)[0]
    
def bmodel(x):
    return (cheat_net_cuda(x).argmax(1)).to(torch.int32)

def cheat(x):
    if not DEBUG: raise
    return cheat_net_cpu.cheat(torch.tensor(x).double()).numpy()

def cheat_cuda(x):
    if not DEBUG: raise
    return cheat_net_cuda.cheat(x)

def gap(x):
    if not DEBUG: raise
    out = cheat_net_cpu(torch.tensor(x).double()).numpy()
    #print(out)
    max_idx = np.argmax(out, 1)
    top = out[np.arange(len(out)), max_idx]
    out[np.arange(len(out)), max_idx] = -100
    return top - np.max(out, 1)

def gapt(x, grad=False):
    if not DEBUG: raise
    if grad:
        out = cheat_net_cuda.forward_grad(x)
    else:
        out = cheat_net_cuda(x)

    max_idx = torch.argmax(out, 1)
    top = out[torch.arange(len(out)), max_idx]
    out[torch.arange(len(out)), max_idx] = -100
    return top - torch.max(out, 1).values

def cheat_num_flips(a,b):
    if not DEBUG: raise
    return np.sum((cheat(a)>0) != (cheat(b)>0))

def cheat_neuron_diff(a,b):
    if not DEBUG: raise
    return np.where((cheat(a)>0).flatten() != (cheat(b)>0).flatten())[0]

def cheat_neuron_diff_cuda(a,b):
    if not DEBUG: raise
    ab = torch.tensor(np.stack([a,b])).double().cuda()
    out = cheat_cuda(ab)>0
    a = out[:,0,:]
    b = out[:,1,:]
    return torch.where(a.flatten() != b.flatten())[0].cpu().numpy()

def cheat_neuron_diff_cuda_2(a,b):
    if not DEBUG: raise
    ab = torch.tensor(np.stack([a,b])).double().cuda()
    out = cheat_cuda(ab)>0
    a = out[:,0,:]
    b = out[:,1,:]
    return torch.where(a.flatten() != b.flatten())[0].cpu().numpy(), np.array(a.flatten().cpu().numpy(), dtype=np.uint8)


# Find the decision boundary, given two inputs:
#    zero must have model(zero) = 0
#    one  must have model(one)  = 1
# returns the midpoint that's almost 0/1 at the same time
def find_decision_boundary(zero=None, one=None, tensor=False):
    if zero is None and one is None:
        points = {}
        while len(points) < 2:
            if not TINY:
                maybe = random.sample(range(len(x_test)), 10)
                maybe = x_test[maybe]
            else:
                maybe = np.random.normal(size=(10, IDIM))
            maybe = torch.tensor(maybe).cuda().double()
            outs = bmodel(maybe)
            for out, point in zip(outs, maybe):
                points[out.item()] = point
        zero, one = list(points.values())[:2]
    #assert model(zero) != model(one)

    model_zero = bmodel(zero)
    last = 1e9
    while torch.sum(torch.abs(zero - one)) > 1e-16 and torch.sum(torch.abs(zero - one)) < last:
        last = torch.sum(torch.abs(zero - one))
        mid = (zero+one)/2
        if bmodel(mid) == model_zero:
            zero = mid
        else:
            one = mid


    if tensor:
        return zero
    return zero.cpu().numpy()

def find_decision_boundary_batched(zero, one):
    zero = torch.tensor(zero).double().cuda()
    one = torch.tensor(one).double().cuda()
    last = torch.tensor(1e9).cuda()

    orig_label = bmodel(zero)[0]
    
    while True:
        s = torch.sum(torch.abs(zero - one), dim=1)
        if not torch.any((s > 1e-14) | (s < last)).item():
            break
        
        last = s
        mid = (zero + one) / 2
        
        idx = bmodel(mid)
        
        zero_mask = (idx == orig_label)
        one_mask = (idx != orig_label)
        
        zero[zero_mask] = mid[zero_mask]
        one[one_mask] = mid[one_mask]

    return zero

# Compute the gradient direction of the decision boundary
# More correctly, this function returns a parallel direction
# to the hyperplane, so we can walk along it.
def get_gradient_dir(x, cache={}, debug=False, step_size=1e-7, dimensions=None):
    if len(cache) > 1e6:
        cache.clear()
    if tuple(x) in cache:
        return cache[tuple(x)]

    original = model(x)

    if dimensions is None:
        dimensions = range(IDIM)
    
    ratios = []
    for i in dimensions:
        if debug:
            print('iter', i)
        xp = np.array(x)


        if debug:
            print('start', gap(xp))
            sig = np.sign(cheat(x).flatten())

        xp[0] += step_size
        if model(xp) != original:
            xp[0] -= step_size*2

        if debug:
            print('   ', gap(xp))
            print('sigchange', np.sum(sig!= np.sign(cheat(xp).flatten())))

        #assert model(xp) == 0

        for step in 10**np.arange(-7, 0, .33):
            xp2 = np.array(xp)
            xp2[i] += step
            if model(xp2) == original:
                xp2[i] -= 2*step
            if model(xp2) != original:
                break
        else:
            assert False

        if debug:
            print('   ', gap(xp2))

        boundary = find_decision_boundary(xp, xp2)

        ratio = (xp[i]-boundary[i])/(step_size)
        ratios.append(ratio)

    # So far we have the gradient direction, now let's make is
    # so that we can go parallel
    ratios = np.array(ratios)

    cache[tuple(x)] = ratios

    return ratios

def vectorized_right_boundary_search(left, orig_label, dimensions):
    device = 'cuda'
    batch_size = IDIM
    
    # Initialize the batch with copies of the left boundary
    batch = left.repeat(batch_size, 1)
    
    # Create a mask to keep track of dimensions that haven't found the boundary yet
    active_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    active_mask[dimensions] = 1
    #print(active_mask)
    
    for step in 10**np.arange(-7, 0, .33):
        #print(step)
        # Try positive step
        pos_batch = batch.clone()
        pos_batch[active_mask] += torch.eye(IDIM, device=device)[active_mask] * step

        pos_results = bmodel(pos_batch)
        #print('aa',pos_results)
        
        # Update batch and mask for positive steps that crossed the boundary
        pos_crossed = (pos_results != orig_label) & active_mask
        #print(pos_crossed)
        #print('act',active_mask)
        #print("Set", torch.where(pos_crossed), torch.sum(pos_crossed.float()))
        batch[pos_crossed] = pos_batch[pos_crossed]
        active_mask[pos_crossed] = False
        
        if not active_mask.any():
            break
        
        # Try negative step for remaining active dimensions
        neg_batch = batch.clone()
        neg_batch[active_mask] -= torch.eye(IDIM, device=device)[active_mask] * step
        
        neg_results = bmodel(neg_batch)
        #print('bb',neg_results)
        
        # Update batch and mask for negative steps that crossed the boundary
        neg_crossed = (neg_results != orig_label) & active_mask
        #print(neg_crossed)
        #print('act',active_mask)
        #print("Set", torch.where(neg_crossed), torch.sum(neg_crossed.float()))

        batch[neg_crossed] = neg_batch[neg_crossed]
        active_mask[neg_crossed] = False
        
        if not active_mask.any():
            break
    
    if active_mask.any():
        raise MathIsHard("Boundary not found for all dimensions")
    
    # Convert results back to numpy arrays
    #print(batch.shape)
    #print('zz',left - batch[0])
    
    return left.repeat(IDIM, 1)[dimensions, :], batch[dimensions, :]

def get_gradient_dir_fast(x, cache={}, debug=False, step_size=1e-7, dimensions=None):
    if len(cache) > 1e6:
        cache.clear()
    if tuple(x) in cache:
        return cache[tuple(x)]

    if dimensions is None:
        dimensions = range(IDIM)
    
    #np.save("/tmp/x.npy", x)
    
    # 1. init
    # find the left and right sides
    leftright = []

    left = np.array(x)
    left = torch.tensor(left, dtype=torch.float64, device='cuda')
    original = bmodel(left)

    left[0] += step_size
    if bmodel(left) != original:
        left[0] -= step_size*2
        
    #assert model(left) == 0
    
    xp, xp2 = vectorized_right_boundary_search(left, original, dimensions)

    ratios = []
    #xp, xp2 = zip(*leftright)

    #print(gap(xp2))
    #print(model(xp))
    #print(model(xp2))
    boundary = find_decision_boundary_batched(xp, xp2)

    ratios = ((xp-boundary)/(step_size))[torch.arange(len(dimensions)), dimensions].cpu().numpy()

    # So far we have the gradient direction, now let's make is
    # so that we can go parallel
    ratios = np.array(ratios)

    cache[tuple(x)] = ratios

    return ratios

def get_normal(x, step_size=1e-6):
    if USE_GRADIENT:
        x = torch.tensor(x, requires_grad=True)
        out = gapt(x.cuda(), grad=True)
        out[0].backward()
        real = np.random.normal(0, 1) * x.grad.cpu().numpy()
        real = norm(real)
        return real
    else:
        try:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size)
        except MathIsHard:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size/10)
        fnormal = norm(fnormal)
    
        return fnormal

def get_normal_t(x, step_size=1e-6):
    if USE_GRADIENT:
        x = torch.tensor(x, requires_grad=True)
        out = gapt(x.cuda(), grad=True)
        out[0].backward()
        real = torch.tensor(np.random.normal(0, 1)) * x.grad
        real = normt(real)
        return real
    else:
        try:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size)
        except MathIsHard:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size/10)
        fnormal = norm(fnormal)
    
        return fnormal
    
def norm(x):
    return x / np.sum(x**2)**.5

def normt(x):
    return x / torch.sum(x**2)**.5
