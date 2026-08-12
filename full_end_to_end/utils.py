# Shared pieces of the signature recovery (find_duals / par_cluster / recover_weights): the sizes of the target,
# the hard-label oracle wrapped for torch, and the two label-only geometric primitives (bisecting to a decision
# boundary, and estimating the boundary's normal vector by finite differences of labels).
import numpy as np
import torch
from oracle import label   # the black box: hard labels only

IDIM = 32                  # input dimension of the target
DIM = 32                   # width of its hidden layers
DEVICE = 'cpu'



class MathIsHard(Exception):
    pass


def bmodel(x):
    """labels of a batch of torch points"""
    return torch.tensor(label(x.cpu().numpy())).reshape(-1).to(torch.int32)


def find_decision_boundary(zero=None, one=None, tensor=False):
    if zero is None and one is None:
        points = {}
        while len(points) < 2:
            maybe = np.random.normal(size=(10, IDIM))
            maybe = torch.tensor(maybe).to(DEVICE).double()
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
    zero = torch.as_tensor(zero).double().to(DEVICE)
    one = torch.as_tensor(one).double().to(DEVICE)
    last = torch.tensor(1e9).to(DEVICE)

    orig_label = bmodel(zero)[0]
    
    while True:
        s = torch.sum(torch.abs(zero - one), dim=1)
        if not torch.any((s > 1e-14) & (s < last)).item():
            break
        
        last = s
        mid = (zero + one) / 2
        
        idx = bmodel(mid)
        
        zero_mask = (idx == orig_label)
        one_mask = (idx != orig_label)
        
        zero[zero_mask] = mid[zero_mask]
        one[one_mask] = mid[one_mask]

    return zero


def get_normal(x, step_size=1e-5, cache={}):
    # Hard-label central finite differences.  Probe x +- h*e_i on every
    # axis; the probe labels give a rough normal v (the sign pattern), and
    # bisecting every probe back onto the boundary along v gives offsets s
    # with n.(+-h*e_i + s*v) the same for all probes (zero if x is exactly
    # on the boundary), i.e. n_i proportional to (s_i^- - s_i^+) and
    # s_i^+ + s_i^- independent of i.  If it is not, another ReLU crossed
    # the probe box: shrink h (then give up).
    if tuple(x) in cache:
        return cache[tuple(x)]
    xt = torch.tensor(np.array(x)).double()
    orig = bmodel(xt)
    eye = torch.eye(IDIM, dtype=torch.float64)
    for h in [step_size, step_size/10, step_size/100]:
        probes = torch.cat([xt + h*eye, xt - h*eye])
        flipped = bmodel(probes) != orig
        v = normt(flipped[:IDIM].double() - flipped[IDIM:].double())
        far = probes - IDIM**.5 * h * v[None, :] * (2*flipped.double() - 1)[:, None]
        if torch.any((bmodel(far) == orig) != flipped):
            continue
        zero = torch.where(flipped[:, None], far, probes)  # label == orig
        one = torch.where(flipped[:, None], probes, far)
        s = (find_decision_boundary_batched(zero, one) - probes) @ v
        off = s[:IDIM] + s[IDIM:]
        if torch.all(torch.abs(off - off.mean()) < 1e-8 * h):
            fnormal = norm((s[IDIM:] - s[:IDIM]).numpy())
            cache[tuple(x)] = fnormal
            return fnormal
    raise MathIsHard


def norm(x):
    return x / np.sum(x**2)**.5


def normt(x):
    return x / torch.sum(x**2)**.5
