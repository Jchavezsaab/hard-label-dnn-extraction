# Finds dual points: walks along a decision boundary until it bends (some neuron toggled there), records the bend,
# steps over it and carries on.  find_dual_points() returns the list of (point before the bend, bend, boundary normal
# before the bend) of one walk; consecutive entries of that list are what the clustering consumes.
import numpy as np
import torch
from utils import IDIM, DEVICE, MathIsHard, bmodel, find_decision_boundary, get_normal


def is_on_decision_boundary(point, delta, normal):
    r = normal * delta  # labels straddling the point along the boundary normal
    left = bmodel(point+r)
    right = bmodel(point-r)

    return left != right


def refine_to_decision_boundary(forward, normal):
    for step in [1e6, 2e6, 5e6, 1e5, 2e5, 5e5, 1e4, 2e4, 5e4, 1e3, 2e3, 5e3, 1e2]:
        r = normal/step  # search back to the boundary along the last normal
        if bmodel(forward+r) != bmodel(forward-r): break
    else:
        return None

    return find_decision_boundary(forward+r, forward-r)


# Find a critical point by walking along the hyperplane until
# we run into a bend, then go a bit further and record that point
def find_dual_points():
    middle_points = []

    start_point = boundary = original_boundary = find_decision_boundary()

    last_dist_to_start = 1e9

    rr = np.random.normal(size=IDIM)
    rr /= np.sum(rr**2)**.5

    while True:
        # Make it a normal vector

        dist_to_start = np.sum((boundary - start_point)**2)**.5
        if np.abs(dist_to_start - last_dist_to_start) < 2e-3:  # stalled: folding back and forth over one ridge
            break
        last_dist_to_start = dist_to_start

        try:
            normal_dir = get_normal(boundary)
        except MathIsHard:
            break

        step_dir = rr - normal_dir * np.dot(normal_dir, rr)/np.dot(normal_dir, normal_dir)
        step_dir /= np.sum(step_dir**2)**.5

        # 1. Get an upper bound on how far we should be moving, exp sampling
        boundaryt = torch.tensor(boundary).to(DEVICE).double()
        step_dirt = torch.tensor(step_dir).to(DEVICE).double()
        normal_dirt = torch.tensor(normal_dir).to(DEVICE).double()
        for step_size in 10**np.arange(-5, 5, .1):

            forward = boundaryt + step_dirt * step_size

            # same tolerance as the binary search below; grows with the drift of the estimated normal
            if not is_on_decision_boundary(forward, 1e-9 + step_size*1e-8, normal_dirt):
                break

            prev_step_size = step_size

        if step_size > 10:
            break

        if step_size <= 1e-4:
            break

        # 2. Binary search on the range
        upper_step = step_size
        lower_step = prev_step_size

        original_boundaryt = torch.tensor(original_boundary).to(DEVICE).double()

        while np.abs(upper_step - lower_step) > 1e-8:

            mid_step = (lower_step + upper_step)/2
            mid_point = original_boundaryt + step_dirt * mid_step

            if is_on_decision_boundary(mid_point, 1e-9 + mid_step*1e-8, normal_dirt):
                lower_step = mid_step
            else:
                upper_step = mid_step

        # 3. Compute the continuation direction

        middle_points.append((original_boundary + step_dir * mid_step / 2,
                              original_boundary + step_dir * mid_step,
                              normal_dir))  # the normal of this linear piece, kept for clustering

        a_bit_past = original_boundaryt + step_dirt * (mid_step + 1e-4)

        next_decision_boundary = refine_to_decision_boundary(a_bit_past, normal_dirt)

        if next_decision_boundary is None:
            break

        boundary = original_boundary = next_decision_boundary





    return middle_points


