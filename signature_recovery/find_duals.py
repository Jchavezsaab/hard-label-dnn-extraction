from utils import *

def is_on_decision_boundary(point, delta):
    if USE_GRADIENT:
        return is_on_decision_boundary_cheat(point, delta)

    r = torch.randn(IDIM).cuda() * delta
    left = bmodel(point+r)
    right = bmodel(point-r)

    return left != right

def is_on_decision_boundary_cheat(point, delta):
    #real = is_on_decision_boundary_real(point, delta)
    #print(gapt(torch.tensor(point).cuda().double()))
    return torch.abs(gapt(torch.tensor(point))) < 1e-10


def refine_to_decision_boundary(forward, cancheat=True):
    if USE_GRADIENT and cancheat:
        return refine_to_decision_boundary_cheat(forward)

    for step in [1e6, 2e6, 5e6, 1e5, 2e5, 5e5, 1e4, 2e4, 5e4, 1e3, 2e3, 5e3, 1e2]:
        r = torch.randn(IDIM, device=forward.device)/step
        if bmodel(forward+r) != bmodel(forward-r): break
    else:
        return None

    return find_decision_boundary(forward+r, forward-r)

def refine_to_decision_boundary_cheat(xo, tolerance=1e-13, max_iterations=10):
    x = torch.tensor(xo).cuda().double()
    y = gapt(x).item()
    def estimate_derivative(x, h=1e-6):
        return (y - gapt(x - h)) / (h)
    for _ in range(max_iterations):
        if abs(y) < tolerance:
            return x.cpu().numpy()
        
        dy_dx = estimate_derivative(x).item()
        if dy_dx == 0:
            return refine_to_decision_boundary_real(x)
        
        x = x - y / dy_dx
        y = gapt(x).item()

    return refine_to_decision_boundary(xo, False)

# Find a critical point by walking along the hyperplane until
# we run into a bend, then go a bit further and record that point
def find_dual_points():
    print()
    print("Start find critical")
    middle_points = []
    left = None
    middle = None
    right = None

    start_point = boundary = original_boundary = find_decision_boundary()

    last_dist_to_start = 1e9

    rr = np.random.normal(size=IDIM)
    rr /= np.sum(rr**2)**.5

    while True:
        #print("Count", len(result))
        # Make it a normal vector
        #try:

        dist_to_start = np.sum((boundary - start_point)**2)**.5
        print("Distance", dist_to_start)
        if np.abs(dist_to_start - last_dist_to_start) < 1e-4:
            break
        last_dist_to_start = dist_to_start

        if USE_GRADIENT:
            try:
                normal_dir = get_normal(boundary)
            except MathIsHard:
                print("Broke")
                break
            
            step_dir = rr - normal_dir * np.dot(normal_dir, rr)/np.dot(normal_dir, normal_dir)
            step_dir /= np.sum(step_dir**2)**.5
        else:
            SZ = 4
            for _ in range(3):
                idxs = np.random.choice(IDIM, size=SZ, replace=False)
                try:
                    step_dir_part = get_gradient_dir_fast(boundary, dimensions=idxs)
                    break
                except MathIsHard:
                    continue
            else:
                break
            step_dir_part[0] *= -(SZ-1)
            step_dir = np.zeros(IDIM)
            step_dir[idxs] = step_dir_part
        
        #print('gg', gap(boundary + step_dir*1e-5))


        #print("Gap", gap(boundary))
        
        # 1. Get an upper bound on how far we should be moving, exp sampling
        # TODO: pull this out and then write a version that's just "on hyperplane" that just checks gap(x) > tol
        boundaryt = torch.tensor(boundary).cuda().double()
        step_dirt = torch.tensor(step_dir).cuda().double()
        for step_size in 10**np.arange(-5, 5, .1):

            forward = boundaryt + step_dirt * step_size

            if not is_on_decision_boundary(forward, 1e-5):
                break

            #new_forward = torch.tensor(refine_to_decision_boundary(forward.cpu().numpy())).cuda()
            #step_dirt = new_forward - torch.tensor(original_boundary).cuda()
            #step_dirt /= torch.sum(step_dirt**2)**.5
            prev_step_size = step_size
        #step_dir = step_dirt.cpu().numpy()

        #print('dd', cheat_neuron_diff(boundary, boundary + step_dir * step_size))
        #print(step_size)
        #print(gap(boundary))
        #print(gap(boundary + step_dir * step_size))

        #forward = forward.cpu().numpy()

        if step_size > 10:
            print("Step too big", step_size)
            break
        
        if step_size <= 1e-4:
            print("Step too small")
            break
        print("Step size", step_size)
        
        # 2. Binary search on the range
        upper_step = step_size
        lower_step = prev_step_size

        original_boundaryt = torch.tensor(original_boundary).cuda().double()
        
        while np.abs(upper_step - lower_step) > 1e-8:
            #after_signature = np.sign(cheat(original_boundary + step_dir * lower_step).flatten())
            #assert np.sum(original_signature != after_signature) == 0

            #print("Search on the range", lower_step, upper_step)
            mid_step = (lower_step + upper_step)/2
            mid_point = original_boundaryt + step_dirt * mid_step

            #after_signature = np.sign(cheat(mid_point).flatten())
            #print("Mid diff", np.sum(original_signature != after_signature))

            if is_on_decision_boundary(mid_point, 1e-9):
                lower_step = mid_step
            else:
                upper_step = mid_step


        # 3. Compute the continuation direction

        middle_points.append((original_boundary + step_dir * mid_step / 2,
                              original_boundary + step_dir * mid_step))
                              

        a_bit_past = original_boundaryt + step_dirt * (mid_step + 1e-4)

        #print('diff',cheat_neuron_diff(original_boundary, a_bit_past))

        #print(gap(a_bit_past))
        next_decision_boundary = refine_to_decision_boundary(a_bit_past)

        if next_decision_boundary is None:
            exit(0)
            print("Hit end of the road")
            break

        if DEBUG and False:
            print('neuron',list(np.where((np.sign(cheat(original_boundary).flatten())!=np.sign(cheat(next_decision_boundary).flatten())))[0]))
            if np.sum(np.sign(cheat(original_boundary).flatten())!=np.sign(cheat(next_decision_boundary).flatten())) != 1:
                print('skip count', np.sum(np.sign(cheat(original_boundary).flatten())!=np.sign(cheat(next_decision_boundary).flatten())))
                print("Skipped over a critical point", len(middle_points))
                print('step size', mid_step)
                print(cheat(a_bit_past))
                print(cheat(next_decision_boundary))
                return middle_points
        #result.append((original_boundary, next_decision_boundary))

        #after_signature = np.sign(cheat(next_decision_boundary).flatten())
        #print('neuron diff',np.sum(original_signature != after_signature))
        #assert np.sum(original_signature != after_signature) == 1

        boundary = original_boundary = next_decision_boundary

        #print('have',boundary)

    print("This path found", len(middle_points))

    #sigs = []
    #for x,y in result:
    #    print(np.sum(np.sign(cheat(x).flatten())!=np.sign(cheat(y).flatten())))


    return middle_points



def main():
    found_points = []
    
    remaining_crits = []
    
    all_points = []

    np.random.seed(None)
    random.seed(None)

    while len(all_points) < 10000:
        print("Status", len(all_points), "/", 10000)
        remaining_crits = find_dual_points()
        remaining_crits = list(zip(remaining_crits, remaining_crits[1:]))
    
        for (left, dual), (right, _) in remaining_crits:
            all_points.append((left, dual, right))


    if not os.path.exists("exp/%d"%SEED):
        os.mkdir("exp/%d"%SEED)
    import pickle
    pickle.dump(all_points, open("exp/%d/duals_%08d.p"%(SEED,random.randint(0, 1000000)),"wb"))
    
    print("Finished")
        
main()
