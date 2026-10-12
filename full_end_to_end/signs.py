# Sign recovery of one neuron from hard labels.
#
# recoverSign() is given the recovered layers up to and including the target layer (the target layer with the signs
# still unknown, i.e. a guess), and the oracle.  At a dual point of the target neuron (on its hyperplane and on the
# decision boundary) it measures the decision boundary's normal on both sides of the neuron's hyperplane.  With g the
# logit difference across the boundary and n_k the input-space normal of target-layer neuron k: on the side where the
# target neuron j is off the normal is m = grad g = sum over the other active k of (dg/dh_k) n_k, on the side where it
# is on it is m + c n_j, c being how strongly the neuron feeds the decision.  The experiment compares both normals with
# v = e_j pinv(M), M = [n_k], the smallest input step that moves neuron j and none of the others: v is orthogonal to
# every other n_k, so the off side's normal is orthogonal to v and the on side's is not (v . n_j = 1).  The side whose
# boundary makes the larger angle with v's hyperplane (|cos(normal, v)| larger) is voted to be the on side.  Exact
# where M has full rank (every earlier neuron on; findDualPoints prefers such points), approximate elsewhere, where e_j
# is only reached in projection.  Majority over many dual points (see decided).
# Weights are in keras layout: weights[l] is (in, out), x @ W + b.
import time
import numpy as np


CONFIDENT = -3.6889       # log of the chance that the vote ever stops on a wrong majority: exp(-3.6889) = 2.5% (decided)
DUAL_FAILURES_MAX = 1000  # dual-point attempts in a row that may fail before findDualPoints gives up
ALL_ON_MARGIN = 0.05      # findDualPoints starts at least this far (input space) from every earlier neuron's hyperplane
COS_MIN = 1e-4            # analyzeDualPoint: the on side's |cos| must reach this (the off side's is 0 up to error)


class ExperimentException(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

def getLocalMatrixAndBias(weights, biases, x0):
    """
    Given the weights and biases up to a certain layer, find the equivalent matrix and bias
    around the vicinity of an input x.

    Parameters
    ----------
    weights:
        A list of weights for the known layers (each of them a 2D array).
    biases:
        A list of biases for the known layers (each of them a 1D array).
    x0:
        A 1D array of inputs (of dimension equal to the second dimension of weights[0])
        OR a 2D array which consists of a vector of inputs (along the first dimension)

    Returns
    -------
    M
        A 2D array representing the local matrix
        OR a 3D array representing one matrix per input (along the first dimension)
    b
        A 1D array representing the local bias vector
        OR a 2D array representing one bias per input (along the first dimension)
    """

    # Special case if x0 is not vectorized
    if len(x0.shape) < 2:
        M, b = getLocalMatrixAndBias(weights, biases, np.array([x0]))
        return M[0], b[0]

    MM = []
    bb = []
    for x0i in x0:
        M = weights[0].copy()
        b = biases[0].copy()
        x = np.matmul(x0i, M) + b
        for layer_id in range(1, len(weights)):
            M_hat = weights[layer_id].copy()
            M_hat[x < 0] = 0
            x = np.matmul(x, M_hat) + biases[layer_id]
            b = np.matmul(b, M_hat) + biases[layer_id]
            M = np.matmul(M, M_hat)

        MM.append(M)
        bb.append(b)

    return np.array(MM), np.array(bb)


def getHiddenVector(weights, biases, l, x, relu=False):
    """
    Computes the hidden vector resulting from applying the first l hidden layers
    to the given input x.

    Parameters
    ----------
    weights : array
        List of weights corresponding to the hidden layers. The i-th element in
        the list is a 2-dimensional array with the weights of the incoming
        connections to the neurons in the i-th hidden layer.
    biases : array
        List of biases corresponding to the hidden layers. The i-th element in
        the list is a 1-dimensional array with the biases of the neurons in the
        i-th hidden layer.
    l : int
        Number of hidden layers to consider.
    x : array
        1-dimensional array representing an input to the DNN.
    relu : bool, optional
        Specifies whether to compute the hidden vector before (relu=False) or
        after (relu=True) the ReLU in layer l.

    Returns
    -------
    array
        The hidden vector corresponding to x after applying the first l hidden
        layers.
    """
    y = x
    for i in range(l):
        y = np.matmul(y, weights[i]) + biases[i]
        if (i < (l - 1)) or relu:
            y *= (y > 0)
    return y


def getLocalMatrixAndBias(weights, biases, x0):
    """
    Given the weights and biases up to a certain layer, find the equivalent matrix and bias
    around the vicinity of an input x.

    Parameters
    ----------
    weights:
        A list of weights for the known layers (each of them a 2D array).
    biases:
        A list of biases for the known layers (each of them a 1D array).
    x0:
        A 1D array of inputs (of dimension equal to the second dimension of weights[0])
        OR a 2D array which consists of a vector of inputs (along the first dimension)

    Returns
    -------
    M
        A 2D array representing the local matrix
        OR a 3D array representing one matrix per input (along the first dimension)
    b
        A 1D array representing the local bias vector
        OR a 2D array representing one bias per input (along the first dimension)
    """

    # Special case if x0 is not vectorized
    if len(x0.shape) < 2:
        M, b = getLocalMatrixAndBias(weights, biases, np.array([x0]))
        return M[0], b[0]

    MM = []
    bb = []
    for x0i in x0:
        M = weights[0].copy()
        b = biases[0].copy()
        x = np.matmul(x0i, M) + b
        for layer_id in range(1, len(weights)):
            M_hat = weights[layer_id].copy()
            M_hat[x < 0] = 0
            x = np.matmul(x, M_hat) + biases[layer_id]
            b = np.matmul(b, M_hat) + biases[layer_id]
            M = np.matmul(M, M_hat)

        MM.append(M)
        bb.append(b)

    return np.array(MM), np.array(bb)


def toggleStatesEqual(weights, biases, x0, x1):
    x00 = x0.copy()
    x11 = x1.copy()
    for i in range(len(weights)):
        x00 = x00.flatten() @ weights[i] + biases[i].flatten()
        x11 = x11.flatten() @ weights[i] + biases[i].flatten()
        if not np.all(np.sign(x00) == np.sign(x11)): return False
        x00[x00 < 0] = 0
        x11[x11 < 0] = 0
    return True

def findDecissionBoundary(f, x0, dx0, tol, inf):
    """
    Starting from x0, walk in the direction of dx0 until we cross a decision boundary, fails if a distance bigger than inf was walked.
    Returns the points right before and after the boundary was crossed (to a distance less than tol), and the classes on either side of the boundary.
    """
    classID = f(x0)
    dx = dx0.copy()
    # Double distance until we cross a boundary
    while True:
        if f(x0 + dx) != classID:
            break
        dx *= 2
        if( np.dot(dx.flatten(),dx.flatten()) > inf**2):
            raise ExperimentException(f"findDecissionBoundary: Walked too far without finding a decision boundary (output {classID}).")
    # Binary search to find the point where the boundary was crossed
    xA = x0.copy()
    timeout = 0
    while np.dot(dx.flatten(), dx.flatten()) > tol**2:
        if f(xA + dx/2) == classID:
            xA += dx/2
        timeout += 1
        dx /= 2
        if timeout > 10000:
            raise ExperimentException(f"findDecissionBoundary: Timeout while searching for decision boundary (output {classID}).")
    if np.all(xA == x0):
        raise ExperimentException("findDecissionBoundary: Decission boundary is too close.")
    return xA, xA + dx, f(xA), f(xA + dx)

def decissionPlaneNormalVector(f, x0, n, eps, tol):
    """
    Computes an unnormalized normal vector to the decision plane at x0, pointing towards the plane, assuming we are at least eps away from a bend of the plane.
    """
    try:
        m = np.random.normal(size=x0.shape)
        for _ in range(1):
            basis = []
            attempts = 0
            xref = None
            while len(basis) < x0.flatten().shape[0] - 1:
                x1 = eps/10*np.random.normal(size=x0.shape)
                x2 = eps/10*np.random.normal(size=x0.shape)
                if np.dot(x1.flatten(), n.flatten()) < 0: x1 = -x1
                if np.dot(x2.flatten(), n.flatten()) < 0: x2 = -x2
                x1 += eps/10*n/np.linalg.norm(n.flatten()) # stay clear of the relu, whose position is only known up to the signature error
                try:
                    x3,_,_,_ = findDecissionBoundary(f, x0 + x1, x2, tol=tol, inf=9*eps/10)
                except:
                    attempts += 1
                    if attempts > 100*x0.flatten().shape[0] - 1:
                        raise ExperimentException("decissionPlaneNormalVector: Too many attempts.")
                    continue
                # For the same reason x0 is only approximately on this side's decision plane: use differences of boundary points
                if xref is None:
                    xref = x3
                    continue
                x = (x3 - xref)/eps
                for _ in range(10):
                    for b in basis:
                        x -= np.dot(x.flatten(), b.flatten()) * b
                basis.append(x/np.linalg.norm(x.flatten()))
            for _ in range(10):
                for b in basis:
                    m -= np.dot(m.flatten(), b.flatten()) * b
    except ExperimentException as e:
        raise ExperimentException(f"decissionPlaneNormalVector: {e.message}")
    return m

def findDualPoints(f, weights, biases, layerId, neuronId, shape, tol, inf, maxFailures=DUAL_FAILURES_MAX):
    """
    Given the weights and biases of previous layers and the current one up to signs, generates dual points of neuron
    neuronId, (x - tol*n, x, x + tol*n): x is projected onto its critical hyperplane (no queries) and then walked inside
    the hyperplane to a decision boundary (hard-label queries only); n is the unit normal of the hyperplane.
    Yields None and stops once maxFailures attempts in a row produced no dual point.
    """
    # Prefer the piece where every previous-layer neuron is active: there the target neuron's normal is generically not
    # in the span of the other target-layer neurons' normals, which analyzeDualPoint needs.  In that piece the network up
    # to the target layer is one affine map, x -> x @ G + c, so the piece is a polyhedron and the target neuron's zero
    # set in it a hyperplane: the nearest point of both is a linear program (weights[:layerId-1] are the layers we
    # recovered below this one; no queries).
    G, c, on_G, on_c = np.eye(shape[0]), np.zeros(shape[0]), [], []
    for l in range(layerId - 1):
        G, c = G @ weights[l], c @ weights[l] + biases[l]
        on_G.append(G.T)
        on_c.append(c)
    if layerId > 1:
        # Weights the clustering set to 0 (exactly 0.0): our row is only the neuron's where those inputs are off, so the
        # piece asks for them off instead.  Their weights are 0, so the target neuron's affine map is unchanged.
        off = weights[layerId - 1][:, neuronId] == 0
        on_G[-1] = np.where(off[:, None], -on_G[-1], on_G[-1])
        on_c[-1] = np.where(off, -on_c[-1], on_c[-1])
    n_target = (G @ weights[layerId - 1])[:, neuronId]
    c_target = (c @ weights[layerId - 1] + biases[layerId - 1])[neuronId]

    def projectOntoAllOnPiece(x):
        """The point of the all-on piece (every earlier neuron at least ALL_ON_MARGIN from its hyperplane) on the target
        neuron's hyperplane that is nearest to x (L-inf), or None if there is none."""
        from scipy.optimize import linprog
        A_on, c_on = np.vstack(on_G), np.concatenate(on_c)
        d = len(x)
        I = np.eye(d)
        A_ub = np.vstack([np.hstack([I, -np.ones((d, 1))]), np.hstack([-I, -np.ones((d, 1))]),
                          np.hstack([-A_on, np.zeros((len(A_on), 1))])])
        b_ub = np.concatenate([x, -x, c_on - ALL_ON_MARGIN * np.linalg.norm(A_on, axis=1)])
        res = linprog(np.r_[np.zeros(d), 1.0], A_ub=A_ub, b_ub=b_ub, A_eq=np.r_[n_target, 0.0][None], b_eq=[-c_target],
                      bounds=[(None, None)] * d + [(0, None)], method="highs")
        if res.status != 0:
            return None
        x = res.x[:d]
        x = x - (x @ n_target + c_target) * n_target / np.dot(n_target, n_target)   # exact: the piece is affine
        return x, n_target / np.linalg.norm(n_target)

    def directionInAllOnPiece(r):
        """The direction nearest to r (L-inf) on the target neuron's hyperplane along which no earlier neuron's
        pre-activation decreases, so that a walk along it never leaves the all-on piece; None if there is none."""
        from scipy.optimize import linprog
        A_on = np.vstack(on_G)
        d = len(r)
        I = np.eye(d)
        res = linprog(np.r_[np.zeros(d), 1.0],
                      A_ub=np.vstack([np.hstack([I, -np.ones((d, 1))]), np.hstack([-I, -np.ones((d, 1))]),
                                      np.hstack([-A_on, np.zeros((len(A_on), 1))])]),
                      b_ub=np.concatenate([r, -r, np.zeros(len(A_on))]), A_eq=np.r_[n_target, 0.0][None], b_eq=[0.0],
                      bounds=[(None, None)] * d + [(0, None)], method="highs")
        if res.status != 0 or np.linalg.norm(res.x[:d]) < 1e-9:
            return None
        return res.x[:d]

    def projectOntoCriticalHyperplane(x):
        # Fallback, where there is no all-on piece: project x as it is.  Iterated, since for deep layers the local
        # matrix depends on x.
        for _ in range(20):
            M,b = getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x.flatten())
            n = M[:,neuronId]
            if np.dot(n,n) == 0:
                raise ExperimentException("projectOntoCriticalHyperplane: the neuron does not depend on x here.")
            y = x.flatten()@n + b[neuronId]
            x = x - y*n/np.dot(n,n)
        return x, n/np.linalg.norm(n)
    failures = 0
    while failures < maxFailures:
        try:
            start = np.random.normal(size=shape)
            projected = projectOntoAllOnPiece(start) if layerId > 1 else None
            x, n = projected or projectOntoCriticalHyperplane(start)
            dx = np.random.normal(size=shape)
            dx = dx - np.dot(dx,n)*n
            if projected:
                # the all-on piece is thin along most directions: walk along one that stays inside it, if there is one
                inside = directionInAllOnPiece(dx / np.linalg.norm(dx))
                if inside is not None:
                    dx = inside
            xA, _, _, _ = findDecissionBoundary(f, x, 1e-3*dx/np.linalg.norm(dx), tol, inf)
            # The hyperplane bends where a previous-layer relu toggles: only keep walks that stayed on one piece of it
            if not toggleStatesEqual(weights[:layerId-1], biases[:layerId-1], x, xA):
                failures += 1
                continue
            failures = 0
            yield xA - tol*n, xA, xA + tol*n
        except ExperimentException:
            failures += 1
            continue
    yield None

def analyzeDualPoint(f, weights, biases, x0, layerId, neuronId, eps, tol):
    """
    One experiment at the dual point x0 of neuron neuronId of layer layerId (1-based).  Measures the decision boundary's
    normal on each side of the neuron's hyperplane (decissionPlaneNormalVector stays eps/10 clear of the hyperplane) and
    its |cos| with v = e_j pinv(M) (see the top of this file).  'ON' is the side where our row (the guessed sign) is
    positive; the vote is right if that side's |cos| is the larger one.
    """
    M, _ = getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x0.flatten())
    n = M[:, neuronId]                       # gradient of our row's pre-activation: +n points to the side where it is > 0
    if np.dot(n, n) == 0:
        raise ExperimentException("analyzeDualPoint: the neuron does not depend on x here.")
    # Weights the clustering set to 0 are unknown: the true normal of such a row has an extra component along each
    # zeroed input that is on here.  Those directions join M as columns v must not move either.
    columns = [M]
    if layerId > 1:
        zeroed = np.flatnonzero((np.delete(weights[layerId - 1], neuronId, axis=1) == 0).any(axis=1))
        if len(zeroed):
            P, p = getLocalMatrixAndBias(weights[:layerId - 1], biases[:layerId - 1], x0.flatten())
            on = (x0.flatten() @ P + p)[zeroed] > 0
            columns.append(P[:, zeroed[on]])
    M_all = np.hstack(columns)
    e_j = np.zeros(M_all.shape[1])
    e_j[neuronId] = 1
    v = e_j @ np.linalg.pinv(M_all)          # smallest input step that moves neuron j and (as far as possible) no other
    if np.dot(v, v) == 0:
        raise ExperimentException("analyzeDualPoint: no input step moves the neuron here.")
    result = {}
    try:
        for side, side_sign in (('ON', 1), ('OFF', -1)):
            m = decissionPlaneNormalVector(f, x0, side_sign * n, eps, tol)
            result['cos' + side] = abs(np.dot(m.flatten(), v)) / (np.linalg.norm(m) * np.linalg.norm(v))
    except ExperimentException as e:
        raise ExperimentException(f"analyzeDualPoint: {e.message}")
    if max(result['cosON'], result['cosOFF']) < COS_MIN:
        # c ~ 0: the neuron hardly feeds this boundary, both sides look like the off side
        raise ExperimentException("analyzeDualPoint: no side's boundary depends on the neuron (cos %.1e, %.1e)."
                                  % (result['cosON'], result['cosOFF']))
    result['right'] = bool(result['cosON'] > result['cosOFF'])
    return result

def getConfidence(votes_m, votes_p): 
    # Check confidence
    if (votes_m==0) and (votes_p==0): 
        return 0.0
    N = max(votes_p, votes_m)
    n = min(votes_p, votes_m)
    logp = -2*(n+N)*(0.5 - n/(n+N))**2 
    if np.isnan(logp): logp = 0.0
    return logp 

def decided(votes_m, votes_p):
    """Whether the vote is confident.  It is checked after every vote, so the bound must hold at whatever count the vote
    stops: the Hoeffding bound of N votes is held to exp(CONFIDENT) / (N (N + 1)), which summed over all N stays below
    exp(CONFIDENT)."""
    N = votes_m + votes_p
    return N > 0 and getConfidence(votes_m, votes_p) < CONFIDENT - np.log(N * (N + 1))


def recoverSign(f, weights, biases, duals, layerId, neuronId, eps, tol, Nmin, Nmax, logFile=None, NfailMax=None):
    """
    Recovers the sign of a single neuron given the weights and biases of previous layers and the current one up to signs.
    duals is a generator that produces dual points (x - tol n, x, x + tol n) of the target neuron (None when it has run
    out); each one is one experiment (analyzeDualPoint).  Stops when the vote is confident, after Nmax usable experiments, after NfailMax failed ones (default
    Nmax), or when the dual points run out, whichever comes first.
    """
    if NfailMax is None:
        NfailMax = Nmax

    N = 0
    NN = 0
    votes_p = 0
    votes_m = 0
    t0 = time.time()

    results=[]
    for dual_point_id,x_dual in enumerate(duals, start=1):

        if x_dual is None:
            print(f"L {layerId}, N {neuronId}: Experiments {N}/{NN}, votes+ {votes_p}, votes- {votes_m}, confidence {getConfidence(votes_m, votes_p)}", file=logFile)
            print("Warning: Ran out of dual points.", file=logFile)
            break
        _, x0, _ = x_dual
        if NN % 1 == 0:
            print(f"L {layerId}, N {neuronId}: Experiments {N}/{NN}, votes+ {votes_p}, votes- {votes_m}, confidence {getConfidence(votes_m, votes_p)}", file=logFile)
        NN += 1
        try:
            t1 = time.time()
            result = analyzeDualPoint(f, weights, biases, x0, layerId, neuronId, eps=eps, tol=tol)
            result['dual_point_id'] = dual_point_id
            if result['right']:
                votes_p += 1
            else:
                votes_m += 1
            logp = getConfidence(votes_m, votes_p)
            N += 1
            result['nExp'] = N
            result['subpoint_time_seconds'] = time.time()-t1
            result['total_execution_time'] = time.time()-t0
            result['logp'] = logp
            results.append(result)
            if (decided(votes_m, votes_p) and N >= Nmin) or (N >= Nmax):
                print(f"Stopping after {N} experiments with confidence {logp:.2f} (votes+ {votes_p}, votes- {votes_m})", file=logFile)
                break
        except ExperimentException as e:
            print(e, file=logFile)
            if NN - N >= NfailMax:
                print(f"Warning: Ran out of usable experiments ({NN - N} failed).", file=logFile)
                break
            continue
    print(f"L {layerId}, N {neuronId}: Experiments {N}/{NN}, votes+ {votes_p}, votes- {votes_m}, confidence {getConfidence(votes_m, votes_p)}", file=logFile)
    return results


def recover_neuron_sign(f, weights, biases, layerId, neuronId, Nmin=20, Nmax=1000, logFile=None):
    """(+1 if the guessed sign of neuron neuronId of layer layerId (1-based) is right, -1 if it must be flipped,
    whether the vote is decided (see decided), the experiments).

    A vote that stopped early (out of dual points or of usable experiments) still decides by majority, but is reported
    as not confident; with no votes at all, or a tie, there is nothing to decide and it raises."""
    shape = (weights[0].shape[0],)
    duals = findDualPoints(f, weights, biases, layerId, neuronId, shape, 1e-14, 1e3)
    results = recoverSign(f, weights, biases, duals, layerId, neuronId, 1e-5, 1e-14, Nmin, Nmax, logFile=logFile)
    votes_right = sum(r['right'] for r in results)
    votes_wrong = len(results) - votes_right
    if votes_right == votes_wrong:
        raise ExperimentException("neuron %d of layer %d: sign undecided, %d votes each way (no more usable experiments)"
                                  % (neuronId, layerId - 1, votes_right))
    confident = decided(votes_wrong, votes_right) and len(results) >= Nmin
    return 1 if votes_right > votes_wrong else -1, confident, results
