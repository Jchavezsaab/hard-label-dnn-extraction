# Sign recovery of one neuron from hard labels (the black-box sign recovery of the paper).
#
# recoverSign() is given the recovered layers up to and including the target layer (the target layer with the signs
# still unknown, i.e. a guess), and the oracle.  At each dual point of the target neuron j the probe is v = e_j pinv(M),
# M the local matrix from the input to the target layer: the input step that moves neuron j and, where M has full rank,
# no other neuron of the target layer.  The decision boundary's normal is measured on both sides of the neuron's
# hyperplane; where the neuron is off the boundary ignores it and its normal is orthogonal to v, where it is on it is
# not.  The side whose boundary makes the larger angle with v's hyperplane (|cos(normal, v)| larger) gets the vote for
# being the on side.  Dual points are taken wherever they fall: M may be singular there, and v then also moves other
# neurons of the target layer, so a single experiment can be wrong; the majority over many dual points decides.
# Weights are in keras layout: weights[l] is (in, out), x @ W + b.
import time
import numpy as np


CONFIDENT = -3.6889       # log of the chance that the majority is wrong at which the vote stops: exp(-3.6889) = 2.5%
DUAL_FAILURES_MAX = 1000  # dual-point attempts in a row that may fail before findDualPoints gives up


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
    def projectOntoCriticalHyperplane(x):
        # Iterated, since for deep layers the local matrix depends on x
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
            x, n = projectOntoCriticalHyperplane(np.random.normal(size=shape))
            dx = np.random.normal(size=shape)
            dx = dx - np.dot(dx,n)*n
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
    One experiment at the dual point x0 of neuron neuronId of layer layerId (1-based): the decision boundary's normal on
    each side of the neuron's hyperplane (decissionPlaneNormalVector stays eps/10 clear of it) and its |cos| with the
    probe v = e_j pinv(M) (see the top of this file).  'ON' is the side where our row (the guessed sign) is positive;
    the vote is for the guessed sign if that side's |cos| is the larger one.
    """
    M, _ = getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x0.flatten())
    n = M[:, neuronId]                       # gradient of our row's pre-activation: +n points to the side where it is > 0
    if np.dot(n, n) == 0:
        raise ExperimentException("analyzeDualPoint: the neuron does not depend on x here.")
    e_j = np.zeros(M.shape[1])
    e_j[neuronId] = 1
    v = e_j @ np.linalg.pinv(M)              # moves neuron j; also other target-layer neurons where M is singular
    result = {}
    try:
        for side, side_sign in (('ON', 1), ('OFF', -1)):
            m = decissionPlaneNormalVector(f, x0, side_sign * n, eps, tol)
            result['cos' + side] = abs(np.dot(m.flatten(), v)) / (np.linalg.norm(m) * np.linalg.norm(v))
    except ExperimentException as e:
        raise ExperimentException(f"analyzeDualPoint: {e.message}")
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
            if (logp < CONFIDENT and N >= Nmin) or (N >= Nmax):
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
    whether the vote reached the confidence CONFIDENT, the experiments).

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
    confident = getConfidence(votes_wrong, votes_right) < CONFIDENT and len(results) >= Nmin
    return 1 if votes_right > votes_wrong else -1, confident, results
