# Sign recovery of one neuron from hard labels (the black-box sign recovery of the paper).
#
# recoverSign() is given the recovered layers up to and including the target layer (the target layer with the signs
# still unknown, i.e. a guess), and the oracle.  At dual points of the target neuron it walks along the decision
# boundary on both sides of the neuron's hyperplane until the boundary bends; if the guessed sign is right, the side
# where the neuron is ON is the one where a bend is found (dON < dOFF).  Majority over many dual points.
# Weights are in keras layout: weights[l] is (in, out), x @ W + b.
import time
import numpy as np


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


def toggledNeuron(weights, biases, x0, x1):
    x00 = x0.copy()
    x11 = x1.copy()
    for i in range(len(weights)):
        x00 = x00.flatten() @ weights[i] + biases[i].flatten()
        x11 = x11.flatten() @ weights[i] + biases[i].flatten()
        for j in range(len(x00)):
            if np.sign(x00[j]) != np.sign(x11[j]):
                return i+1, j
        x00[x00 < 0] = 0
        x11[x11 < 0] = 0
    raise ExperimentException("No toggled neuron found")
    return None, None

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

def walkToDecisionPlaneBend(f, xp, xm, dx0, m0, tol, inf):
    """
    Given two points xp and xm that are close together on opposite sides of the decision plane, walks in the direction
    of dx0 (projected onto the decision plane given by m0) until we find a bend in the decision plane.
    Returns the point just after the bend (to a distance tol)
    """
    if (f(xp) == f(xm)):
        raise ExperimentException("walkToDecisionPlaneBend: Points are not on opposite sides of the decision boundary.")
    dx = dx0 - np.dot(dx0, m0)*m0/np.dot(m0, m0)
    # Double displacement until we cross boundary
    while True:
        if f(xp+dx) == f(xm+dx):
            break
        dx *= 2
        if( np.dot(dx.flatten(),dx.flatten()) > inf**2):
            raise ExperimentException(f"walkToDecisionPlaneBend4: Walked too far without finding a bend.")
    # Binary search to find the point where the boundary was crossed
    timeout = 0
    xxp = xp.copy()
    xxm = xm.copy()
    while np.dot(dx.flatten(), dx.flatten()) > tol**2:
        if f(xxp+dx/2) != f(xxm+dx/2):
            xxp += dx/2
            xxm += dx/2
        dx /= 2
        timeout += 1
        if timeout > 10000:
            raise ExperimentException(f"walkToDecisionPlaneBend4: Timeout while searching for decision boundary bend.")
    if np.all(xxp == xp):
        raise ExperimentException("walkToDecisionPlaneBend: Bend is too close.")
    return xxp+dx

def findDualPoints(f, weights, biases, layerId, neuronId, shape, tol, inf):
    """
    Given the weights and biases of previous layers and the current one up to signs, generates dual points of neuron
    neuronId, (x - tol*n, x, x + tol*n): x is projected onto its critical hyperplane (no queries) and then walked inside
    the hyperplane to a decision boundary (hard-label queries only); n is the unit normal of the hyperplane.
    """
    def projectOntoCriticalHyperplane(x):
        # Iterated, since for deep layers the local matrix depends on x
        for _ in range(20):
            # Prefer the piece where every previous-layer neuron is active (full control of the target layer): flip the inactive
            # ones on, layer by layer (weights[:l], l < layerId, are the layers we recovered below this one; no queries)
            for l in range(1, layerId):
                M,b = getLocalMatrixAndBias(weights[:l], biases[:l], x.flatten())
                y = x.flatten()@M + b
                x = x + (np.abs(y) - y)@np.linalg.pinv(M)
            M,b = getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x.flatten())
            n = M[:,neuronId]
            y = x.flatten()@n + b[neuronId]
            x = x - y*n/np.dot(n,n)
        return x, n/np.linalg.norm(n)
    while True:
        try:
            x, n = projectOntoCriticalHyperplane(np.random.normal(size=shape))
            dx = np.random.normal(size=shape)
            dx = dx - np.dot(dx,n)*n
            xA, _, _, _ = findDecissionBoundary(f, x, 1e-3*dx/np.linalg.norm(dx), tol, inf)
            # The hyperplane bends where a previous-layer relu toggles: only keep walks that stayed on one piece of it
            if not toggleStatesEqual(weights[:layerId-1], biases[:layerId-1], x, xA):
                continue
            yield xA - tol*n, xA, xA + tol*n
        except ExperimentException:
            continue

def getWalkingDirection(weights, biases, neuronId, x0):
    """
    Given the weights and biases of previous layers and the current one up to signs, computes the optimal walking direction
    for neuron neuronId by multiplying its signature by the pseudoinverse of the local matrix.
    """
    M,_ = getLocalMatrixAndBias(weights, biases, x0.flatten())
    if np.linalg.cond(M)<1e14:
        y = np.zeros(shape=[weights[-1].shape[1]])
        y[neuronId] = 1
        dx = y@np.linalg.pinv(M)
        return dx
    if len(weights) > 1:
        Fm1,bm1 = getLocalMatrixAndBias(weights[:-1], biases[:-1], x0.flatten())
        y = x0.flatten()@Fm1 + bm1.flatten()
        Fm1[:, np.where(y <= 0)] = 0
        invF = np.linalg.pinv(Fm1)
    else:
        # special case for first hidden layer
        Fm1 = np.identity(x0.flatten().shape[0])
        invF = Fm1
    sig = weights[-1][:, neuronId].copy() # signature of target neuron
    dx = sig@invF # optimal wiggle
    return dx

def analyzeDualPoint(f, weights, biases, xp, xm, layerId, neuronId, eps, tol, inf, pastRelusMax=0):
    """
    Given the weights and biases of the previous layers and the current one up to signs, and xp,xm on opposite sides of a relu (at least eps away from it)
    for neuron neuronId in layer layerId, walks in the optimal direction and computes the distance until a future neuron is toggled on either side.
    The walking direction will be adjusted after crossing a previous-layer relu, up to a timeout of pastReluMax times. 
    """
    try:
        result = {}
        dx0 = getWalkingDirection(weights[:layerId], biases[:layerId], neuronId, xp)
        M0,b0 = getLocalMatrixAndBias(weights[:layerId], biases[:layerId], xp.flatten())
        yp = (xp.flatten()@M0 + b0.flatten())[neuronId]
        if yp < 0:
            xp, xm = xm, xp

        dON = 0
        dOFF = 0
        for (side, side_sign) in [('ON', 1), ('OFF', -1)]:
            if side == 'ON':xA = xp.copy()
            else: xA = xm.copy()
            M = M0.copy()
            b = b0.copy()
            dx = dx0.copy()
            n = M[:,neuronId].copy()*side_sign
            pastRelus = 0
            while True:
                m = decissionPlaneNormalVector(f, xA, n, eps, tol)

                dx = (dx - np.dot(dx,m)*m/np.dot(m,m))*side_sign # project onto the decision boundary

                xAp = xA + eps*m/np.linalg.norm(m)
                xAm = xA - eps*m/np.linalg.norm(m)

                try:
                    xB = walkToDecisionPlaneBend(f, xAp, xAm, dx, m, tol, inf=1e3)
                except ExperimentException as e: # no bend at all on this side: f ignores dx there, i.e. the neuron is OFF (full control)
                    if 'Walked too far' not in e.message: raise
                    if side=='ON': dON = 1e10
                    else: dOFF = 1e10
                    break

                if side=='ON': dON += np.linalg.norm((xB - xA).flatten())
                else: dOFF += np.linalg.norm((xB - xA).flatten())

                xB += dx * 10*eps/ np.linalg.norm(dx.flatten())

                if toggleStatesEqual(weights[:layerId], biases[:layerId], xA, xB):
                    break

                pastRelus += 1
                if pastRelus > pastRelusMax:
                    if side=='ON': dON = 1e10
                    else: dOFF = 1e10
                    break
                prevLayer, prevNeuron = toggledNeuron(weights[:layerId], biases[:layerId], xA, xB)
                M,b = getLocalMatrixAndBias(weights[:prevLayer], biases[:prevLayer], xB.flatten())
                n = M[:,prevNeuron].copy()
                y = xB.flatten()@M + b.flatten()
                if y[prevNeuron] < 0:
                    n = -n
                dx = getWalkingDirection(weights[:layerId], biases[:layerId], neuronId, xB)
                xA = xB.copy()
                result['paseRelus'+side] = pastRelus
        result['dON'] = dON
        result['dOFF'] = dOFF
        result['succcess'] = True
        if dON==1e10 and dOFF==1e10:
            raise ExperimentException(f"analyzeDualPoint: Too many past relus on both sides.")
    except ExperimentException as e:
        raise ExperimentException(f"analyzeDualPoint: {e.message}")
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

def recoverSign(f, weights, biases, duals, layerId, neuronId, eps, tol, inf, Nmin, Nmax, pastRelusMax, logFile=None):
    """
    Recovers the sign of a single neuron given the weights and biases of previous layers and the current one up to signs.
    duals is a generator that produces dual points (xp,xm) on opposite sides of the relu for the target neuron.
    """

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
        xp, _, xm = x_dual
        if NN % 1 == 0:
            print(f"L {layerId}, N {neuronId}: Experiments {N}/{NN}, votes+ {votes_p}, votes- {votes_m}, confidence {getConfidence(votes_m, votes_p)}", file=logFile)
        NN += 1
        try:
            t1 = time.time()
            result = analyzeDualPoint(f, weights, biases, xp, xm, layerId, neuronId, eps=eps, tol=tol, inf=inf, pastRelusMax=pastRelusMax)
            result['dual_point_id'] = dual_point_id
            dON, dOFF = result['dON'], result['dOFF']
            if dON < dOFF: 
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
            if (logp < -3.6889 and N >= Nmin) or (N >= Nmax):   # exp(-3.6889): a wrong majority is less than 2.5% likely
                print(f"Stopping after {N} experiments with confidence {logp:.2f} (votes+ {votes_p}, votes- {votes_m})", file=logFile)
                break
        except ExperimentException as e:
            print(e, file=logFile)
            continue
    print(f"L {layerId}, N {neuronId}: Experiments {N}/{NN}, votes+ {votes_p}, votes- {votes_m}, confidence {getConfidence(votes_m, votes_p)}", file=logFile)
    return results


def recover_neuron_sign(f, weights, biases, layerId, neuronId, Nmin=20, Nmax=1000, logFile=None):
    """+1 if the guessed sign of neuron neuronId of layer layerId (1-based) is right, -1 if it must be flipped."""
    shape = (weights[0].shape[0],)
    duals = findDualPoints(f, weights, biases, layerId, neuronId, shape, 1e-14, 1e3)
    results = recoverSign(f, weights, biases, duals, layerId, neuronId, 1e-5, 1e-14, 1e14, Nmin, Nmax, 0, logFile=logFile)
    votes_right = sum(r['dON'] < r['dOFF'] for r in results)
    votes_wrong = len(results) - votes_right
    assert votes_right != votes_wrong, "neuron %d of layer %d: tie" % (neuronId, layerId)
    return 1 if votes_right > votes_wrong else -1, results
