# ---------------------------------------------------
# Prepare environment
# ---------------------------------------------------
import os
# Disable CUDA to avoid issues with multiprocessing
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Restrict numpy to occupy only 1 thread on the CPU (multithreads are better employed by launching the analyzes of multiple neurons in parallel)
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
# Prevent file locking errors
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Don't show TensorFlow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Disable oneDNN custom operations (this avoid round-off errors from different computation orders)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---------------------------------------------------
# TensorFlow
# ---------------------------------------------------
import tensorflow as tf
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

# potentially set backend to high precision
tf.keras.backend.set_floatx('float64')

# ---------------------------------------------------
# Other imports
# ---------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
import sys

class ExperimentException(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

def getLocalMatrixAndBias(weights, biases, x0):
    """
    Get the local matrix and bias of the DNN at a given point x0.
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
    xB = x0 + dx
    timeout = 0
    while np.dot((xB - xA).flatten(), (xB - xA).flatten()) > tol**2:
        if f((xA + xB) / 2) == classID:
            xA = (xA + xB) / 2
        else:
            xB = (xA + xB) / 2
        timeout += 1
        if timeout > 10000:
            raise ExperimentException(f"findDecissionBoundary: Timeout while searching for decision boundary (output {classID}).")
    if np.all(xA == x0):
        raise ExperimentException("findDecissionBoundary: Decission boundary is too close.")
    return xA, xB, f(xA), f(xB)

def findDecisionBoundaryPM(f, x0, dx0, tol, inf):
    """
    Starting from x0 and walking in both directions +dx0 and -dx0, finds the first boundary crossed.
    Returns the point just after the boundary is crossed (to a distance less than tol) and the sign of the direction in which it was found.
    """
    try:
        xp, _, _,_ = findDecissionBoundary(f, x0, dx0, tol, inf)
        dp = np.linalg.norm((xp - x0).flatten())
    except ExperimentException as e:
        mp = e.message
        if e.message == "findDecissionBoundary: Decission boundary is too close.":
            dp = 0
            xp = x0.copy()
        else:
            dp = 1e9
    try:
        xm,_,_,_ = findDecissionBoundary(f, x0, -dx0, tol, inf)
        dm = np.linalg.norm((xm - x0).flatten())
    except ExperimentException as e:
        mm = e.message
        if e.message == "findDecissionBoundary: Decission boundary is too close.":
            dm = 0
            xm = x0.copy()
        else:
            dm = 1e9
    if dp > 1e8 and dm > 1e8:
        print(mp, mm)
        raise ExperimentException("findDecisionBoundaryPM: No decision boundary found.")
    elif dp < dm:
        return xp, 1
    else:
        return xm, -1

def decissionPlaneNormalVector(f, x00, n, eps, tol, weights=None, biases=None):
    """
    Computes an unnormalized normal vector to the decision plane at x0, pointing towards the plane, assuming we are at least eps away from a bend of the plane.
    """
    m = np.random.normal(size=x00.shape)
    c0 = f(x00)
    for _ in range(1):
        basis = []
        attempts = 0
        while len(basis) < x00.flatten().shape[0] - 1:
            x1 = eps/10*np.random.normal(size=x00.shape)
            x2 = eps/10*np.random.normal(size=x00.shape)
            if np.dot(x1.flatten(), n.flatten()) < 0: x1 = -x1
            if np.dot(x2.flatten(), n.flatten()) < 0: x2 = -x2
            try: x0,_,_,_ = findDecissionBoundary(f, x00, x2, tol, inf=eps)
            except: continue
            if not f(x0+x1) == c0: continue
            try:
                xA,_,c0A, c1A = findDecissionBoundary(f, x0 + x1, x2, tol=tol, inf=9*eps/10)
                assert(c0A == c0)
                assert(c1A != c0)
                dA2 = np.dot((xA-x0).flatten(), (xA-x0).flatten())
            except:
                xA = None
                dA2 = 1e9
            try:
                xB,_,c0B, c1B = findDecissionBoundary(f, x0 + x1, -x2, tol=tol, inf=9*eps/10)
                assert(c0B == c0)
                assert(c1B != c0)
                dB2 = np.dot((xB-x0).flatten(), (xB-x0).flatten())
            except:
                xB = None
                dB2 = 1e9
            if dA2 < dB2 and xA is not None:
                x = (xA-x0)/eps
            elif xB is not None:
                x = (xB-x0)/eps
            else: 
                attempts += 1
                if attempts > 100*x0.flatten().shape[0] - 1:
                    raise ExperimentException("decissionPlaneNormalVector: Could not find normal vector.")
                continue
            dx = x0.copy()
            for _ in range(1):
                x += dx - x0
                for b in basis:
                    x -= np.dot(x.flatten(), b.flatten()) * b
                dx, _ = findDecisionBoundaryPM(f, x0 + x, x2, tol, inf=1)
            basis.append(x/np.linalg.norm(x.flatten()))
        for _ in range(10):
            for b in basis:
                m -= np.dot(m.flatten(), b.flatten()) * b
    _,sign = findDecisionBoundaryPM(f, x0, m*eps, tol=tol, inf=10*eps)
    return sign*m

def walkToDecisionPlaneBend4(f, xp, xm, dx0, m0, eps, tol, inf):
    assert(f(xp) != f(xm))
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
    return xxp

def decissionPlaneNormalVector_whitebox(weights, biases, xi):
    M, b = getLocalMatrixAndBias(weights, biases, xi)
    z = xi@M+b
    a = np.argsort(z)[-1]
    b = np.argsort(z)[-2]
    assert(a != b)
    m = M[:,a] - M[:,b]
    if z[a] - z[b] > 0:
        return -m
    else:
        return m

def walkToDecisionPlaneBend_whitebox(weights, biases, x0, dx0, tol, inf):
    dx = dx0.copy()
    # Half displacement until we are not crossing the bend
    while True:
        if toggleStatesEqual(weights, biases, x0 + dx, x0):
            break
        dx /= 2
        if np.dot(dx.flatten(),dx.flatten()) < (1e3*tol)**2:
            raise ExperimentException("walkToDecisionPlaneBend_whitebox: Bend is too close.")
    # Now double it until we cross it
    while True:
        if not toggleStatesEqual(weights, biases, x0 + dx, x0):
            break
        dx *= 2
        if( np.dot(dx.flatten(),dx.flatten()) > inf**2):
            raise ExperimentException(f"walkToDecisionPlaneBend_whitebox: Walked too far without finding a bend.")
    # Binary search to find the point where the boundary was crossed
    x = x0.copy()
    timeout = 0
    while np.dot(dx.flatten(), dx.flatten()) > tol**2:
        if toggleStatesEqual(weights, biases, x + dx/2, x):
            x = x + dx / 2
        dx /= 2
        timeout += 1
        if timeout > 10000:
            raise ExperimentException(f"walkToDecisionPlaneBend_whitebox: Timeout while searching for decision boundary bend.")
    return x + dx

def walkToRelu_whitebox(f, weights, biases, x0, dx0, eps, inf):
    dx = dx0.copy()
    while toggleStatesEqual(weights, biases, x0 + dx, x0):
        dx *= 2
        if np.dot(dx.flatten(), dx.flatten()) > inf**2:
            raise ExperimentException(f"walkToRelu_whitebox: Walked too far without finding a ReLU.")
    x = x0.copy()
    while np.dot(dx.flatten(), dx.flatten()) > eps**2:
        if toggleStatesEqual(weights, biases, x + dx/2, x):
            x = x + dx / 2
        dx /= 2
    return x, dx

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

def getWalkingDirection(weights, biases, neuronId, x0):
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

def analyzeDualPoint(f, shape, weights, biases, xp, xm, layerId, neuronId, eps, tol, inf, pastRelusMax=0, cheatm = 0, cheatw = False):
    dx0 = getWalkingDirection(weights[:layerId], biases[:layerId], neuronId, xp)
    M,b = getLocalMatrixAndBias(weights[:layerId], biases[:layerId], xp.flatten())
    yp = (xp.flatten()@M + b.flatten())[neuronId]
    if yp < 0:
        xp, xm = xm, xp

    dON = 0
    dOFF = 0
    for (side, side_sign) in [('ON', 1), ('OFF', -1)]:
        if side == 'ON':xA = xp.copy()
        else: xA = xm.copy()
        n = M[:,neuronId].copy()*side_sign
        pastRelus = 0
        while True:
            if cheatm: m = decissionPlaneNormalVector_whitebox(weights, biases, xA.flatten())
            else: m = decissionPlaneNormalVector(f, xA, n, eps, tol, weights, biases)
            mm = decissionPlaneNormalVector_whitebox(weights, biases, xA.flatten())# DEB
            print(side,'mb',m/np.linalg.norm(m))# DEB
            print(side,'mw',mm/np.linalg.norm(mm))# DEB
            dx = (dx0 - np.dot(dx0,m)*m/np.dot(m,m))*side_sign # project onto the decision boundary

            #DEB4
            # try:
            #     _,xAm,_,_ = findDecissionBoundary(f, xA, m, tol, inf=10*eps)
            # except ExperimentException as e:
            #     print(e)
            #     assert(0)
            try:
                xp = xA + eps*m/np.linalg.norm(m)
                xm = xA - eps*m/np.linalg.norm(m)
                print(f(xA), f(xp), f(xm))
            except ExperimentException as e:
                print(e)
                

            if cheatw: xB = walkToDecisionPlaneBend_whitebox(weights, biases, xA, dx, tol, inf)
            else: xB = walkToDecisionPlaneBend4(f, xp, xm, dx, m, eps, tol, inf)
            xBB = walkToDecisionPlaneBend_whitebox(weights, biases, xA, dx, tol, inf)#DEB
            print(side,'db', np.linalg.norm(xB-xA))#DEB
            print(side,'dw', np.linalg.norm(xBB-xA))#DEB
            if side=='ON': dON += np.linalg.norm((xB - xA).flatten())
            else: dOFF += np.linalg.norm((xB - xA).flatten())

            # #DEB
            # if toggleStatesEqual(weights, biases, xA, xB):
            #     xB, dxB = walkToRelu_whitebox(f, weights, biases, xB, xB-xA, eps, inf)
            #     xB = xB + dxB

            xB += dx * 10*eps/ np.linalg.norm(dx.flatten())

            if toggleStatesEqual(weights[:layerId], biases[:layerId], xA, xB):
                break

            print(toggledNeuron(weights, biases, xA, xB))#DEB
            pastRelus += 1
            if pastRelus > pastRelusMax:
                raise ExperimentException(f"analyzeDualPoint: Too many past ReLUs.")
            prevLayer, prevNeuron = toggledNeuron(weights[:layerId], biases[:layerId], xA, xB)
            M,b = getLocalMatrixAndBias(weights[:prevLayer], biases[:prevLayer], xB.flatten())
            n = M[:,prevNeuron].copy()
            y = xB.flatten()@M + b.flatten()
            if y[prevNeuron] < 0:
                n = -n
            dx0 = getWalkingDirection(weights[:layerId], biases[:layerId], neuronId, xB)
            xA = xB.copy()
    print("d",dON, dOFF)#DEB
    return dON, dOFF

def dualGenerator(name, layer, neuron):
    duals = np.load(f"../data/duals/{name}/{layer}_{neuron}.npy", allow_pickle=True)
    for x0, x1, x2 in duals:
        yield x0,x2
    while True:
        yield None
        
def get_confidence(votes_m, votes_p): 
    # Check confidence
    if (votes_m==0) and (votes_p==0): 
        return 0.0
    N = max(votes_p, votes_m)
    n = min(votes_p, votes_m)
    logp = -2*(n+N)*(0.5 - n/(n+N))**2 
    if np.isnan(logp): logp = 0.0
    return logp 

def main(layerId, neuronId, Nmin, Nmax):

    tf.keras.backend.set_floatx('float64')
    name = "unitary_8_8x8_5_float64"
    model = tf.keras.models.load_model(f"../data/{name}.keras")
    Nlayers = 0
    weights, biases = [],[]
    for layer in model.layers:
        if type(layer) == tf.keras.layers.Dense:
            weights.append(layer.get_weights()[0])
            biases.append(layer.get_weights()[1])
            Nlayers += 1
    try: shape = model.input_shape[1:]
    except: shape = [x for x in model.get_config()["layers"][0]["config"]["batch_shape"] if x]
    f = lambda x: np.argmax((model.predict(x.reshape((1,)+shape), verbose = 0)))

    N = 0
    NN = 0
    votes_p = 0
    votes_m = 0

    skip=47#DEB
    skept=0#DEB
    for x_dual in dualGenerator(name, layerId, neuronId):

        skept += 1#DEB
        if skept < skip: continue #DEB
        if x_dual is None:
            print("Ran out of dual points")
            break
        xp, xm = x_dual
        if NN % 1 == 0:
            print(f"L {layerId}, N {neuronId}: Experiments {N}/{NN}, votes+ {votes_p}, votes- {votes_m}, confidence {get_confidence(votes_m, votes_p)}")
        NN += 1
        try:
            dON, dOFF = analyzeDualPoint(f, shape, weights, biases, xp, xm, layerId, neuronId, eps=1e-7, tol=1e-14, inf=1e14)
            if dON < dOFF: 
                votes_p += 1
            else:
                votes_m += 1
            logp = get_confidence(votes_m, votes_p)
            N += 1
            if (logp < -3.6889 and N >= Nmin) or (N >= Nmax):
                print(f"Stopping after {N} experiments with confidence {logp:.2f} (votes+ {votes_p}, votes- {votes_m})")
                break
        except ExperimentException as e:
            print(e)
            continue
    print(f"L {layerId}, N {neuronId}: Experiments {N}/{NN}, votes+ {votes_p}, votes- {votes_m}, confidence {get_confidence(votes_m, votes_p)}")
    return votes_p, votes_m, get_confidence(votes_m, votes_p)

   

if __name__ == "__main__":
    Nmin = 1
    Nmax = 1

    layerId = 2
    neuronId = 5
    main(layerId, neuronId, Nmin, Nmax)

    # votes_p = []
    # votes_m = []
    # confidence = []
    # for layerId in range(1, 8):
    #     votes_p.append([])
    #     votes_m.append([])
    #     confidence.append([])
    #     for neuronId in range(8):
    #         votes_p1, votes_m1, confidence1 = main(layerId, neuronId, Nmin, Nmax)
    #         votes_p[-1].append(votes_p1)
    #         votes_m[-1].append(votes_m1)
    #         confidence[-1].append(confidence1)
    # print()
    # print()
    # for layerId in range(1, 8):
    #     print(f"Layer {layerId}:")
    #     for neuronId in range(8):
    #         print(f"{votes_p[layerId-1][neuronId]:3d} / {votes_m[layerId-1][neuronId]:3d} = {confidence[layerId-1][neuronId]:.2f}", end='\t')
    #     print()
