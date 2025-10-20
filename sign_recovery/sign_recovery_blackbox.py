# ---------------------------------------------------
# Prepare environment
# ---------------------------------------------------
import pathlib
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
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
import sys
from sign_recovery_common import getLocalMatrixAndBias, ExperimentException, toggleStatesEqual, toggledNeuron, parseArguments, parseRange
from sign_recovery_whitebox import decissionPlaneNormalVector_whitebox, walkToDecisionPlaneBend_whitebox


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
            while len(basis) < x0.flatten().shape[0] - 1:
                x1 = eps/10*np.random.normal(size=x0.shape)
                x2 = eps/10*np.random.normal(size=x0.shape)
                if np.dot(x1.flatten(), n.flatten()) < 0: x1 = -x1
                if np.dot(x2.flatten(), n.flatten()) < 0: x2 = -x2
                try:
                    x3,_,_,_ = findDecissionBoundary(f, x0 + x1, x2, tol=tol, inf=9*eps/10)
                except:
                    attempts += 1
                    if attempts > 100*x0.flatten().shape[0] - 1:
                        raise ExperimentException("decissionPlaneNormalVector: Too many attempts.")
                    continue
                x = (x3 - x0)/eps
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
        assert(0)
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

def getWalkingDirection(weights, biases, neuronId, x0):
    """
    Given the weights and biases of previous layers and the current one up to signs, computes the optimal walking direction
    for neuron neuronId by multiplying its signature by the pseudoinverse of the local matrix.
    """
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

def analyzeDualPoint(f, weights, biases, xp, xm, layerId, neuronId, eps, tol, inf, pastRelusMax=0, cheatm = 1, cheatw = 1):
    """
    Given the weights and biases of the previous layers and the current one up to signs, and xp,xm on opposite sides of a relu (at least eps away from it)
    for neuron neuronId in layer layerId, walks in the optimal direction and computes the distance until a future neuron is toggled on either side.
    The walking direction will be adjusted after crossing a previous-layer relu, up to a timeout of pastReluMax times. 
    """
    try:
        dx0 = getWalkingDirection(weights[:layerId], biases[:layerId], neuronId, xp)
        M0,b0 = getLocalMatrixAndBias(weights[:layerId], biases[:layerId], xp.flatten())
        yp = (xp.flatten()@M0 + b0.flatten())[neuronId]
        if yp < 0:
            xp, xm = xm, xp
        
        #DEB
        MM,bb = getLocalMatrixAndBias(weights, biases, xp.flatten())
        print("y", yp)
        print("z", xp@MM+bb)

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
            print(side, xA@M+b, n@M)# DEB
            # if side == 'ON': continue
            while True:
                if cheatm: m = decissionPlaneNormalVector_whitebox(weights, biases, xA.flatten())
                else: m = decissionPlaneNormalVector(f, xA, n, eps, tol)
                mm = decissionPlaneNormalVector_whitebox(weights, biases, xA.flatten())# DEB
                print(side,'mb',m/np.linalg.norm(m))# DEB
                print(side,'mw',mm/np.linalg.norm(mm))# DEB
                dx = (dx - np.dot(dx,m)*m/np.dot(m,m))*side_sign # project onto the decision boundary

                xAp = xA + eps*m/np.linalg.norm(m)
                xAm = xA - eps*m/np.linalg.norm(m)

                if cheatw: xB = walkToDecisionPlaneBend_whitebox(weights, biases, xA, dx, tol, inf)
                else: xB = walkToDecisionPlaneBend(f, xAp, xAm, dx, m, tol, inf)
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
                dx = getWalkingDirection(weights[:layerId], biases[:layerId], neuronId, xB)
                xA = xB.copy()
    except ExperimentException as e:
        raise ExperimentException(f"analyzeDualPoint: {e.message}")
    return dON, dOFF
        
def getConfidence(votes_m, votes_p): 
    # Check confidence
    if (votes_m==0) and (votes_p==0): 
        return 0.0
    N = max(votes_p, votes_m)
    n = min(votes_p, votes_m)
    logp = -2*(n+N)*(0.5 - n/(n+N))**2 
    if np.isnan(logp): logp = 0.0
    return logp 

def recoverSign(model, shape, weights, biases, duals, layerId, neuronId, eps, tol, inf, Nmin, Nmax, pastRelusMax):
    """
    Recovers the sign of a single neuron given the weights and biases of previous layers and the current one up to signs.
    duals is a generator that produces dual points (xp,xm) on opposite sides of the relu for the target neuron.
    """


    model = tf.keras.models.load_model(f"../data/{args.model}.h5")
    f = lambda x: np.argmax((model.predict(x.reshape((1,)+shape), verbose = 0)))

    N = 0
    NN = 0
    votes_p = 0
    votes_m = 0

    
    with open(f"../data/results/{args.model}/{layerId}_{neuronId}.txt", "w") as outf:
        for x_dual in duals:

            if x_dual is None:
                print(f"L {layerId}, N {neuronId}: Experiments {N}/{NN}, votes+ {votes_p}, votes- {votes_m}, confidence {getConfidence(votes_m, votes_p)}", file=outf)
                print("Warning: Ran out of dual points.", file=outf)
                break
            xp, _, xm = x_dual
            if NN % 1 == 0:
                print(f"L {layerId}, N {neuronId}: Experiments {N}/{NN}, votes+ {votes_p}, votes- {votes_m}, confidence {getConfidence(votes_m, votes_p)}", file=outf)
            NN += 1
            try:
                dON, dOFF = analyzeDualPoint(f, weights, biases, xp, xm, layerId, neuronId, eps=eps, tol=tol, inf=inf, pastRelusMax=pastRelusMax)
                if dON < dOFF: 
                    votes_p += 1
                else:
                    votes_m += 1
                logp = getConfidence(votes_m, votes_p)
                N += 1
                if (logp < -3.6889 and N >= Nmin) or (N >= Nmax):
                    print(f"Stopping after {N} experiments with confidence {logp:.2f} (votes+ {votes_p}, votes- {votes_m})")
                    break
            except ExperimentException as e:
                print(e, file=outf)
                continue
        print(f"L {layerId}, N {neuronId}: Experiments {N}/{NN}, votes+ {votes_p}, votes- {votes_m}, confidence {getConfidence(votes_m, votes_p)}", file=outf)

   

if __name__ == "__main__":

    args = parseArguments(sys.argv[1:])
    tf.keras.backend.set_floatx('float64')

    model = tf.keras.models.load_model(f"../data/{args.model}.h5")
    Nlayers = 0
    weights, biases = [],[]
    for layer in model.layers:
        if type(layer) == tf.keras.layers.Dense:
            weights.append(layer.get_weights()[0])
            biases.append(layer.get_weights()[1])
            Nlayers += 1
    try: shape = model.input_shape[1:]
    except: shape = [x for x in model.get_config()["layers"][0]["config"]["batch_shape"] if x]


    try:
        neurons = parseRange(args.neuron)
        layers = parseRange(args.layer)
    except:
        print("Failed to parse neuron/layer range")
        exit(-1)

    pathlib.Path(f"../data/results/{args.model}").mkdir(parents=True, exist_ok=True)

    def func(jobId):
        if jobId // len(neurons) >= len(layers): return
        layerId = layers[jobId // len(neurons)]
        neuronId = neurons[jobId % len(neurons)]
        duals = np.load(f"../data/duals/{args.model}/{layerId}_{neuronId}.npy", allow_pickle=True)
        recoverSign(args.model, shape, weights, biases, duals, layerId, neuronId, 1e-7, 1e-14, 1e14, args.Nmin, args.Nmax, args.pastRelusMax)
        
    if args.j == 1:
        for i in range(len(neurons)*len(layers)):
            print(func(i))

    else:
        with Pool(args.j) as p:
            print(p.map(func, [i for i in range(len(neurons)*len(layers))]))
