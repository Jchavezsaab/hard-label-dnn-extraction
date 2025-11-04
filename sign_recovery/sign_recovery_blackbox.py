# ---------------------------------------------------
# Prepare environment
# ---------------------------------------------------
import sys
import os
# Disable CUDA to avoid issues with multiprocessing
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Prevent file locking errors
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Don't show TensorFlow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Disable oneDNN custom operations (this avoid round-off errors from different computation orders)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def parseArguments(argv=None):

    # ---------------------------------------------------
    # Parse arguments from command line
    # ---------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Run sign recovery.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- add arguments to parser
    parser.add_argument('--model', type=str,
                        help='The path to a keras.model (https://www.tensorflow.org/tutorials/keras/save_and_load).')
    parser.add_argument('--layer', type=str,
                        help='The ID of the target layers, separated by commas without spaces, e.g. "1,2,3".')
    parser.add_argument('--neuron', type=str,
                        help="Target neuron IDs, separated by commas and using - for ranges, e.g. '0,10,240-250'")
    parser.add_argument('--runID', type=str,
                        help="Custom run label (to avoid overwritting results from previous runs)")
    parser.add_argument('--Nmin', type=int,
                        help="Minimum number of experiments to be conducted per neuron.")
    parser.add_argument('--Nmax', type=int,
                        help="Maximum number of experiments to be conducted per neuron.")
    parser.add_argument('--pastRelusMax', type=int,
                        help="Number of past-layer relus that can be crossed before aborting an experiment.")
    parser.add_argument('--j', type=int,
                        help="Number of concurrent jobs.")

    # ---- default values
    defaults = {'model': "unitary_32_8x4_4_float64",
                'layer': '1',
                'neuron': '0',
                'runID': None,
                'Nmin': 20,
                'Nmax': 1000,
                'pastRelusMax': 0,
                'j': 1
                }

    # ---- parse args
    parser.set_defaults(**defaults)

    if not argv: args = parser.parse_args()
    else: args = parser.parse_args(argv)

    return args

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

# ---------------------------------------------------
# TensorFlow / NumPy
# ---------------------------------------------------
# If using multiple threads, turn off multithreading at the numpy level
args = parseArguments(sys.argv[1:])
if args.j != 1:
    from multiprocessing import Pool
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
import tensorflow as tf

# ---------------------------------------------------
# Other imports
# ---------------------------------------------------
import pandas as pd
import time
from common import getLocalMatrixAndBias, ExperimentException, parseRange, importModelParameters, getSavePath

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

                xB = walkToDecisionPlaneBend(f, xAp, xAm, dx, m, tol, inf=1e3)

                if side=='ON': dON += np.linalg.norm((xB - xA).flatten())
                else: dOFF += np.linalg.norm((xB - xA).flatten())

                xB += dx * 10*eps/ np.linalg.norm(dx.flatten())

                if toggleStatesEqual(weights[:layerId], biases[:layerId], xA, xB):
                    break

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
                result['paseRelus'+side] = pastRelus
        result['dON'] = dON
        result['dOFF'] = dOFF
        result['succcess'] = True
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
            result = (analyzeDualPoint(f, weights, biases, xp, xm, layerId, neuronId, eps=eps, tol=tol, inf=inf, pastRelusMax=pastRelusMax))
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
            if (logp < -3.6889 and N >= Nmin) or (N >= Nmax): # logp < -2.9957: probability of wrong guess is less than 5%
                print(f"Stopping after {N} experiments with confidence {logp:.2f} (votes+ {votes_p}, votes- {votes_m})", file=logFile)
                break
            result['logp'] = logp
            results.append(result)
        except ExperimentException as e:
            print(e, file=logFile)
            continue
    print(f"L {layerId}, N {neuronId}: Experiments {N}/{NN}, votes+ {votes_p}, votes- {votes_m}, confidence {getConfidence(votes_m, votes_p)}", file=logFile)
    return results
   

if __name__ == "__main__":

    args = parseArguments(sys.argv[1:])
    tf.keras.backend.set_floatx('float64')

    # Load model
    model, weights, biases, Nlayers, shape = importModelParameters(f"../data/{args.model}.keras")

    # Target neurons/layers
    try:
        neurons = parseRange(args.neuron)
        layers = parseRange(args.layer)
    except:
        print("Failed to parse neuron/layer range")
        exit(-1)

    # Obfuscate the signs
    whitebox_weights = [w.copy() for w in weights]
    whitebox_biases = [b.copy() for b in biases]
    real_signs = []
    for layer in range(Nlayers):
        if layer+1 in layers:
            signs = np.sign(np.random.uniform(low=-1, high=+1, size=weights[layer].shape[1]))
            signs[[x for x in range(weights[layer].shape[1]) if x not in neurons]] = 1
        else:
            signs = np.ones(weights[layer].shape[1])
        real_signs.append(signs)
        weights[layer] *= real_signs[-1].reshape(1,-1)
        biases[layer] *= real_signs[-1]

    # Blackbox access to the model
    def f(x):
        # return np.argmax((model.predict(x.reshape((1,)+shape), verbose = 0)))
        M,b = getLocalMatrixAndBias(whitebox_weights, whitebox_biases, x)
        return np.argmax(x@M+b)

    # Wrapper for the function that recovers sign of a single neuron
    def func(jobId):
        if jobId // len(neurons) >= len(layers): return
        layerId = layers[jobId // len(neurons)]
        neuronId = neurons[jobId % len(neurons)]
        duals = np.load(f"../data/dual_points_{args.model}/layer{layerId}_neuron{neuronId}.npy", allow_pickle=True)
        obfuscated_weights = whitebox_weights[:layerId-1] + [weights[layerId-1]]
        obfuscated_biases = whitebox_biases[:layerId-1] + [biases[layerId-1]]
        path = getSavePath(args.model, layerId, neuronId, runID=args.runID, mkdir=True, whitebox=False)
        with open(f"{path}log.log", 'w') as logFile:
            results=recoverSign(f, obfuscated_weights, obfuscated_biases, duals, layerId, neuronId, 1e-7, 1e-14, 1e14, args.Nmin, args.Nmax, args.pastRelusMax, logFile=logFile)
        df = pd.DataFrame(results)
        df['dOFF/dON'] = df.dOFF / df.dON
        df['Vote dOFF>dON'] = df.dOFF > df.dON
        df['Real Sign'] = real_signs[layerId-1][neuronId]
        df.to_pickle(f'{path}df.pkl')

    if args.j == 1:
        for i in range(len(neurons)*len(layers)):
            func(i)

    else:
        with Pool(args.j) as p:
            p.map(func, [i for i in range(len(neurons)*len(layers))])
