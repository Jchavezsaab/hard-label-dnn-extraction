# ---------------------------------------------------
# Prepare environment
# ---------------------------------------------------

import os, sys

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
        description='Run the energy sign recovery.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- add arguments to parser
    parser.add_argument('--model', type=str,
                        help='The path to a keras.model (https://www.tensorflow.org/tutorials/keras/save_and_load).')
    parser.add_argument('--j', type=int,
                        help='Number of parallel threads')

    # ---- default values
    defaults = {'model': "unitary_leaky_32_32x3_10_float64",
                'j': 1,
                }

    # ---- parse args
    parser.set_defaults(**defaults)

    if not argv: args = parser.parse_args()
    else: args = parser.parse_args(argv)

    return args


# If using multiple threads, turn off multithreading at the numpy level
args = parseArguments(sys.argv[1:])
if args.j != 1:
    from multiprocessing import Pool
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import tensorflow as tf
# potentially set backend to high precision
tf.keras.backend.set_floatx('float64')

def importModelParameters(modelName):
    model = tf.keras.models.load_model(modelName)
    Nlayers = 0
    weights, biases = [],[]
    for layer in model.layers:
        if type(layer) == tf.keras.layers.Dense:
            weights.append(layer.get_weights()[0])
            biases.append(layer.get_weights()[1])
            Nlayers += 1
    try: shape = model.input_shape[1:]
    except: shape = [x for x in model.get_config()["layers"][0]["config"]["batch_shape"] if x]
    return model, weights, biases, Nlayers, shape

class ExperimentException(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

def getLocalMatrixAndBiasLeaky(weights, biases, x0, alpha = 0.1):
    """
    Given the weights and biases up to a certain layer, find the equivalent matrix and bias
    around the vicinity of an input x0. alpha is the leaky ReLU parameter, which defines the
    slope of the negative part of the activation function.
    """

    # Special case if x0 is not vectorized
    if len(x0.shape) < 2:
        M, b = getLocalMatrixAndBiasLeaky(weights, biases, np.array([x0]), alpha=alpha)
        return M[0], b[0]

    MM = []
    bb = []
    for x0i in x0:
        M = weights[0].copy()
        b = biases[0].copy()
        x = np.matmul(x0i, M) + b
        for layer_id in range(1, len(weights)):
            M_hat = weights[layer_id].copy()
            M_hat[x < 0] *= alpha
            x = np.matmul(x, M_hat) + biases[layer_id]
            b = np.matmul(b, M_hat) + biases[layer_id]
            M = np.matmul(M, M_hat)

        MM.append(M)
        bb.append(b)

    return np.array(MM), np.array(bb)

def findDecisionBoundaryLeaky(f, x0, dx0, tol, inf):
    """
    Given oracle access f to the model, starting at x0 and walking in the direction dx0, finds a point at the
    decision boundary up to precision tol. Fails if the distance walked is greater than inf.
    """
    classID = f(x0)
    dx = dx0.copy()
    # Double distance until we cross a boundary
    while True:
        if f(x0 + dx) != classID:
            break
        dx *= 2
        if( np.dot(dx.flatten(),dx.flatten()) > inf**2):
            raise ExperimentException(f"findDecisionBoundary: Walked too far without finding a decision boundary (output {classID}).")
    # Binary search to find the point where the boundary was crossed
    xA = x0.copy()
    timeout = 0
    while np.dot(dx.flatten(), dx.flatten()) > tol**2:
        if f(xA + dx/2) == classID:
            xA += dx/2
        timeout += 1
        dx /= 2
        if timeout > 10000:
            raise ExperimentException(f"findDecisionBoundary: Timeout while searching for decision boundary (output {classID}).")
    if np.all(xA == x0):
        raise ExperimentException("findDecisionBoundary: Decision boundary is too close.")
    return xA

def getDecisionPointLeaky(f, shape, tol=1e-13, inf=1e7, seed=None):
    """
    Finds a random point at the decision boundary.
    """
    if seed: np.random.seed(seed)
    while True:
        x0 = np.random.normal(size=shape)
        dx = np.random.normal(size=shape)
        try:
            return findDecisionBoundaryLeaky(f, x0, dx, tol=tol, inf = inf)
        except ExperimentException as e:
            continue

def hiddenLayerValuesLeaky(weights, biases, x, alpha=0.1):
    """
    Given the weights and biases up to a certain layer, computes the value
    of the neurons at this layer on input x.
    """
    y = x.copy().flatten()
    for i in range(len(weights)-1):
        y = y@weights[i] + biases[i]
        y [y < 0] *= alpha
    y = y@weights[-1] + biases[-1]
    return y

def decisionPlaneNormalVectorLeaky(f, x0, eps = 1e-7, tol=1e-13):
    """
    Given a point x0 at the decision boundary and at least eps away from a dual point,
    computes the normal vector of the decision plane.
    """
    try:
        m = np.random.normal(size=x0.shape)
        for _ in range(1):
            basis = []
            attempts = 0
            while len(basis) < x0.flatten().shape[0] - 1:
                x1 = eps/10*np.random.normal(size=x0.shape)
                x2 = eps/10*np.random.normal(size=x0.shape)
                try:
                    x3 = findDecisionBoundaryLeaky(f, x0 + x1, x2, tol=tol, inf=9*eps/10)
                except ExperimentException as e:
                    try:
                        x3 = findDecisionBoundaryLeaky(f, x0 + x1, -x2, tol=tol, inf=9*eps/10)
                    except ExperimentException as e:
                        attempts += 1
                        if attempts > 100*x0.flatten().shape[0] - 1:
                            raise ExperimentException("decisionPlaneNormalVector: Too many attempts.")
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
        raise ExperimentException(f"decisionPlaneNormalVector: {e.message}")
    return m

def outputFunction(weights, biases, x0, m, alpha=0.1):
    """
    Given the weights and biases of previous layers and of the current one up to signs,
    and the decision plane normal vector m at a decision-plane point x0, computes
    the output function c = m*F^-1.
    """
    z = np.linalg.pinv(weights[0])@m
    y = x0@weights[0]+biases[0]
    for i in range(1, len(weights)):
        M = weights[i].copy()
        M[y < 0] *= alpha
        z = np.linalg.pinv(M)@z
        y = y@M + biases[i]
    return z

def analyzeDecisionPoint(f, weights, biases, layerId, x, whitebox_weights, whitebox_biases):
    """
    Input: The weights and biases of previous layers and of the current one up to signs,
    oracle access to the network output f, a point x at the decision boundary, and a layer number.
    Output: The assumed signs and output functions for each neuron in the target layer, and the
    decision plane normal vector.
    """
    m = decisionPlaneNormalVectorLeaky(f, x)
    y = hiddenLayerValuesLeaky(weights[:layerId], biases[:layerId], x)
    s = np.sign(y)
    c = np.abs(outputFunction(weights[:layerId], biases[:layerId], x, m))
    #DEBUG
    MM,bb = getLocalMatrixAndBiasLeaky(whitebox_weights[:layerId], whitebox_biases[:layerId], x)
    yyi = x@MM+bb
    yyi[yyi < 0] *= 0.1
    G=getLocalMatrixAndBiasLeaky(whitebox_weights[layerId:], whitebox_biases[layerId:], yyi)[0]
    MM,bb=getLocalMatrixAndBiasLeaky(whitebox_weights, whitebox_biases, x)#DEB
    z = x@MM+bb
    C = G[:,np.argmax(z)] - G[:,np.argsort(z)[-2]]
    mm = MM[:,np.argmax(z)] - MM[:,np.argsort(z)[-2]]
    # print('m',m/np.linalg.norm(m))
    # print('M',mm/np.linalg.norm(mm))
    C[yyi < 0] *= 0.1
    # print('C',(C/np.linalg.norm(C))[:2])
    # print('c',(c/np.linalg.norm(c))[:2])
    M,b = getLocalMatrixAndBiasLeaky(weights[:layerId], biases[:layerId], x)
    return [s,c,m]

if __name__ == "__main__":
    args = parseArguments(sys.argv[1:])
    tf.keras.backend.set_floatx('float64')

    # Load the model
    model, weights, biases, Nlayers, shape = importModelParameters(f"../data/{args.model}.keras")

    # Obfuscate the signs
    whitebox_weights = [w.copy() for w in weights]
    whitebox_biases = [b.copy() for b in biases]
    real_signs = []
    for layer in range(Nlayers):
        real_signs.append(np.sign(np.random.uniform(low=-1, high=+1, size=weights[layer].shape[1])))
        weights[layer] *= real_signs[-1].reshape(1,-1)
        biases[layer] *= real_signs[-1]

    # Blackbox oracle access to the model
    def f(x):
        # return np.argmax((model.predict(x.reshape((1,)+shape), verbose = 0)))
        M,b = getLocalMatrixAndBiasLeaky(whitebox_weights, whitebox_biases, x)
        return np.argmax(x@M+b)

    # Recover the signs
    decisionBoundaryPoints = []
    m = [] # m_i will contain the normal vector of the dual plane at the i-th decision-boundary point
    for layer in range(1,Nlayers):
        print(f"Hidden Layer {layer}/{Nlayers-1}")
        s = [] # s_i will contain the assumed sign of each neuron in the target layer for the i-th decision-boundary point
        c = [] # c_i will contain the output function at the i-th decision-boundary point

        # First we update s,c to reuse decision-boundary points found when attacking previous layers
        for i in range(len(decisionBoundaryPoints)):
            y = hiddenLayerValuesLeaky(weights[:layer], biases[:layer], decisionBoundaryPoints[i])
            s.append(np.sign(y))
            c.append(np.abs(outputFunction(weights[:layer], biases[:layer], decisionBoundaryPoints[i], m[i])))

        # Now we collect more decision-boundary points as needed
        while True:
            if len(decisionBoundaryPoints) > 0:
                # Compute the recovered sign of each neuron
                recovered_signs = []
                nON = []
                nOFF = []
                for neuron in range(weights[layer-1].shape[1]):
                    ON = np.array(c)[np.array(s)[:,neuron] > 0, neuron]
                    nON.append(len(ON))
                    OFF = np.array(c)[np.array(s)[:,neuron] <= 0, neuron]
                    nOFF.append(len(OFF))
                    recovered_signs.append(1 - 2*(OFF.mean() > ON.mean()))
                recovered_signs = np.array(recovered_signs)

                print(f"Layer: {layer}, Experiments: {len(decisionBoundaryPoints)}, Correct signs: {sum(recovered_signs == real_signs[layer-1])}/{weights[layer-1].shape[1]}", end='\r')
                
                # Stop once all recovered signs are correct
                if np.all(recovered_signs == real_signs[layer-1]) and len(decisionBoundaryPoints)>100:
                    print(f"Layer: {layer}, Experiments: {len(decisionBoundaryPoints)}, Correct signs: {sum(recovered_signs == real_signs[layer-1])}/{weights[layer-1].shape[1]}")
                    weights[layer-1] *= recovered_signs.reshape(1,-1)
                    break
                # else:
                    # print("Wrong signs:")
                    # for neuron in [x for x in range(weights[layer-1].shape[1]) if np.isnan(recovered_signs[x]) or recovered_signs[x] != real_signs[layer-1][x]]:
                    #     print(f"neuron {neuron} (+/- cases: {nON[neuron]}/{nOFF[neuron]})", end = " ; ")
                    # print("\n")

            # Get new decision-boundary point
            if(args.j == 1):
                xi = getDecisionPointLeaky(f, shape)
                si, ci, mi = analyzeDecisionPoint(f, weights, biases, layer, xi, whitebox_weights, whitebox_biases)
                decisionBoundaryPoints.append(xi)
                m.append(mi)
                s.append(si)
                c.append(ci)
            else:
                def func(seed):
                    try:
                        xi = getDecisionPointLeaky(f,shape, seed=seed)
                        return [xi] + analyzeDecisionPoint(f, weights, biases, layer, xi, whitebox_weights, whitebox_biases)
                    except ExperimentException as e:
                        return func(seed+100)
                with Pool(args.j) as p:
                    outputs = p.map(func, [len(m)+i for i in range(args.j)])
                    for i in range(len(outputs)):
                        decisionBoundaryPoints.append(outputs[i][0])
                        s.append(outputs[i][1])
                        c.append(outputs[i][2])
                        m.append(outputs[i][3])