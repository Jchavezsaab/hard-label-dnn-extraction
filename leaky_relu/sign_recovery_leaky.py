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

class ExperimentException(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

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

def getDecisionPointLeaky(f, shape, tol=1e-13, inf=1e7):
    """
    Finds a random point at the decision boundary.
    """
    while True:
        x0 = np.random.uniform(size=shape)
        dx = np.random.normal(size=shape)
        try:
            return findDecisionBoundaryLeaky(f, x0, dx, tol=tol, inf = inf)
        except ExperimentException as e:
            print(e)
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

# #DEBUG
# def decisionPlaneNormalVectorLeaky_whitebox(weights, biases, xi, alpha=0.1):
#     M, b = getLocalMatrixAndBiasLeaky(weights, biases, xi, alpha=alpha)
#     z = xi@M+b
#     return (M[:,np.argmax(z)] - M[:,np.argsort(z)[-2]])

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

def main():
    tf.keras.backend.set_floatx('float64')

    # Load the model

    try: modelName = sys.argv[1]
    except: modelName = "unitary_leaky_32_8x4_4_float64"  # Default model
    model, weights, biases, Nlayers, shape = importModelParameters(f"../data/{modelName}.keras")

    # #DEBUG
    # white_weights = [w.copy() for w in weights]
    # white_biases = [b.copy() for b in biases]

    # Obfuscate the signs
    real_signs = []
    for layer in range(Nlayers):
        real_signs.append(np.sign(np.random.uniform(low=-1, high=+1, size=weights[layer].shape[1])))
        weights[layer] *= real_signs[-1].reshape(1,-1)
        biases[layer] *= real_signs[-1]

    # Blackbox oracle access to the model
    f = lambda x: np.argmax((model.predict(x.reshape((1,)+shape), verbose = 0)))

    # Recover the signs
    decisionBoundaryPoints = []
    m = [] # m_i will contain the normal vector of the dual plane at the i-th decision-boundary point
    for layer in range(1,Nlayers+1):
        print(f"Layer {layer}/{Nlayers}")
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

                print(f"Layer: {layer}, Experiments: {len(decisionBoundaryPoints)}, Correct signs: {sum(recovered_signs == real_signs[layer-1])}/{weights[layer-1].shape[1]}")
                
                # Stop once all recovered signs are correct
                if np.all(recovered_signs == real_signs[layer-1]):
                    weights[layer-1] *= recovered_signs.reshape(1,-1)
                    break
                else:
                    print("Wrong signs:")
                    for neuron in [x for x in range(weights[layer-1].shape[1]) if np.isnan(recovered_signs[x]) or recovered_signs[x] != real_signs[layer-1][x]]:
                        print(f"neuron {neuron}, +cases: {nON[neuron]}, -cases:{nOFF[neuron]}", end = " / ")
                    print()

            # Get new decision-boundary point
            xi = getDecisionPointLeaky(f, shape)
            decisionBoundaryPoints.append(xi)
            mi = decisionPlaneNormalVectorLeaky(f, xi)
            # mi = decisionPlaneNormalVectorLeaky_whitebox(white_weights, white_biases, xi)#DEBUG
            m.append(mi)

            # Compute predicted sign and output function
            yi = hiddenLayerValuesLeaky(weights[:layer], biases[:layer], xi)
            s.append(np.sign(yi))
            c.append(np.abs(outputFunction(weights[:layer], biases[:layer], xi, mi)))

            # #DEBUG:
            # MM,bb = getLocalMatrixAndBiasLeaky(white_weights[:layer], white_biases[:layer], xi)
            # yyi = xi@MM+bb
            # yyi[yyi < 0] *= 0.1
            # G=getLocalMatrixAndBiasLeaky(white_weights[layer:], white_biases[layer:], yyi)[0]
            # MM,bb=getLocalMatrixAndBiasLeaky(white_weights, white_biases, xi)#DEB
            # z = xi@MM+bb
            # C = G[:,np.argmax(z)] - G[:,np.argsort(z)[-2]]
            # C[yyi < 0] *= 0.1
            # print('C',(C/np.linalg.norm(C))[:2])
            # print('c',(c[-1]/np.linalg.norm(c[-1]))[:2])
            # M,b = getLocalMatrixAndBiasLeaky(weights[:layer], biases[:layer], xi)

if __name__ == "__main__":
    main()