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

def classOutput(weights, biases, x0):
    x = x0.copy()
    for i in range(len(weights)):
        x = x.flatten() @ weights[i] + biases[i].flatten()
        if i < len(weights)-1:
            x[x < 0] = 0
    return np.argmax(x)

def findDecissionBoundary(weights, biases, x0, dx0, tol, inf):
    """
    Starting from x0, walk in the direction of dx0 until we cross a decision boundary, fails if a distance bigger than inf was walked.
    Returns the point right before the boundary was crossed (to a distance less than tol), and the classes on either side of the boundary.
    """
    classID = classOutput(weights, biases, x0)
    dx = dx0.copy()
    # Reduce displacement until we are not crossing any decission boundaries
    while True:
        if np.dot(dx.flatten(),dx.flatten()) < (1e3*tol)**2:
            raise ExperimentException("Decission boundary is too close.")
        if classOutput(weights, biases, x0 + dx) == classID:
            break
        dx /= 2
    # Now increase it until we cross the first decision boundary
    while True:
        if( np.dot(dx.flatten(),dx.flatten()) > inf**2):
            raise ExperimentException(f"Walked too far without finding a decision boundary (output {classID}).")
        if classOutput(weights, biases, x0 + dx) != classID:
            break
        dx *= 2
    # Binary search to find the point where the boundary was crossed
    xA = x0.copy()
    xB = x0 + dx
    timeout = 0
    while np.dot((xB - xA).flatten(), (xB - xA).flatten()) > tol**2:
        if classOutput(weights, biases, (xA + xB) / 2) == classID:
            xA = (xA + xB) / 2
        else:
            xB = (xA + xB) / 2
        timeout += 1
        if timeout > 10000:
            raise ExperimentException(f"Timeout while searching for decision boundary (output {classID}).")
    return xA, classOutput(weights, biases, xA), classOutput(weights, biases, xB)

def decissionPlaneNormalVector(weights, biases, x0):
    """
    Computes an unnormalized normal vector to the decision plane at x0, pointing towards the plane, assuming we are at least eps away from a bend of the plane.
    """
    M,b = getLocalMatrixAndBias(weights, biases, x0)
    z = x0.flatten() @ M + b.flatten()
    return M[:,np.argsort(z)[-1]] - M[:,np.argsort(z)[-2]]

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

def walkToDecisionPlaneBend(weights, biases, x0, dx0, tol, inf):
    dx = dx0.copy()
    # Half displacement until we are not crossing the bend
    while True:
        if np.dot(dx.flatten(),dx.flatten()) < (1e3*tol)**2:
            raise ExperimentException("Bend is too close.")
        if toggleStatesEqual(weights, biases, x0 + dx, x0):
            break
        dx /= 2
    # Now double it until we cross it
    while True:
        if( np.dot(dx.flatten(),dx.flatten()) > inf**2):
            raise ExperimentException(f"Walked too far without finding a bend.")
        if not toggleStatesEqual(weights, biases, x0 + dx, x0):
            break
        dx *= 2
    # Binary search to find the point where the boundary was crossed
    x = x0.copy()
    timeout = 0
    while np.dot(dx.flatten(), dx.flatten()) > tol**2:
        if toggleStatesEqual(weights, biases, x + dx/2, x):
            x = x + dx / 2
        dx /= 2
        timeout += 1
        if timeout > 10000:
            raise ExperimentException(f"Timeout while searching for decision boundary bend.")
    return x, dx

def findDualPoint(shape, weights, biases, eps, tol, inf):
    while True:
        try:
            x0 = np.random.uniform(size=shape)
            dx0 = np.random.normal(size=shape)
            x, c0, c1 = findDecissionBoundary(weights, biases, x0, dx0, tol, inf)
            break
        except ExperimentException as e:
            print(e)
            continue
    m = decissionPlaneNormalVector(weights, biases, x)
    timeout = 0
    while True:
        try:
            dx = np.random.normal(size=shape)
            dx -= np.dot(dx.flatten(), m.flatten()) * m / np.dot(m.flatten(), m.flatten())
            xdual, dx = walkToDecisionPlaneBend(weights, biases, x, dx, tol, inf)
            layerId, neuronId = toggledNeuron(weights, biases, xdual, xdual+dx)
            return xdual, layerId, neuronId, dx
        except ExperimentException as e:
            print(e)
            timeout += 1
            if timeout > 5:
                return findDualPoint(shape, weights, biases, eps, tol, inf)
            continue

def printHist(duals):
    for layer in range(len(duals)):
        print(f"Layer {layer}:", end=" ")
        for neuron in range(len(duals[layer])):
            print(f"{len(duals[layer][neuron])},", end=" ")
        print()

def saveDuals(duals, name):
    os.makedirs(f"../data/dual_points_{name}", exist_ok=True)
    for layer in range(len(duals)):
        for neuron in range(len(duals[layer])):
            np.save(f"../data/dual_points_{name}/layer{layer+1}_neuron{neuron}.npy", np.array(duals[layer][neuron]))


def main():

    tf.keras.backend.set_floatx('float64')
    name = "unitary_32_8x4_4_float64"
    model = tf.keras.models.load_model(f"../data/{name}.keras")
    Nlayers = 0
    weights, biases = [],[]
    for layer in model.layers:
        if type(layer) == tf.keras.layers.Dense:
            weights.append(layer.get_weights()[0])
            biases.append(layer.get_weights()[1])
            Nlayers += 1
    try: shape = model.input_shape[1:]
    except: [x for x in model.get_config()["layers"][0]["config"]["batch_shape"] if x]

    duals = []
    for layer in range(Nlayers):
        duals.append([])
        for neuron in range(weights[layer].shape[1]):
            duals[-1].append([])

    N = 0
    while True:
        x, layerId, neuronId, dx = findDualPoint(shape, weights, biases, eps=1e-7, tol=1e-14, inf=1e7)
        duals[layerId-1][neuronId].append([x, x+dx/2, x+dx])
        N += 1
        # M,b = getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x.flatten())
        # print(layerId, neuronId, x.flatten()@M+b.flatten())
        if N%100 == 0:
            print(f"Found {N} dual points.")
        if N % 1000 == 0:
            printHist(duals)
            saveDuals(duals, name)
            print(f"Saved dual points to ../data/dual_points_{name}/")

   

if __name__ == "__main__":
    main()