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

import blackbox
import whitebox
import common

class ExperimentException(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

def getLocalMatrixAndBiasLeaky(weights, biases, x0, alpha = 0.1):
    """
    Given the weights and biases up to a certain layer, find the equivalent matrix and bias
    around the vicinity of an input x0.

    Parameters
    ----------
    weights:
        A list of weights for the known layers (each of them a 2D array).
    biases:
        A list of biases for the known layers (each of them a 1D array).
    x0:
        A 1D array of inputs (of dimension equal to the second dimension of weights[0])
        OR a 2D array which consists of a vector of inputs (along the first dimension)
    alpha:
        The leaky ReLU parameter, which defines the slope of the negative part of the activation function.

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
        M, b = getLocalMatrixAndBiasLeaky(weights, biases, np.array([x0]))
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

def findDecissionBoundaryLeaky(f, x0, dx0, tol = 1e-14, inf=1e7):
    classID = f(x0)
    dx = dx0.copy()
    # Reduce displacement until we are not crossing any decission boundaries
    while True:
        if np.linalg.norm(dx) < 1e3*tol:
            raise ExperimentException("Decission boundary is too close.")
        if f(x0 + dx) == classID:
            break
        dx /= 2
    # Now increase it until we cross the first decision boundary
    while True:
        if( np.linalg.norm(dx) > inf):
            raise ExperimentException("Walked too far without finding a decision boundary.")
        if f(x0 + dx) != classID:
            break
        dx *= 2
    # Binary search to find the point where the boundary was crossed
    xA = x0.copy()
    xB = x0 + dx
    while np.linalg.norm(xB - xA) > tol:
        if f((xA + xB) / 2) == classID:
            xA = (xA + xB) / 2
        else:
            xB = (xA + xB) / 2
    return xA, f(xA), f(xB)

def getDualPointLeaky(f, shape, tol = 1e-14, inf=1e7):
    while True:
        x0 = np.random.uniform(size=shape)
        dx = np.random.normal(size=shape)
        try:
            return findDecissionBoundaryLeaky(f, x0, dx, tol = tol, inf = inf)
        except ExperimentException as e:
            continue

def hiddenLayerValuesLeaky(weights, biases, x):
    M,b = getLocalMatrixAndBiasLeaky(weights, biases, x)
    return x@M + b

def decissionPlaneNormalVectorLeaky(f, x0, eps = 1e-7, tol = 1e-14):
    basis = []
    c0 = f(x0)
    while len(basis) < x0.flatten().shape[0] - 1:
        x1 = eps*np.random.normal(size=x0.shape)
        x2 = eps*np.random.normal(size=x0.shape)
        if not f(x0+x1) == c0:
            continue
        c1 = f(x0-x1)
        if c1 == c0:
            continue
        try:
            xA, c0A, c1A = findDecissionBoundaryLeaky(f, x0 + x1, x2, tol=tol, inf=10*eps)
            dA2 = np.dot(xA-x0, xA-x0)
        except:
            xA = None
            dA2 = 1e9
        try:
            xB, c0B, c1B = findDecissionBoundaryLeaky(f, x0 + x1, -x2, tol=tol, inf=10*eps)
            dB2 = np.dot(xB-x0, xB-x0)
        except:
            xB = None
            dB2 = 1e9
        if dA2 < dB2 and xA is not None and (c0,c1)==(c0A,c1A):
            x = (x0-xA)/eps
        elif xB is not None and (c0,c1)==(c0B,c1B):
            x = (x0-xB)/eps
        else: 
            continue
        for b in basis:
            x -= np.dot(x, b) * b / np.dot(b, b)
        basis.append(x)
    m = np.random.normal(size=x0.shape)
    for b in basis:
        m -= np.dot(m, b) * b / np.dot(b, b)
    return m

def decissionPlaneNormalVectorLeaky_whitebox(weights, biases, xi):
    M, b = getLocalMatrixAndBiasLeaky(weights, biases, xi)
    z = xi@M+b
    return (M[:,np.argmax(z)] - M[:,np.argsort(z)[-2]])

def main():

    model = tf.keras.models.load_model("../data/unitary_256_64x256_10_float64.keras")
    Nlayers = (len(model.layers)+1)//2
    weights, biases = [],[]
    for i in range(Nlayers):
        weights.append(model.layers[2*i].get_weights()[0])
        biases.append(model.layers[2*i].get_weights()[1])
    f = lambda x: np.argmax(model.predict(x.reshape(1, -1), verbose = 0))
    shape = (weights[0].shape[0],)

    m = []
    x = []
    for layer in range(1,Nlayers+1):
        print(f"Layer {layer}/{Nlayers}")
        s = []
        c = []
        for i in range(len(x)):
            M,b = getLocalMatrixAndBiasLeaky(weights[:layer], biases[:layer], x[i])
            y = x[i]@M+b
            s.append(np.sign(y))
            c.append(np.abs(np.linalg.pinv(M) @ m[i]))

        while True:
            if len(x) > 0:
                ONOFF = []
                for neuron in range(weights[layer-1].shape[1]):
                    ON = np.array(c)[np.array(s)[:,neuron] > 0, neuron]
                    OFF = np.array(c)[np.array(s)[:,neuron] <= 0, neuron]
                    ONOFF.append(ON.mean()/OFF.mean())

                print(f"L: {layer}, N: {len(x)}, Correct: {len([x for x in ONOFF if x > 1])}/{weights[layer-1].shape[1]}")
                if np.all(np.array(ONOFF) > 1):
                    break

            print("Getting dual point")#DEB
            xi,_,_ = getDualPointLeaky(f, shape)
            print("Done")#DEB
            x.append(xi)
            print("Getting m")#DEB
            # mi = decissionPlaneNormalVectorLeaky(f, xi)
            mi = decissionPlaneNormalVectorLeaky_whitebox(weights, biases, xi)
            print("Done")#DEB
            m.append(mi)
    
            print("Getting M")#DEB
            M,b = getLocalMatrixAndBiasLeaky(weights[:layer], biases[:layer], xi)
            print("Done")#DEB
            yi = xi@M+b
            s.append(np.sign(yi))
            c.append(np.abs(np.linalg.pinv(M) @ mi))
            print("All done")#DEB

if __name__ == "__main__":
    main()

    # m = []
    # x = []
    # for layer in range(1,Nlayers+1):
    #     print(f"Layer {layer}/{Nlayers}")
    #     votes = np.zeros(weights[layer-1].shape[1])
    #     for i in range(len(x)):
    #         MM,bb = getLocalMatrixAndBiasLeaky(weights[:layer], biases[:layer], x[i])
    #         s = (np.sign(x[i]@MM + bb))
    #         c = (np.linalg.pinv(MM) @ m[i])
    #         votes[s*np.abs(c) > s*np.abs(c).mean()] += 1
    #         votes[s*np.abs(c) <= s*np.abs(c).mean()] -= 1

    #     while True:
    #         print(f"N: {len(x)}, Correct: {np.sum(votes > 0)}/{shape[0]}")
    #         if np.all(votes > 0):
    #             break

    #         xi,_,_ = getDualPointLeaky(f, shape)
    #         x.append(xi)
    #         M, b = getLocalMatrixAndBiasLeaky(weights, biases, xi)
    #         z = xi@M+b
    #         m.append(M[:,np.argmax(z)] - M[:,np.argsort(z)[-2]])

    #         MM,bb = getLocalMatrixAndBiasLeaky(weights[:layer], biases[:layer], xi)
    #         s = np.sign(xi@MM + bb)
    #         c = np.linalg.pinv(MM) @ m[-1]
    #         print(s)
    #         print(c)
    #         print(votes > 0)
    #         print(votes)
    #         votes[s*np.abs(c) > s*np.abs(c).mean()] += 1
    #         votes[s*np.a