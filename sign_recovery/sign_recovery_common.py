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

def parseRange(s):
    out = []
    for x in s.split(','):
        if '-' in x:
            a,b = x.split('-')
            out += list(range(int(a), int(b)+1))
        else:
            out += [int(x)]
    return out

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
