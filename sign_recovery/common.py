# ---------------------------------------------------
# Prevent file locking errors
# ---------------------------------------------------
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ---------------------------------------------------
# Imports
# ---------------------------------------------------
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import shutil

def getFormattedTimestamp():
    from datetime import datetime
    # Format the timestamp
    formatted_timestamp = datetime.now().strftime('%Y-%m-%d')
    return formatted_timestamp

def getSavePath(modelname, layerID, neuronID, runID=None, mkdir=True, whitebox=True):
    """mkdir: If `True` the directory will be deleted if it already exists."""
    from pathlib import Path

    if runID:
        runID = '_'+runID
    else:
        runID = ''

    pathName = f"results/{'whitebox' if whitebox else 'blackbox'}/{modelname.split('.')[0]}{runID}/layerID_{layerID}/neuronID_{neuronID}/"

    if mkdir:
        if os.path.exists(pathName): shutil.rmtree(pathName, ignore_errors=True)
        Path(pathName).mkdir(parents=True, exist_ok=True)

    return pathName

def parseArguments_whitebox(argv=None):

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
    parser.add_argument('--layerID', type=int,
                        help='The ID of your target layer (as enumerated in model.layers).')
    parser.add_argument('--neuronID', type=int,
                        help="Specific target neuron IDs, e.g. '0 10 240'")
    parser.add_argument('--runID', type=str,
                        help="Custom run label (to avoid overwritting results from previous runs)")
    parser.add_argument('--nExp', type=int,
                        help="Number of points to be investigated.")
    parser.add_argument('--analyzeWiggleSensitivity', type=str,
                        help="If 'True' the sensitivity of the target layer output to a wiggle in the input will be analyzed")
    parser.add_argument('--analyzeSpeed', type=str,
                        help="If 'True' the average speed with which all future neurons in the network move will be analyzed")
    parser.add_argument('--handlePrevLayerToggles', type=str,
                        help="If 'True' we continue moving along the decision hyperplane if a neuron in the previous layer was toggled.")
    parser.add_argument('--nToggles', type=int,
                        help="Number of future-layer neurons to be toggled before concluding the experiment")
    parser.add_argument('--nDebug', type=str,
                        help="If 'True' the code will skip consistency checks and logging.")
    parser.add_argument('--filepath_load_x0', type=str, 
                        help="HARDLABEL: Filepath to a *.npy file from which to load dual or critical points")
    parser.add_argument('--nExpMin', type=int,
                        help="HARDLABEL: minimum number of dual points")
    parser.add_argument('--choose_dx', type=str,
                        help="HARDLABEL: 'along_decision_boundary', 'perfect_control_along_decision_boundary'")

    # ---- default values
    defaults = {'model': "cifar10_3x256_64_10_float64",
                'layerID': 2,
                'neuronID': '10',
                'runID': None,
                'nExp': 400,
                'analyzeWiggleSensitivity': 'False',
                'analyzeSpeed': 'False',
                'handlePrevLayerToggles': 'True',
                'nToggles': 1,
                'nDebug': 'False',
                'filepath_load_x0': '../data/dual_points_cifar10_3x256_64_10_float64', 
                'nExpMin': 25, 
                'choose_dx': 'along_decision_boundary',
                }

    # ---- parse args
    parser.set_defaults(**defaults)

    if not argv: args = parser.parse_args()
    else: args = parser.parse_args(argv)

    return args


def parseArguments_blackbox(argv=None):

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

def parseRange(s):
    out = []
    for x in s.split(','):
        if '-' in x:
            a,b = x.split('-')
            out += list(range(int(a), int(b)+1))
        else:
            out += [int(x)]
    return out

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
