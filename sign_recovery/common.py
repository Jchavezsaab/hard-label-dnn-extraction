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

def getSavePath(modelname, layerID, neuronID, runID=None, mkdir=True):
    """mkdir: If `True` the directory will be deleted if it already exists."""
    from pathlib import Path

    if runID:
        runID = '_'+runID
    else:
        runID = ''

    pathName = f"results/model_{modelname.split('.')[0]}{runID}/layerID_{layerID}/neuronID_{neuronID}/"

    if mkdir:
        if os.path.exists(pathName): shutil.rmtree(pathName, ignore_errors=True)
        Path(pathName).mkdir(parents=True, exist_ok=True)

    return pathName

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
    parser.add_argument('--layerID', type=int,
                        help='The ID of your target layer (as enumerated in model.layers).')
    parser.add_argument('--neuronID', type=int,
                        help="Specific target neuron IDs, e.g. '0 10 240'")
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
    defaults = {'model': "./deti/modelweights/model_cifar10_256_256_256_256.keras",
                'layerID': 2,
                'neuronID': '10',
                'nExp': 400,
                'analyzeWiggleSensitivity': 'False',
                'analyzeSpeed': 'False',
                'handlePrevLayerToggles': 'True',
                'nToggles': 1,
                'nDebug': 'False',
                'filepath_load_x0': '', 
                'nExpMin': 25, 
                'choose_dx': 'along_decision_boundary',
                }

    # ---- parse args
    parser.set_defaults(**defaults)

    if not argv: args = parser.parse_args()
    else: args = parser.parse_args(argv)

    return args
