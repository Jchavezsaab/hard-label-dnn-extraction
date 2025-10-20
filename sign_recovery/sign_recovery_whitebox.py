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
from sign_recovery_common import getLocalMatrixAndBias, ExperimentException, toggleStatesEqual

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