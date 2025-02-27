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

N_CID = 2 # how many of the strongest output layer classes to consider 

# ---------------------------------------------------
# Functions related to toggles and boundaries
# ---------------------------------------------------
def neuron_toggle_state(weights, biases, x0, with_output_classes=False):
    """
    Returns the toggled state of neurons at each layer.

    'with_output_classes': If set to 'True', we consider the last weights to be connected to a softmax layer. 
    A 'toggle' in the softmax layer occurs when the output class with the highest probability 
    changes from one class (or output neuron) to another.
    """
    if with_output_classes: 
        weightsReLU = weights[:-1] # the last weights are connected to a softmax layer, not ReLU
    else: 
        weightsReLU = weights

    x = x0.copy()
    states = []
    for i in range(len(weightsReLU)):
        x = x@weightsReLU[i] + biases[i]
        states.append(x > 0)
        x[x<0] = 0.0

    if with_output_classes: 
        x = x@weights[-1] + biases[-1] 
        if N_CID == 1: 
            states.append(x==np.max(x))
        elif N_CID > 1: 
            ids_sorted = np.argsort(x)[::-1]
            _states = np.zeros_like(x)
            _states[ids_sorted[:N_CID]] = 1
            states.append(_states)
    return states

def get_classID(weights, biases, x0): 
    # get the output states
    states = neuron_toggle_state(weights, biases, x0, with_output_classes=True)[-1] 
    return np.argwhere(states == np.max(states)).flatten() #np.argmax(states[-1])

def toggle_states_equal(weights, biases, x0, x1, with_output_classes=False):
    """
    Returns True if the toggled states of the neurons are equal at each layer.
    """
    states0 = neuron_toggle_state(weights, biases, x0, with_output_classes=with_output_classes)
    states1 = neuron_toggle_state(weights, biases, x1, with_output_classes=with_output_classes)
    for i in range(len(states0)):
        if not (states0[i] == states1[i]).all():
            return False
    return True

def find_point_of_boundary(weights, biases, x0, dx, infinity=1e10, eps=1e-6, with_output_classes=False):
    """
    Find the point of the boundary when starting from x0 in direction of dx using binary search.
    """
    x00 = x0.copy()
    # Double movement until we are past a boundary
    x1 = x0 + dx
    while toggle_states_equal(weights, biases, x0, x1, with_output_classes=with_output_classes):
        # x1 = 2*x1 - x0
        dx = 2.0*dx
        x1 = x0 + dx
        if np.linalg.norm(x1) > infinity: return np.full(x0.shape, infinity) 

    # Binary search to recover the exact toggling point
    x_mid = (x0 + x1)/2
    while np.linalg.norm(x1 - x0) > eps:
        if toggle_states_equal(weights, biases, x_mid, x0, with_output_classes=with_output_classes):
            x0 = x_mid.copy()
        else:
            x1 = x_mid.copy()
        x_mid = (x0 + x1)/2

    # Ensure that the returned boundary point is just over the toggling boundary
    if toggle_states_equal(weights, biases, x00, x1, with_output_classes=with_output_classes):
        raise Exception('Boundary search failed! The toggling states are equal at the boundary point and the original point.')

    return x1

def which_neuron_toggled(weights, biases, x0, x1, with_output_classes=False):
    """
    Change w.r.t. v9: The return value now contains the layerID of the toggled neuron
    """

    states0 = neuron_toggle_state(weights, biases, x0, with_output_classes=with_output_classes)
    states1 = neuron_toggle_state(weights, biases, x1, with_output_classes=with_output_classes)

    results = []

    for i in range(len(states0)):
        s0 = np.array(states0[i])
        s1 = np.array(states1[i])
        nIDtoggled = np.argwhere(s0 != s1)
        record_s1 = s1[nIDtoggled]
        if len(nIDtoggled)>0:
            # note that the 0th entry in the states-list
            # contains the output of the first hidden layer
            # with layerID = 1.
            # Therefore increase the i-count by 1:
            results.append([i+1, nIDtoggled.flatten(), record_s1])

    return results

# ---------------------------------------------------
# Functions related to neural network outputs
# ---------------------------------------------------
def get_target_neuron_output_at_x(x, weights, biases, layerId, neuronId):
    F,b = blackbox.getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x) # transformations from input to layerId
    y0 = (x@F + b)[neuronId] # error value of the target neuron (should be zero but wont be exactly)
    return y0

def get_target_neuron_output(x, dx, weights, biases, layerId, neuronId):
    F,b = blackbox.getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x) # transformations from input to layerId
    # y0 = (x@F + b)[neuronId] # error value of the target neuron (should be zero but wont be exactly)
    dy = (dx@F)[neuronId] # size of the wiggle produced by dx
    return dy

def get_neuron_values(x, weights, biases):
    """Given a neural network defined by 'weights' and 'biases',
        and an input x,
        get the outputs after ReLUs of each neuron in the network.

        Returns: A list of arrays, in which each array corresponds to the output
            values of the neurons in one layer of the neural network.
            v9: The return value doesn't correspond to the layerID,
                ... i.e. values[0] contains the output of the neurons in hidden layer 1 (layerID=1)
            v10: - The return value corresponds to the layerID
                ... i.e., values[1] contains the output of the neurons in layerID=1 and values[0] is an empty list
                - Return the values BEFORE ReLUs (to also calculate the speed of neurons that toggle OFF-ON)
    """
    values = [[]]
    for i in range(len(weights)):
        x = x@weights[i] + biases[i]
        values.append(np.copy(x)) # collect values before ReLUs
        x[x<0] = 0.0
    return values

def get_target_layer_output_norm_after_ReLU(x, dx, weights, biases, layerId):
    F,b = blackbox.getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x) # transformations from input to layerId
    y0 = (x@F + b) # target layer output at x
    dy = (dx@F) # size of the wiggle produced by dx
    # logger.debug(f"Number of OFF neurons: \t {np.sum([y0<=0.0])}/{len(dy)}")
    dy[y0<=0.0] = 0.0
    return np.linalg.norm(dy)

def get_target_layer_average_entries_without_neuronId(x, dx, weights, biases, layerId, neuronId):
    F,b = blackbox.getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x) # transformations from input to layerId
    y0 = (x@F + b) # target layer output at x
    dy = (dx@F) # size of the wiggle produced by dx
    dy[y0<=0.0] = 0.0
    dy[neuronId] = 0.0
    dy = np.abs(dy)
    dy_nonzero = [entry for entry in dy if entry!=0.0]
    return np.mean(dy_nonzero)

# ---------------------------------------------------
# Functions related to hyperplanes and angles
# ---------------------------------------------------

def project_onto_hyperplane(vector, normal):
    scalar_projection = np.dot(vector, normal) / np.dot(normal, normal)
    vector_projection = scalar_projection * normal
    return vector - vector_projection

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degree between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def check_if_on_ReLU_hyperplane(x0, weights, biases, layerId, neuronId, tol, raise_exception=False):
    logger = logging.getLogger('root')
    # Check if we are still on the ReLU hyperplane.
    # If we are the target neuron output is below the tolerance:
    targetOut0 = np.abs(get_target_neuron_output_at_x(x0, weights, biases, layerId, neuronId))
    is_ok = (targetOut0 <= tol)

    if is_ok:
        add = f"CONTROL: \t OK! \t (target neuron output < tolerance: |{targetOut0:.3E}|<{tol:.3E})"
    else:
        add = f"CONTROL: \t WARNING! \t (target neuron output > tolerance: |{targetOut0:.3E}|>{tol:.3E})"
        if raise_exception: raise Exception(add)

    logger.debug(f"{add}")
    return is_ok


from typing import Literal

_DX_TYPES = Literal['along_decision_boundary',
                    'orthogonal',
                    'optimal_EC24', 
                    'along_decision_boundary_with_prop', 
                    'perfect_control_along_decision_boundary']

class ExperimentException(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

def get_m(x, weights, biases, use_strongest_output_port):
    m,_ = blackbox.getLocalMatrixAndBias(weights, biases, x)
    cID = get_classID(weights, biases, x) if use_strongest_output_port else 0
    if N_CID == 1: 
        cID = cID[0]
        return m[:,cID]
    if N_CID == 2: 
        return m[:,cID[0]] - m[:,cID[1]]

# =========== Obtain walking direction ===========
def get_dx(x, eps, weights, biases, layerId, neuronId, choose_dx, use_strongest_output_port, model=None):
    n,_ = blackbox.getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x)
    n = n[:,neuronId]
    # m,_ = blackbox.getLocalMatrixAndBias(weights, biases, x)
    m = get_m(x, weights, biases, use_strongest_output_port)

    def get_Mb_of_layerId(x): 
        weights_sig, biases_sig = whitebox.getSignatures(model, layerId)
        M,b = blackbox.getLocalMatrixAndBias(weights_sig[:layerId], biases_sig[:layerId], x)
        # if layerId > 1:
        #     M,b = blackbox.getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x) # transformation from input to layerId
        # else:
        #     M,b = np.identity(x.shape[0]), np.zeros(x.shape[0]) # special case for first hidden layer
        return M,b
    
    def computePreimageDelta(x, dlx):
        # Propagate dlx back to the network input: 
        M, _ = get_Mb_of_layerId(x)
        return dlx@np.linalg.pinv(M)

    if choose_dx=='along_decision_boundary': # Project the above direction onto the decision boundary
        # BEFORE
        #cID = get_classID(weights, biases, x) if use_strongest_output_port else 0
        # dx = project_onto_hyperplane(n, m[:,cID])
        # AFTER: 
        dx = project_onto_hyperplane(n, m)
    elif choose_dx == 'orthogonal': # Walk perpendicular to ReLU hyperplane
        dx = n.copy()
    elif choose_dx == 'optimal_EC24': # Optimal wiggle from EC24
        if layerId > 1:
            Fm1,_ = blackbox.getLocalMatrixAndBias(weights[:layerId-1], biases[:layerId-1], x) # transformation from input to layerId-1
            invF = np.linalg.pinv(Fm1)
        else:
            Fm1,_ = np.identity(x.shape[0]), np.zeros(x.shape[0]) # special case for first hidden layer
            invF = Fm1
        sig = weights[layerId-1][:, neuronId] # signature of target neuron
        mt = m@Fm1 # propagate m to the target layer
        dx = project_onto_hyperplane(sig, mt)
        dx = dx@invF # optimal wiggle is sig*F^-1
    elif choose_dx == 'optimal_along_decision_boundary':
        if layerId > 1:
            Fm1,bm1 = blackbox.getLocalMatrixAndBias(weights[:layerId-1], biases[:layerId-1], x) # transformation from input to layerId-1
            y = x@Fm1 + bm1
            Fm1[:, np.where(y <= 0)] = 0
            invF = np.linalg.pinv(Fm1)
        else:
            # special case for first hidden layer
            Fm1 = np.identity(x.shape[0])
            invF = Fm1
        sig = weights[layerId-1][:, neuronId] # signature of target neuron
        dx = sig@invF # optimal wiggle
        dx -= np.dot(dx,m)*m/np.dot(m,m) # project onto the decision boundary
    elif choose_dx == 'perfect_control_along_decision_boundary': 
        # In particular, let ei be the unit vector in the space Rd1.
        # Number of neurons in target layer
        nNeurons = len(biases[layerId-1])
        ei = np.zeros(nNeurons)
        # COMMENT: Slight deviation from Carlini: We are not using the unit vector, but a scaled unit vector of size epsilon
        ei[neuronId] = eps
        dx = computePreimageDelta(x, ei)
        # weights, biases = whitebox.getSignatures(model, args.layerID)
        # cID = get_classID(weights, biases, x) if use_strongest_output_port else 0
        # dx = project_onto_hyperplane(dx, m[:,cID])
        dx = project_onto_hyperplane(dx, m)
    else:
        raise Exception(f"Parameter choose_dx='{choose_dx}', but has to be one of {_DX_TYPES}.")
    dx = dx * eps / np.abs(get_target_neuron_output(x, dx, weights, biases, layerId, neuronId))
    return dx 

def analyze_x_dual(x_dual, weights, biases, layerId, neuronId, 
                tol, eps,
                analyze_wiggle_sensitivity = False,
                analyze_speed = False,
                handle_previous_layer_toggles = True,
                collect_n = 1,
                choose_dx: _DX_TYPES = 'along_decision_boundary_with_prop',
                model = None, 
                ndebug = False,):

    INFINITY = 1e10
    results = {'CHOOSE_DX': choose_dx}

    if not ndebug:
        logger = logging.getLogger('root')
        # Confirm that the dual point is indeed on the ReLU hyperplane:
        is_ok = check_if_on_ReLU_hyperplane(x_dual, weights, biases, layerId, neuronId, tol, raise_exception=False)
        message = "CONTROL DUAL POINT: Check that target neuron output < tolerance ({tol:.3E}) at x_dual."
        logger.debug(message)
        if not is_ok: raise ExperimentException(message)

    # =========== NORMAL VECTOR OF RELU HYPERPLANE ===========
    # transformations from input to layerId
    F,_ = blackbox.getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x_dual)
    n = F[:,neuronId].copy()
    n = n / np.linalg.norm(n)

    for (side, side_sign) in [('ON', 1), ('OFF', -1)]:

        # =========== Adjust INFINITY on the OFF side if we have perfect control ===========
        # If we have perfect control over the target layer, there will NEVER be a future toggle on the OFF side. 
        # If we leave the value for infinity extremely high, and encounter many past layer toggles, the algorithm
        # ...would therefore never finish. 
        if (side == 'OFF') and (choose_dx=='perfect_control_along_decision_boundary'): 
            INFINITY = distance * 3 # If we have walked to 3x the distance on the previous side, consider this point infinity.

        # =========== Initial point displaced by small amount ===========
        xA = x_dual + side_sign*n*eps

        # =========== Check that neuron is indeed ON/OFF on the expected sides ===========

        if not ndebug:
            targetOut = get_target_neuron_output_at_x(xA, weights, biases, layerId, neuronId)
            if (np.sign(targetOut) != side_sign):
                raise ExperimentException(f"""Target Neuron Output Control FAILED:
                                Observed neuron value {targetOut:.3E} on {side} side.""")

        cID = get_classID(weights, biases, xA)
        dx = get_dx(xA, eps, weights, biases, layerId, neuronId, choose_dx, True, model=model)
        # =========== Scale dx and ensure it points in the right direction ===========
        if np.sign(np.dot(dx, n)) != side_sign: dx = -dx
        
        if not ndebug:
            dx
            targetOut = get_target_neuron_output_at_x(x_dual+dx, weights, biases, layerId, neuronId)
            if (np.sign(targetOut) != side_sign):
                raise ExperimentException(f"""Target Neuron Output Control FAILED:
                                Observed neuron value {targetOut:.3E} on {side} side.""")

        # =========== Analyze sensitivity to wiggle ===========
        if analyze_wiggle_sensitivity:
            results['vL'+side] = get_target_layer_output_norm_after_ReLU(xA, dx, weights, biases, layerId)
            cross_norm_expected = eps/np.sqrt(len(weights[layerId]))
            if not ndebug:
                logger.debug(f"{side}-Side  \t Expected target neuron change: {eps:.3E} \t Observed target neuron change: |{get_target_neuron_output(xA, dx, weights, biases, layerId, neuronId):.3E}|")
                logger.debug(f"{side}-Side \t Expected non-target neuron change: {cross_norm_expected:.3E} \t Observed non-target neuron change: {get_target_layer_average_entries_without_neuronId(xA, dx, weights, biases, layerId, neuronId):.3E}")
        # =========== Angle between relu and walking direction ===========
        results['dx'+side+'Angle'] = 90.0 - angle_between(dx, side_sign*n)

        # =========== Distance to another relu hyperplane ===========
        total_past_neurons = sum([w.shape[-1] for w in weights[:layerId]])
        total_future_neurons = sum([w.shape[-1] for w in weights[layerId:]])
        future_relus_crossed = 0
        past_relus_crossed = 0
        same_relu_crossed = 0
        angle = 0 
        distance = 0
        n0 = n.copy()
        x = xA.copy()
        while future_relus_crossed < collect_n:
            xB = find_point_of_boundary(weights, biases, x, dx, eps=tol, with_output_classes=True)
            fromFuture = toggle_states_equal(weights[:layerId], biases[:layerId], x, xB) if np.linalg.norm(xB)<INFINITY else True
            distance_current_patch = np.linalg.norm(xB-x) if np.linalg.norm(xB)<INFINITY else INFINITY
            if fromFuture:
                future_relus_crossed += 1
                past_relus_crossed = 0
                # If the distance in the current patch is close to the numerical precision tolerance, 
                # ... don't count the toggle as a proper future toggle after all:
                if distance_current_patch < eps: future_relus_crossed -= 1
            if distance_current_patch < eps: same_relu_crossed += 1 
            else:
                past_relus_crossed += 1
                if not handle_previous_layer_toggles:
                    raise ExperimentException("Toggle is not valid. Discard investigation.")
                if past_relus_crossed > max(3*total_past_neurons/total_future_neurons, 10):
                    collect_n = future_relus_crossed
                    if not ndebug: logger.debug("Got stuck in too many past-layer relus.")
                    break
                    # raise ExperimentException("Got stuck in too many past-layer relus.")
            if same_relu_crossed > 3: 
                collect_n = future_relus_crossed
                if not ndebug: logger.debug("Got stuck in too many past-layer relus.")
                break
                # raise ExperimentException("Got stuck in the same ReLU too many times.")
            distance += distance_current_patch   # (3)raw distance of each patch
            if not ndebug: toggled = which_neuron_toggled(weights, biases, x, xB, with_output_classes=True)
            #distance += np.dot(xB-x, n)/np.linalg.norm(n) # (2)distance perpendicular to the new relu of each patch
            # =========== Record patch distances =============
            if fromFuture and (distance_current_patch > eps): 
                results['d'+side+str(future_relus_crossed)] = distance
                results['cID'+side+str(future_relus_crossed)] = cID
                if future_relus_crossed <= 1: 
                    dc = distance
                else: 
                    dc = distance - results['d'+side+str(future_relus_crossed-1)]
                results['d'+side+'_patch_'+str(future_relus_crossed)] = dc
            # =========== Analyze sensitivity to wiggle in current patch ===========
            if analyze_wiggle_sensitivity and fromFuture:
                results['vL'+side+str(future_relus_crossed)] = get_target_layer_output_norm_after_ReLU(x, dx, weights, biases, layerId)
                results['vLcross'+side+str(future_relus_crossed)] = get_target_layer_average_entries_without_neuronId(x, dx, weights, biases, layerId, neuronId)
                #cross_norm_expected = eps/np.sqrt(len(weights[layerId]))
                # if not ndebug:
                #     logger.debug(f"{side}-Side  \t Expected target neuron change: {eps:.3E} \t Observed target neuron change: |{get_target_neuron_output(x, dx, weights, biases, layerId, neuronId):.3E}|")
                #     logger.debug(f"{side}-Side \t Expected non-target neuron change: {cross_norm_expected:.3E} \t Observed non-target neuron change: {get_target_layer_average_entries_without_neuronId(x, dx, weights, biases, layerId, neuronId):.3E}")
            # =========== Analyze speed in current patch ===========
            if analyze_speed and fromFuture:
                val_xA = get_neuron_values(x, weights, biases)
                val_xB = get_neuron_values(xB, weights, biases)
                val_xA = np.concatenate(val_xA[layerId+1:]) # list of all future values at initial point
                val_xB = np.concatenate(val_xB[layerId+1:])
                # the following original lines don't work on inhomogeneous width of future layers: 
                # val_xA = np.array(val_xA[layerId+1:]).flatten()  # list of all future values at initial point
                # val_xB = np.array(val_xB[layerId+1:]).flatten() # list of all future values at final point
                speed = (val_xB - val_xA) / distance_current_patch # results['d'+side] # obtain 'speed' dividing by distance
                results['speed_'+side+'_mean'+str(future_relus_crossed)] = np.mean(np.abs(speed))
                #results['speed_'+side+'_max'] = np.max(np.abs(speed))
                speed_filtered = speed[np.sign(val_xA) != np.sign(speed)] # only consider neurons that move towards a toggling point
                results['speed_'+side+'_dirT'+str(future_relus_crossed)] = np.mean(np.abs(speed_filtered))
            #====================================================================
            # =========== Prepare for walking to the next future ReLU ===========
            x = xB.copy()
            dx0 = dx.copy() # take note of the original dx direction for debugging purposes
            nold = n.copy()
            n,_ = blackbox.getLocalMatrixAndBias(weights[:layerId], biases[:layerId], x)
            n = n[:,neuronId]
            m,_ = blackbox.getLocalMatrixAndBias(weights, biases, x)
            cID = get_classID(weights, biases, x)
            #dx = get_dx(x, weights, biases, layerId, neuronId, choose_dx)
            dx = get_dx(x, eps, weights, biases, layerId, neuronId, choose_dx, True, model=model)
            # =========== Scale dx and ensure it points in the right direction ===========
            if np.sign(np.dot(dx, n)) != side_sign: dx = -dx
            new_angle = angle_between(dx0, dx)
            if not ndebug: logger.debug(f"Side: {side}\t Angle(dx0, dx): {new_angle:.2f} deg \t Angle(n, nold): {angle_between(n, nold):.2f} deg \t Distance in patch: {distance_current_patch:.3E} \t Total distance: {distance:.3E} \t Future toggles: {future_relus_crossed} \t Past toggles: {past_relus_crossed} \t Toggled: {toggled} \t class: {cID}")


        results['nT'+side] = future_relus_crossed
        results['d'+side] = distance

    # ========== Discard if no toggles were recorded ===========
    if (results['nTON']==0) or (results['nTOFF']==0): 
        message = f"DISCARD POINT! No toggles were recorded successfully: results['nTON']={results['nTON']}, results['nTOFF']={results['nTOFF']}."
        if not ndebug: logger.debug(message)
        raise ExperimentException(message)

    # =========== Adjust dON if necessary ===========
    nT = results['nTOFF']
    if results['nTON'] > results['nTOFF']: 
        if not ndebug: 
            logger.debug(f"Adjust dON, since there were more toggles on the on-side than the off-side: results['nTON']={results['nTON']} > results['nTOFF']={results['nTOFF']}. \n results['dON']={results['dON']} will be adjusted to value at nT={nT}: {results[f'dON{nT}']}")
            logger.debug(f"")
        results['dON'] = results[f'dON{nT}']

    # =========== Comparing ON vs OFF results ===========
    results['SUCCESS'] = results['dOFF'] > results['dON']
    # =========== Additional analysis ===========
    if not ndebug: 
        logger.debug("Check patch-by-patch distances for success (dOFF_patch>dON_patch) and ratio (dOFF_patch/dON_patch):")
        votes_off_larger_on = []
        for nt in range(1, nT+1): 
            success_nt = results['dOFF_patch_'+str(nt)] > results['dON_patch_'+str(nt)]
            votes_off_larger_on.append(success_nt)
            ratio_nt = (results['dOFF_patch_'+str(nt)]-results['dON_patch_'+str(nt)])/np.abs(results['dOFF_patch_'+str(nt)]+results['dON_patch_'+str(nt)])
            logger.debug(f"\t nT={nt:^3} \t success? {str(success_nt):^10} \t (dON-dOFF)/|dON+dOFF|: {ratio_nt*100:.2f} \t cIDON = {results['cIDON'+str(nT)]}, cIDOFF = {results['cIDOFF'+str(nT)]}")
        logger.debug(f"===> #(dOFF_patch>dON_patch) = {np.sum(votes_off_larger_on)}/{len(votes_off_larger_on)} = {np.sum(votes_off_larger_on)/len(votes_off_larger_on)*100:.2f}%")
    if analyze_wiggle_sensitivity:
        results['EXP_LAYER_CONTROL_OK'] = results['vLON'+str(nT)]>results['vLOFF'+str(nT)]
        results['||vON||/||vOFF||'] = results['vLON'+str(nT)]/results['vLOFF'+str(nT)]
        results['vLcrossON'] = results['vLcrossON'+str(nT)]
        results['vLcrossOFF'] = results['vLcrossOFF'+str(nT)]
        if not ndebug: 
            logger.debug(f"analyze_wiggle_sensitivity: results['EXP_LAYER_CONTROL_OK'] at nT={nT}: {results['EXP_LAYER_CONTROL_OK']} \t results['||vON||/||vOFF||'] at nT={nT}: {results['||vON||/||vOFF||']}")
            #if (not results['EXP_LAYER_CONTROL_OK']) and (not ndebug):
            logger.debug("\t Check where the control breaks down:")
            for nt in range(1, nT+1): 
                logger.debug(f"\t nT={nt}: \t control ok? ||vON||/||vOFF||={results['vLON'+str(nt)]/results['vLOFF'+str(nt)]:.2f}")
    if analyze_speed:
        results['INTUITION_speed_OK'] = results['speed_ON_mean'+str(nT)]>results['speed_OFF_mean'+str(nT)]
        results['INTUITION_speed_dirT_OK'] = results['speed_ON_dirT'+str(nT)]>results['speed_OFF_dirT'+str(nT)]
        if not ndebug: logger.debug(f"results['INTUITION_speed_dirT_OK'] = results['speed_ON_dirT']>results['speed_OFF_dirT'] at nT={str(nT)}: {results['INTUITION_speed_dirT_OK']} = {results['speed_ON_dirT'+str(nT)]:.3E}>{results['speed_OFF_dirT'+str(nT)]:.3E}")
        #if (not results['INTUITION_speed_dirT_OK']) and (not ndebug): 
        if not ndebug:
            logger.debug("\t Check intuition on speed for all future neurons (MEAN_ALL) and only the ones moving towards their toggling points (MEAN_DIR):")
            for nt in range(1, nT+1): 
                ratio_dirT = results['speed_ON_dirT'+str(nt)] / results['speed_OFF_dirT'+str(nt)]
                ratio_meanT = results['speed_ON_mean'+str(nt)] / results['speed_OFF_mean'+str(nt)]
                logger.debug(f"""\t nT={nt}: \t MEAN_DIR ON>OFF? {results['speed_ON_dirT'+str(nt)]>results['speed_OFF_dirT'+str(nt)]} \
                (speed ON/OFF: {ratio_dirT:.2f}) \
                \t MEAN_ALL ON>OFF? {results['speed_ON_mean'+str(nT)]>results['speed_OFF_mean'+str(nT)]} (speed ON/OFF: {ratio_meanT:.2f})""")

    if not ndebug:
        ratioOFFON = results['dOFF']/results['dON']
        logger.debug(f"dOFF/dON = {ratioOFFON:.1f}")
        logger.debug(results)

    return results

def get_confidence(votes_m, votes_p): 
    # Check confidence
    if (votes_m==0) and (votes_p==0): 
        return 0.0
    N = max(votes_p, votes_m)
    n = min(votes_p, votes_m)
    logp = -2*(n+N)*(0.5 - n/(n+N))**2 
    if np.isnan(logp): logp = 0.0
    return logp 

def analyze_df(df): 
    
    nTOFF = np.max(df.nTOFF.values)
    nTON = np.max(df.nTON.values)
    nT = min(nTON, nTOFF)

    for i in range(1, nT+1): 
        df[f"dOFF>dON{i}"] = df[f"dOFF_patch_{i}"] > df[f"dON_patch_{i}"]
        df[f"dOFF<dON{i}"] = df[f"dOFF_patch_{i}"] < df[f"dON_patch_{i}"]

    df[f"votesOFF>ON"] = df[[f"dOFF>dON{i}" for i in range(1, nT+1)]].sum(1)
    df[f"votesOFF<ON"] = df[[f"dOFF<dON{i}" for i in range(1, nT+1)]].sum(1)

    #df['perc_diff'] = np.abs(df.dOFF-df.dON) / (df.dOFF + df.dON) * 100.
    #df = df[df.perc_diff > PERC_DIFF_LIM]

    votes_p = np.sum(df[f"votesOFF>ON"]) # np.sum(df.dOFF > df.dON)
    votes_m = np.sum(df[f"votesOFF<ON"]) # np.sum(df.dOFF < df.dON)
    logp = get_confidence(votes_m, votes_p)
    # # Check confidence
    # N = max(votes_p, votes_m)
    # n = min(votes_p, votes_m)
    # logp = -2*(n+N)*(0.5 - n/(n+N))**2 
    # if np.isnan(logp): logp = 0.0
    
    t_median = np.median(df.subpoint_time_seconds)
    nExp = len(df)
    nDualPoints = df.dual_point_id.values[-1]
    ratio = np.mean(df.dOFF) / np.mean(df.dON)
    # ratio = np.mean(df[[f"dOFF_patch_{i}" for i in range(1, nT+1)]]) / np.mean(df[[f"dON_patch_{i}" for i in range(1, nT+1)]])
    # ratio = np.mean(df[[f"dOFF_patch_{i}" for i in range(1, nT+1)]] / df[[f"dON_patch_{i}" for i in range(1, nT+1)]])
    # votes = np.sum(df.dOFF > df.dON) - np.sum(df.dOFF < df.dON)
    votes = np.sum(df[f"votesOFF>ON"]) - np.sum(df[f"votesOFF<ON"])
    votes_g = np.sum(df.dOFF > df.dON) - np.sum(df.dOFF < df.dON)

    _results = {
        'ratio': ratio,
        'votes': votes,
        'votes_g': votes_g,
        'final_logp': logp,
        'ratio_success': True if ratio>1.0 else False,
        'votes_success': True if votes>0.0 else False,
        'nExp': nExp,
        'nDualPoints': nDualPoints, 
        'timePointMedian': t_median,
        'nExp x timePointMedian': t_median * nExp, 
        'timeTotal': df.total_execution_time.values[-1],
        }

    return _results

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main(argv):

    t0 = time.time() # start timer

    args = common.parseArguments(argv)

    print(f"Parsed arguments for sign recovery: \n\t {args}.")

    model = tf.keras.models.load_model(args.model)

    # ---------------------------------------------------
    # Setup File Paths
    # ---------------------------------------------------
    modelname = args.model.split('/')[-1].replace('.keras', '')
    savePath = common.getSavePath(modelname, args.layerID, args.neuronID)

    # ---------------------------------------------------
    # Prepare logging
    # ---------------------------------------------------
    import logging
    import sys
    if not eval(args.nDebug) :
        logname = savePath + 'log.log'
        print(f"Log information will be saved to {logname}.")
        logging.basicConfig(filename=logname, #stream=sys.stdout, #
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            force=True,
                            level=logging.DEBUG)
        logging.StreamHandler(sys.stdout)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    print(args)

    # ---------------------------------------------------
    # Load model, weights and biases (whitebox)
    # ---------------------------------------------------
    model = tf.keras.models.load_model(args.model)
    neuronId = int(args.neuronID)
    layerId = int(args.layerID)

    # Whitebox:
    weights, biases = whitebox.getWeightsAndBiases(model, range(1, len(model.layers)))

    # ---------------------------------------------------
    # Check that we can attack the desired layer
    # ---------------------------------------------------
    if args.layerID > len(weights):
        raise Exception(f"The provided target layer ID {args.layerID} exceeds the length of weights ({len(weights)}) in the model ")
    if args.layerID == 0:
        raise Exception(f"Only hidden layers can be attacked, not the input layer (layerID={args.layerID}). Please pass a layerID>0.")
    if args.layerID == len(weights):
        raise Exception(f"The output layer with layerID({args.layerID})=len(weights) cannot be attacked using this method.")
    if args.layerID == (len(weights)-1):
        print(f"WARNING: The penultimate layer with layerID={args.layerID} should not be attacked with this method (since all toggles have to be in future layers).")

    if not eval(args.nDebug) : logger.debug(f"Check that model weights are dtype tf.float64: \t {model.weights[0].dtype}")

    # ---------------------------------------------------
    # Set parameters
    # ---------------------------------------------------
    eps= 1e-6
    tol= 1e-6
    analyze_wiggle_sensitivity = eval(args.analyzeWiggleSensitivity)
    analyze_speed = eval(args.analyzeSpeed)
    handle_previous_layer_toggles = eval(args.handlePrevLayerToggles)
    ndebug = eval(args.nDebug)

    if not ndebug:
        logger.info(f"eps: {eps}")
        logger.info(f"tol: {tol}")
        logger.info(f"explore_dual_space: {args.exploreDualSpace}")

    # ---------------------------------------------------
    # Main Part
    # ---------------------------------------------------
    nExp = 0
    fail = 0
    nExpMin = args.nExpMin
    logp = 0
    results = []

    # Initial call to print 0% progress
    printProgressBar(0, args.nExp, prefix = 'Progress:', suffix = 'Complete', length = 50)

    # Load dual points from file: 
    dual_point_id = 0
    if not ndebug: logger.debug(f"Load dual points from file: {args.filepath_load_x0}.")
    X_DUAL = np.load(args.filepath_load_x0)
    def get_dual_point(dual_point_id): 
        if dual_point_id < len(X_DUAL):
            x_dual = X_DUAL[dual_point_id]
        else: 
            x_dual = None
        dual_point_id += 1
        return x_dual, dual_point_id


    while nExp < args.nExp:

        x_dual, dual_point_id = get_dual_point(dual_point_id)
        if x_dual is None: 
            print(f"WARNING: Could not complete {args.nExp} experiments; got only {nExp} experiments with the amount of dual points provided.")
            break 

        if not ndebug: logger.debug(f"\n\n=================== Experiment ID #{nExp} \t Dual Point ID: {dual_point_id} ===================")

        # =========== ANALYZE THE DUAL POINT ===========
        try:
            #
            t1 = time.time()
            analysis = analyze_x_dual(x_dual, weights, biases, layerId, neuronId, tol, eps,
                                        analyze_speed = analyze_speed,
                                        analyze_wiggle_sensitivity = analyze_wiggle_sensitivity,
                                        handle_previous_layer_toggles = handle_previous_layer_toggles,
                                        collect_n = args.nToggles,
                                        choose_dx = args.choose_dx,
                                        model = model, 
                                        ndebug = ndebug, )
            analysis['nExp'] = (nExp+1)
            analysis['dual_point_id'] = dual_point_id
            analysis['subpoint_time_seconds'] = time.time()-t1
            analysis['total_execution_time'] = time.time()-t0
            analysis['logp'] = logp
            results.append(analysis)

            nExp += 1
            df = pd.DataFrame(results)
            logp = analyze_df(df)['final_logp']
            tpassednow = time.time() - t0 # check runtime
            if (logp < -3.6889) and (nExp >= nExpMin): # logp < -2.9957: probability of wrong guess is less than 5%
                printProgressBar(nExp, args.nExp, prefix = 'Progress:', suffix = f"Completed early with 95% confidence! \t({int(100*fail/(nExp+fail))}% exclusions, logp={int(100*logp)/100}, Seconds passed: {tpassednow:.0f}))     \n", length = 50)
                break 

        except ExperimentException as ex:
            fail += 1
            if not ndebug: logger.debug(str(ex))

        tpassednow = time.time() - t0 # check runtime
        printProgressBar(nExp, args.nExp, prefix = 'Progress:', suffix = f"Complete\t({int(100*fail/(nExp+fail))}% exclusions, logp={int(100*logp)/100}, Seconds passed: {tpassednow:.0f}, nExp: {nExp}, DualPointID: {dual_point_id})     ", length = 50)

    # ---------------------------------------------------
    # Save Results
    # ---------------------------------------------------
    df = pd.DataFrame(results)
    df['dOFF/dON'] = df.dOFF / df.dON
    df['Vote dOFF>dON'] = df.dOFF > df.dON
    filename_md = savePath+'df.md'
    print(f"Results were saved to: \t {filename_md}.")
    df.to_markdown(filename_md)
    filename_csv = savePath+'df.csv'
    print(f"Results were saved to: \t {filename_csv}.")
    df.to_csv(filename_csv)
    df.to_pickle(savePath+'df.pkl')

    # ---------------------------------------------------
    # Print Results
    # ---------------------------------------------------
    table_with_results = {'layerID': layerId, 'neuronID': neuronId} | analyze_df(df)
    print(pd.DataFrame([table_with_results]).to_markdown())

    runtime = df['total_execution_time'].values[-1]

    if analyze_wiggle_sensitivity:
        print(f"layer {layerId} - neuron {neuronId} \t (runtime: {runtime:.0f} sec)", f"\t ||vON||/||vOFF||: {np.mean(df['||vON||/||vOFF||']):.2f}+-{np.std(df['||vON||/||vOFF||']):.2f}", f"\t Success rate: {df.SUCCESS.sum()/len(df)*100:.2f}%")
    else:
        print(f"layer {layerId} - neuron {neuronId} \t (runtime: {runtime:.0f} sec)", f"\t Success rate: {df.SUCCESS.sum()/len(df)*100:.2f}%")

    # ---------------------------------------------------
    # Create Figures
    # ---------------------------------------------------
    number_of_graphs = 2
    if analyze_wiggle_sensitivity: number_of_graphs += 1
    current_graph = 0
    nrows, ncols = (number_of_graphs+2)//3, 3
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols,nrows*4),
                        sharex=False, sharey=False,
                        gridspec_kw={'hspace': 0.3, 'wspace': 0.2})

    if number_of_graphs <= 3: ax = [ax] # ensure ax always has 2 indices
    plt.suptitle(f"Path: {savePath}; target neuron {neuronId}", fontsize=8)

    # ------------ Sensitivity graph ------------
    if analyze_wiggle_sensitivity:
        plt.sca(ax[(current_graph//3)][current_graph%3])
        current_graph += 1
        running_ratio_v = np.array([np.mean(df['||vON||/||vOFF||'][:i]) for i in np.arange(1, len(df))])
        plt.plot(running_ratio_v, label=f"{running_ratio_v[-1]:.2f}")
        plt.title('target layer output norm ratio')
        plt.ylabel('mean(||vON||/||vOFF||')
        plt.xlabel('dual point number')
        plt.axhline(1.0, c='grey')
        plt.legend()
        plt.grid()

    # ------------ Distance ratio graph ------------
    plt.sca(ax[(current_graph//3)][current_graph%3])
    current_graph += 1
    dOFF_running_mean = np.array([np.mean(df.dOFF[:i]) for i in np.arange(1, len(df))])
    dON_running_mean = np.array([np.mean(df.dON[:i]) for i in np.arange(1, len(df))])
    running_ratio = dOFF_running_mean/dON_running_mean
    plt.plot(running_ratio, label=f"{running_ratio[-1]:.2f}")
    plt.title('toggling distance OFF/ON ratio')
    plt.ylabel('mean(dOFF) / mean(dON)')
    plt.xlabel('dual point number')
    plt.axhline(1.0, c='grey')
    plt.grid()
    plt.legend()

    # ------------ Distance votes graph ------------
    plt.sca(ax[(current_graph//3)][current_graph%3])
    current_graph += 1
    dfOFF_running_votes = np.array([np.sum(df['Vote dOFF>dON'][:i])-np.sum(~df['Vote dOFF>dON'][:i]) for i in np.arange(1, len(df))])
    plt.plot(dfOFF_running_votes)
    plt.title('running votes: dOFF>dON')
    plt.ylabel('#(votes dOFF>dON) - #(votes dON>dOFF)')
    plt.xlabel('dual point number')
    plt.axhline(0.0, c='grey')
    plt.grid()

    plt.savefig(savePath+'figures.png')

if __name__=='__main__':
    main(sys.argv[1:])
