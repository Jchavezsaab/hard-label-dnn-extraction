"""
Collection of functions and parameters for analysing a Deep Neural Network
(DNN) as a white box.
"""

import numpy as np
from keras import Model
from blackbox import getHiddenVector,getLocalMatrixAndBias

def getWeightsAndBiases(model, layers):
    weights = []
    biases = []
    for l in layers:
        w, b = model.get_layer(index=l).get_weights()
        weights.append(np.copy(w))
        biases.append(np.copy(b))
    return weights, biases

def getRealSigns(model, layerID):
    weights, biases = getWeightsAndBiases(model, range(1, layerID + 1))
    signsLayer = np.sign(weights[-1][0])
    return signsLayer

def getSignatures(model, layerID):
    """Simulates the signature recovery and returns the corresponding weights, biases."""
    weights, biases = getWeightsAndBiases(model, range(1, layerID + 1))
    signsLayer = np.sign(weights[-1][0])
    weights[-1] = signsLayer[np.newaxis, :] * weights[-1]
    biases[-1] = signsLayer * biases[-1]
    return weights, biases

def signIsCorrect(neuronID, w, w0):
    return (w[:,neuronID]==w0[:,neuronID]).all()

def getScrambledSigns(w, b):
    w = w.copy()
    b = b.copy()
    nNeurons = w.shape[-1]
    #------------------------------
    # my sign guess and starting point
    #------------------------------
    # as a starting point, we assume that half of the signatures have wrong signs
    for nID in range(nNeurons):
        sign = np.random.choice([+1, -1])
        w[:,nID] = sign * w[:,nID]
        b[nID] = sign * b[nID]
    return w, b

def toggleSign(neuronID, w, b): 
    w[:,neuronID] = (-1) * w[:,neuronID]
    b[neuronID] = (-1) * b[neuronID]
    return w, b

def getTogglingPoints(model, layerID, neuronID, funcEps): 
    """Find at which `epsilon` values a function of epsilon `funcEps` leads to the toggling of a specific neuron
    `neuronID` in layer `layerID` of a TensorFlow model `model`.
    
    For example: 
    >>> funcEps = lambda x: deti.interpol.linearMorphEps(myfrog, mycar, x)
    >>> getTogglingPoints(model, layerID, neuronID, funcEps)
    """
    import scipy.optimize
    
    weights, bias = getNeuronWeightBias(model, layerID, neuronID)
    func = lambda x: getLiEquation(x, funcEps, weights, bias) # the neuron will be toggled when this equation is equal to zero.
    epsilons = scipy.optimize.fsolve(func, 0) 
    
    return epsilons

def getLiEquation(epsilon, funcMorphEpsilon, weights, bias):
    """Given the neurons `weights` w1...wn and `bias` b, return the equation
    
        w1 * p1 + ... + wn * pn + b,
        
    where the values of `p` are given by a morph function dependent on `epsilon` 
    
        (p1, ..., pn) = funcMorphEpsilon(epsilon).
    """
    pvec = funcMorphEpsilon(epsilon) # morphed image at position epsilon
    pvec = pvec.flatten()            # flattened morphed image
    LiEquation = np.dot(weights, pvec.flatten()) + bias
    return LiEquation

def getNeuronWeightBias(model, layerID, neuronID): 
    """Get the neuron weights and bias of neuron `neuronID` in layer `layerID` of a TensorFlow model.
    """
    
    weightsAndBiases = model.layers[layerID].weights
    
    weights = weightsAndBiases[0]
    weightsOfNeuron = weights.numpy()[:, neuronID]
    
    bias = weightsAndBiases[1]
    biasOfNeuron = bias.numpy()[neuronID]
    
    return weightsOfNeuron, biasOfNeuron

def getNeuronSignature(model, layerID, neuronID): 
    """
    Get the neuron signature of neuron `neuronID` in layer `layerID` of a TensorFlow model. 
    The neuron signature is obtained by dividing the weight of each incoming connection `w1...wn` by the weight of the 
    first connection `w1`, i.e. 
    
        (w1/w1, w2/w1, ..., wn/w1).
    
    To obtain the weights and biases themselves, please use getNeuronWeightBias. 
    """
    
    weightsOfNeuron, _ = getNeuronWeightBias(model, layerID, neuronID)
    
    w1 = weightsOfNeuron[0]
    return [w/w1 for w in weightsOfNeuron]

def getLayerOutputs(model, testInput, onlyLayerID=None):
    """
    For a neural network model, collect the intermediate outputs of all layers*  for a test input.
    
    *or only one particular layer identified by its `layerID` in model.layers via the  `onlyLayerID` parameter
    """
    
    outputOfAllLayers = []

    for layerID, layer in enumerate(model.layers):
        
        if onlyLayerID is not None and layerID != onlyLayerID:
            continue

        intermediateLayerModel = Model(inputs=model.input, outputs=model.get_layer(layer.name).output)
        intermediateOutput = intermediateLayerModel(testInput).numpy()
        outputOfAllLayers.append(intermediateOutput)
        
    if onlyLayerID is not None: outputOfAllLayers = outputOfAllLayers[0]

    return outputOfAllLayers

def findToggledNeuronsInLayer(model, layerID, interpolatedImages, debug=False):
    """
    For a given model find the toggled neurons in layer `layer_id` when moving from image x1 to x2
    via the interpolatedImages.
    
    Get the `interpolatedImages` by using (for example) the function `getInterpolatedImages`.
     
    Returns:
        An array that contains in which of the `n` steps which neuron was toggled.
        For example, the following output means that first neuron 12 was toggled in step 3007:
        array([[3007,   12],
               [6103,   19],
               [7742,    4],
               [8067,    2],
               [9543,   15],
               [9556,   15],
               [9557,   15]])
    """
     
    #-----------------------------------------------
    # Get layer outputs for interpolated images
    #-----------------------------------------------
    outputLayer = getLayerOutputs(model, interpolatedImages, onlyLayerID=layerID)
    
    #-----------------------------------------------
    # Analyze activity and toggling
    #----------------------------------------------- 
    activeInLayer = (outputLayer > 0).astype(int)        # find if the neuron was active or not
    toggled = np.diff(activeInLayer, axis=0)             # find toggling points for each neuron (axis=0)
    if debug:
        print(toggled)
    
    #-----------------------------------------------
    # Return which neuron was toggled in which step
    #-----------------------------------------------
    toggledStepNeuron = np.argwhere((toggled == 1) | (toggled == -1))
    return toggledStepNeuron

"""
A simple linear interpolation between two input images x1 and x2
"""
linearMorph = lambda x1, x2, i, steps: x1 + (x2 - x1) / steps * i

def getInterpolatedImages(x1, x2, morph=linearMorph, n=10_000):
    """
    Get the interpolated images between x1 and x2.
    
    morph: morph function to move from x1 to x2. Required functional form is morph(x1, x2, i, n),
    where `n` is the number of steps with which to move x1 into x2 and
    `i=0...n-1` is the current step id.
    """
    #-----------------------------------------------
    # Interpolate between x1, x2
    #-----------------------------------------------
    morphX = np.zeros((n,) + x1.shape)

    for i in range(n):
        morphX[i] = morph(x1, x2, i, n)
     
    return morphX

def collectWeightAndBiasLists(model, layerID):
    """Helper function: Collect lists of all previous layers weight and biases matrices up to (not including) layer `layerID`. 
    
    Returns: Ws, Bs (list of all numpy array weight matrices before layerID, list of all numpy array bias vectors before layerID)
    """
    
    Ws = []
    Bs = []
    
    # for all previous layers, collect the weights and biases: 
    for pID in range(0, layerID):
        
        weightsAndBiases = model.layers[pID].weights
        
        if len(weightsAndBiases) == 0: 
            continue
            
        w = weightsAndBiases[0]
        w = w.numpy()
        
        b = weightsAndBiases[1].numpy()
        
        Ws += [w] 
        Bs += [b]
        
    return Ws, Bs

def getOutputMatrixWhitebox(x, model, layerId, ReLUInOutFunc=False):
    weights, biases = getWeightsAndBiases(model, range(1, layerId + 1))
    # Output of layer layerId before ReLus
    y = getHiddenVector(weights, biases, layerId, x, relu=False)

    weights, biases = getWeightsAndBiases(model, range(layerId + 1, len(model.layers)))
    if ReLUInOutFunc:
        weights[0][y < 0] = 0
    else:
        y[y < 0] = 0
    o, b = getLocalMatrixAndBias(weights, biases, y)