"""
Collection of functions related to unitary balanced Deep Neural Networks (DNNs).
"""

import tensorflow as tf
# tf.keras.backend.set_floatx('float64')

import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Flatten


def randomUnitVector(n):
    """
    Generate a random vector of length n over the real numbers with 2-norm equal
    to 1.

    Parameters
    ----------
    n : int
        Length of the vector.

    Returns
    -------
    array
        Vector of length n with 2-norm equal to 1.
    """
    v = np.random.normal(size=n)
    return v / np.linalg.norm(v)

def newRandomModel(inputShape, neuronsHiddenLayers, outputs, norm):
    """
    Create a new DNN (Keras Model object) with the following characteristics:

        - The input has shape inputShape;
        - The number of hidden layers is the lenght of neuronsHiddenLayers and
          the number of neurons in each hidden layer is given by the
          corresponding entry in neuronsHiddenLayers;
        - The number of outputs is outputs;
        - The weights of each neuron are given by a vector chosen uniformly at
          random whose norm is norm, and
        - The bias of each neuron is zero.

    Parameters
    ----------
    inputShape : tuple
        The shape of the input to the DNN.
    neuronsHiddenLayers : array
        List with the number of neurons in each hidden layer.
    outputs : int
        Number of outputs.
    norm : float
        Norm of the weight vector of the neurons.

    Returns
    -------
    Model
        Model object corresponding to a DNN with the characteristics above.

    Example
    -------
        inputShape = (1024,)
        neuronsHiddenLayers = [256] * 4
        outputs = 10
        model = deti.randomdnn.newRandomModel(inputShape, neuronsHiddenLayers, outputs, 1.0)
    """
    
    # Input layer
    x0 = layers.Input(shape=inputShape, name="input")
    # Hidden layers
    x = x0
    for i, n in enumerate(neuronsHiddenLayers):
        x = Dense(units = n, activation=LeakyReLU(negative_slope=0.1), name=f"denseLayer{i + 1}",
                         kernel_initializer=initializers.Zeros(),
                         bias_initializer=initializers.Zeros())(x)
    # Output layer
    x = layers.Dense(outputs, name="output",
                     kernel_initializer=initializers.Zeros(),
                     bias_initializer=initializers.Zeros())(x)
    # Model
    model = Model(inputs=x0, outputs=x)
    
    # Set random vectors as weights
    for layer in model.layers:
        if type(layer) != layers.Dense:
            continue
        dim, n = layer.get_weights()[0].shape
        weights = np.array([norm * randomUnitVector(dim) for i in range(n)]).T
        layer.set_weights([weights, layer.get_weights()[1]])
    
    return model

def newRandomBalancedModel(inputShape, neuronsHiddenLayers, outputs, norm, low=-1, high=1, samples=100000):
    """
    Create a new DNN (Keras Model object) with the following characteristics:

        - The input has shape inputShape;
        - The number of hidden layers is the lenght of neuronsHiddenLayers and
          the number of neurons in each hidden layer is given by the
          corresponding entry in neuronsHiddenLayers;
        - The number of outputs is outputs;
        - The weights of each neuron are given by a vector chosen uniformly at
          random whose norm is norm, and
        - The bias of each neuron is chosen so that each neuron has a 50%
          probability of being active.

    Parameters
    ----------
    inputShape : tuple
        The shape of the input to the DNN.
    neuronsHiddenLayers : array
        List with the number of neurons in each hidden layer.
    outputs : int
        Number of outputs.
    norm : float
        Norm of the weight vector of the neurons.
    low : float, optional
        Lower bound for sampling inputs to the DNN when setting the bias.
    high : float, optional
        Upper bound for sampling inputs to the DNN when setting the bias.
    smaples : int, optional
        Number of sampled random inputs to the DNN when setting the bias.

    Returns
    -------
    Model
        Model object corresponding to a DNN with the characteristics above.

    Example
    -------
        inputShape = (1024,)
        neuronsHiddenLayers = [256] * 4
        outputs = 10
        model = deti.randomdnn.newRandomBalancedModel(inputShape, neuronsHiddenLayers, outputs, 1.0)
    """
    
    # DNN with random unitary vectors as weights and zero biases
    model = newRandomModel(inputShape, neuronsHiddenLayers, outputs, norm)
    
    # Sample random inputs
    Y = np.random.uniform(low=low, high=high, size=(samples, inputShape[0]))
    
    # Set bias such that around 50% of the samples activate each neuron
    for i in range(1, len(model.layers)):
        weights, _ = model.layers[i].get_weights()
        # The i-th column of Y contains the values for the i-th neuron
        Y = np.matmul(Y, weights)
        # Get biases from the median of the columns
        biases = -np.median(Y, axis=0)
        # Update biases on the model
        model.layers[i].set_weights([weights, biases])
        # Update Y for next layer
        Y += biases[np.newaxis, :]
        Y *= (Y > 0)
    
    return model

def pred(x, model):
    y = x.copy()
    for i in range(len(model.weights)//2):
        y = y@model.weights[2*i] + model.weights[2*i+1]
        if i < 4: y[y<0] *= 0.1
    return y

def leakydnn( hidden_sizes, num_classes):
    """
    Creates a Deep Neural Network (DNN) model using Leaky ReLU activation functions.

    Args:
        input_shape (tuple): The shape of the input data (e.g., (784,) for MNIST).
        num_classes (int): The number of output classes.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    hidden_layers = []
    for n in hidden_sizes:
        hidden_layers.append(layers.Flatten())
        hidden_layers.append(layers.Dense(n))
        hidden_layers.append(layers.LeakyReLU(negative_slope=0.1))

    model = models.Sequential([
        # Input layer
        layers.Input(shape=(32,32,3)),
    ] + hidden_layers + [

        # Output layer (softmax for multi-class classification)
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

    return model


# model = newRandomBalancedModel([32], [32]*4, 10, 1.0)
model = leakydnn([64]*64, 10)
print(model.layers[1].get_weights()[0])
model.save("cifar_64x64_float64.h5")