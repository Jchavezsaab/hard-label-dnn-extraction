"""
Collection of functions related to unitary balanced Deep Neural Networks (DNNs).
"""

import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU

def leakydnn(input_shape, hidden_sizes, num_classes):
    """
    Creates a Deep Neural Network (DNN) model using Leaky ReLU activation functions.

    Args:
        input_shape (tuple): The shape of the input data (e.g., (784,) for MNIST).
        num_classes (int): The number of output classes.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    hidden_layers = []
    for n in hidden_sizes:
        hidden_layers.append(layers.Dense(n))
        hidden_layers.append(layers.LeakyReLU(negative_slope=0.1))

    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
    ] + hidden_layers + [

        # Output layer (softmax for multi-class classification)
        layers.Dense(num_classes)
    ])

    
    # Set random vectors as weights
    for layer in model.layers:
        if type(layer) != layers.Dense:
            continue
        dim, n = layer.get_weights()[0].shape
        b = []
        for _ in range(dim):
            v = np.random.uniform(size=(n,))
            for bi in b:
                v -= np.dot(bi,v)*bi
            b.append(v/np.linalg.norm(v))
        weights = np.array(b)

        biases = np.zeros(n)
        layer.set_weights([weights, biases])

    # Sample random inputs
    Y = np.random.normal(size=(100000, input_shape[0]))
    
    # Set bias such that each layer's vector has norm 1 on average and around 50% of the samples activate each neuron
    for i in range(0, len(model.layers)):
        print(model.layers[i])
        if not isinstance(model.layers[i], Dense): continue
        weights, biases = model.layers[i].get_weights()
        Y = np.matmul(Y, weights)
        scale = np.linalg.norm(Y, axis=1).mean()
        weights = weights.copy()/scale
        Y /= scale
        biases = -np.median(Y,axis=0)
        # Update biases on the model
        model.layers[i].set_weights([weights, biases])
        # Update Y for next layer
        Y += biases[np.newaxis, :]
        Y[Y < 0] *= 0.1

    return model


model = leakydnn([64], [64]*4, 10)
model.save("../data/unitary_leaky_64_64x4_10_float64.keras")