# The hidden activation of the target, as the attack models it: ReLU (LEAKY_COEFF = 0) or LeakyReLU with negative
# slope LEAKY_COEFF, the same for every hidden unit.  Like the input dimension, it is read from the model's public
# description (oracle.leaky_coeff()), never from its weights.
#
# A unit with pre-activation z outputs relu(z) = z for z > 0 and LEAKY_COEFF z otherwise; its local slope is slope(z),
# 1 or LEAKY_COEFF.  With LEAKY_COEFF = 0 an "off" unit outputs exactly 0, and every caller keeps its ReLU-only shortcuts
# (masks, zeroed rows) bit for bit.  Read it as activation.LEAKY_COEFF: oracle.set_model() updates it when a script
# switches the model after importing this file.
import numpy as np

from oracle import leaky_coeff

LEAKY_COEFF = leaky_coeff()


def relu(z):
    """The activation, elementwise (numpy)."""
    if LEAKY_COEFF == 0:
        return np.maximum(z, 0)
    return np.where(z > 0, z, LEAKY_COEFF * z)


def slope(z):
    """d relu / dz, elementwise: 1 where z > 0, else LEAKY_COEFF (0 for ReLU), as float64."""
    if LEAKY_COEFF == 0:
        return (z > 0).astype(np.float64)
    return np.where(z > 0, 1.0, LEAKY_COEFF)


def relu_torch(x):
    """The activation of a torch tensor."""
    import torch.nn.functional as F
    if LEAKY_COEFF == 0:
        return F.relu(x)
    return F.leaky_relu(x, LEAKY_COEFF)
