# The oracle is the only file that the attack can access.

import numpy as np

_QUERIES = 0


def _load_weights():
    import tensorflow as tf
    m = tf.keras.models.load_model("../data/unitary_32_32x3_10_float64.keras")
    return [(l.get_weights()[0], l.get_weights()[1])
            for l in m.layers if len(l.get_weights()) == 2]


_W = None


def _forward64(x):
    import torch
    global _W
    if _W is None:
        _W = [(torch.tensor(w, dtype=torch.float64),
               torch.tensor(b, dtype=torch.float64)) for w, b in _load_weights()]
    t = torch.tensor(np.asarray(x, dtype=np.float64)).reshape(-1, 32)
    for i, (w, b) in enumerate(_W):
        t = t @ w + b
        if i < len(_W) - 1:
            t = t.clamp(min=0)
    return t.numpy()


def label(x):
    """Hard labels only. Scalar int for a single input, array for a batch."""
    global _QUERIES
    x = np.asarray(x)
    single = x.ndim == 1
    n = 1 if single else x.shape[0]
    _QUERIES += n
    out = _forward64(x)
    lab = np.argmax(out, axis=1).astype(np.int64)
    return int(lab[0]) if single else lab


def query_count():
    return _QUERIES


def reset_query_count():
    global _QUERIES
    _QUERIES = 0
    return 0
