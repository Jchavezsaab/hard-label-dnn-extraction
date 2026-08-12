# The black box.  The attack may use label() and query_count() and nothing else from this file.
import io
import os
import zipfile

import numpy as np

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "unitary_32_32x3_10_float64.keras")

_QUERIES = 0
_LAYERS = None


def _load_weights():
    """[(W (in, out), b (out,)) per Dense layer], read straight out of the .keras file (a zip holding an hdf5 file)."""
    import h5py
    with zipfile.ZipFile(MODEL) as archive:
        weights = h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r")
    names = sorted((name for name in weights["layers"] if name.startswith("dense")),
                   key=lambda name: int(name.split("_")[1]) if "_" in name else 0)
    layers = []
    for name in names:
        group = weights["layers"][name]["vars"]
        layers.append((np.array(group["0"], dtype=np.float64), np.array(group["1"], dtype=np.float64)))
    return layers


def _logits(x):
    global _LAYERS
    if _LAYERS is None:
        _LAYERS = _load_weights()
    h = np.asarray(x, dtype=np.float64).reshape(-1, 32)
    for i, (W, b) in enumerate(_LAYERS):
        h = h @ W + b
        if i < len(_LAYERS) - 1:
            h = np.maximum(h, 0)
    return h


def label(x):
    """Hard labels only: an int for one input, an array of ints for a batch."""
    global _QUERIES
    x = np.asarray(x)
    single = x.ndim == 1
    _QUERIES += 1 if single else len(x)
    labels = np.argmax(_logits(x), axis=1)
    return int(labels[0]) if single else labels


def query_count():
    return _QUERIES
