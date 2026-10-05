# The black box.  The attack may use label(), query_count() and architecture() and nothing else from this file.
#
# The model is the .keras file named by the environment variable ORACLE_MODEL (set_model() sets it, so that worker
# processes inherit it), by default data/unitary_32_32x3_10_float64.keras.
import io
import json
import os
import zipfile

import numpy as np

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "unitary_32_32x3_10_float64.keras")

_QUERIES = 0
_NET = None


def model_path():
    return os.path.abspath(os.environ.get("ORACLE_MODEL", DEFAULT_MODEL))


def set_model(path):
    """Point the oracle (and every process started after this call) at another .keras file."""
    global _NET
    os.environ["ORACLE_MODEL"] = os.path.abspath(path)
    _NET = None


def _load_weights():
    """[(W (in, out), b (out,)) per Dense layer], read straight out of the .keras file (a zip holding an hdf5 file)."""
    import h5py
    with zipfile.ZipFile(model_path()) as archive:
        weights = h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r")
    names = sorted((name for name in weights["layers"] if name.startswith("dense")),
                   key=lambda name: int(name.split("_")[1]) if "_" in name else 0)
    layers = []
    for name in names:
        group = weights["layers"][name]["vars"]
        layers.append((np.array(group["0"], dtype=np.float64), np.array(group["1"], dtype=np.float64)))
    return layers


def _load_activations():
    """The activation after every Dense layer, from the model's config: ("relu", 0), ("leaky_relu", slope) or ("linear", 0).

    The activation of the last Dense layer is left out: it is monotone (linear / softmax) and cannot change an argmax."""
    with zipfile.ZipFile(model_path()) as archive:
        config = json.loads(archive.read("config.json"))
    activations = []
    for layer in config["config"]["layers"]:
        kind, cfg = layer["class_name"], layer["config"]
        if kind == "Dense":
            activations.append(("relu", 0.0) if cfg.get("activation") == "relu" else ("linear", 0.0))
        elif kind == "ReLU":
            activations[-1] = ("relu", 0.0)
        elif kind == "LeakyReLU":
            activations[-1] = ("leaky_relu", float(cfg.get("negative_slope", cfg.get("alpha", 0.3))))
        elif kind == "Activation" and cfg.get("activation") == "relu":
            activations[-1] = ("relu", 0.0)
    return activations[:-1]


def _net():
    global _NET
    if _NET is None:
        layers = _load_weights()
        activations = _load_activations()
        assert len(activations) == len(layers) - 1, "%s: %d Dense layers but %d hidden activations in the config" % (
            model_path(), len(layers), len(activations))
        _NET = layers, activations
    return _NET


def architecture():
    """The public shape of the target (no weights): input dimension, hidden widths, number of classes, hidden activations."""
    layers, activations = _net()
    return dict(
        input_dim=layers[0][0].shape[0],
        widths=[W.shape[1] for W, _ in layers[:-1]],
        classes=layers[-1][0].shape[1],
        activations=[kind for kind, _ in activations],
    )


def _logits(x):
    layers, activations = _net()
    h = np.asarray(x, dtype=np.float64).reshape(-1, layers[0][0].shape[0])
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(activations):
            kind, slope = activations[i]
            if kind == "relu":
                h = np.maximum(h, 0)
            elif kind == "leaky_relu":
                h = np.where(h > 0, h, slope * h)
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
