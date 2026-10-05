"""
Trains a dense ReLU network intrusion detector on KDD Cup 99 and saves it as a float64 .keras file the end-to-end
attack can target (python full_end_to_end/run.py --model ../data/kddcup99_32_64x4_4_float64.keras).

Data: KDD Cup 99 through scikit-learn (pip install scikit-learn); sklearn.datasets.fetch_kddcup99 downloads it on first
use and caches it (by default the 10% subset, 494,021 connections).  Each connection is labelled normal or with one of
22 attacks, grouped into the usual families DoS / Probe / R2L / U2R; U2R (52 connections) is too rare to learn and is
dropped.  The three categorical features (protocol_type, service, flag) are one-hot encoded, which with the 38 numeric
ones gives 118 candidate inputs: enough for an input layer at least as wide as the hidden layers (the attack needs a
non-expanding network, every width <= the one below it; the script refuses anything else).

The model sees preprocessed features: x -> sign(x) log(1 + |x|) (byte counts span many orders of magnitude), then
standardized, then every usable one of them (or the FEATURES most class-discriminative ones).  That preprocessing is NOT part of the .keras file,
which holds only Input -> [Dense, ReLU] x layers -> Dense (logits); it is written next to it as <model>.preprocess.json.
The network is saved exactly as trained (no pruning, no folding).

Usage: ids_dnn_generator.py [--features 0] [--hidden 64,64,64,64] [--epochs 30] [--per-class 200000] [--full]
                            [--data-home DIR] [--out PATH]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from sklearn.datasets import fetch_kddcup99  # noqa: E402
from tensorflow.keras import layers, models, optimizers, callbacks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

FAMILIES = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
}
CLASSES = ["Normal", "DoS", "Probe", "R2L"]
CATEGORICAL = ["protocol_type", "service", "flag"]


def load(per_class, rng, full, data_home):
    """All connections of the known families as (features DataFrame, class index array), at most per_class per class."""
    frame = fetch_kddcup99(as_frame=True, percent10=not full, data_home=data_home).frame
    print("KDD Cup 99: %d connections" % len(frame), flush=True)
    labels = frame.pop("labels").map(lambda b: (b.decode() if isinstance(b, bytes) else b).rstrip("."))
    y = labels.map(lambda name: FAMILIES.get(name))
    for c in CATEGORICAL:
        frame[c] = frame[c].map(lambda b: b.decode() if isinstance(b, bytes) else b)
    X = pd.get_dummies(frame, columns=CATEGORICAL, prefix_sep="=", dtype=np.float64).astype(np.float64)
    keep = y.notna().values
    X, y = X[keep], y[keep].map(CLASSES.index).values

    chosen = []
    for k in range(len(CLASSES)):
        idx = np.flatnonzero(y == k)
        chosen.append(rng.choice(idx, per_class, replace=False) if len(idx) > per_class else idx)
    chosen = np.sort(np.concatenate(chosen))
    return X.iloc[chosen].reset_index(drop=True), y[chosen]


def select_features(Z, y, names, n):
    """Drop constant and near-duplicate columns, then keep the n columns with the largest ANOVA F statistic (n = 0: all)."""
    std = Z.std(axis=0)
    cols = np.flatnonzero(std > 1e-12)
    corr = np.abs(np.corrcoef(Z[:, cols], rowvar=False))
    distinct = []
    for i in range(len(cols)):
        if all(corr[i, j] < 0.99 for j in distinct):
            distinct.append(i)
    cols = cols[distinct]

    mean = Z[:, cols].mean(axis=0)
    between = np.zeros(len(cols))
    within = np.zeros(len(cols))
    for k in np.unique(y):
        Zk = Z[y == k][:, cols]
        between += len(Zk) * (Zk.mean(axis=0) - mean) ** 2
        within += ((Zk - Zk.mean(axis=0)) ** 2).sum(axis=0)
    F = between / np.maximum(within, 1e-300)
    assert n <= len(cols), "only %d usable features, asked for %d" % (len(cols), n)
    cols = np.sort(cols[np.argsort(-F)[:n or len(cols)]])
    return cols, [names[c] for c in cols]


def dnn(input_dim, hidden_sizes, num_classes):
    """Input -> [Dense, ReLU] per hidden layer -> Dense logits, as in dnn_generator.py."""
    hidden_layers = []
    for n in hidden_sizes:
        hidden_layers.append(layers.Dense(n))
        hidden_layers.append(layers.ReLU())
    return models.Sequential([layers.Input(shape=(input_dim,))] + hidden_layers + [layers.Dense(num_classes)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--features", type=int, default=0, help="input size (default 0: every usable feature)")
    parser.add_argument("--hidden", default="64,64,64,64", help="hidden widths, comma separated")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--per-class", type=int, default=200000, help="cap on the flows kept per class")
    parser.add_argument("--full", action="store_true", help="the full 4.9M connections instead of the 10% subset")
    parser.add_argument("--data-home", help="where scikit-learn caches the dataset (default ~/scikit_learn_data)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    hidden = [int(n) for n in args.hidden.split(",")]
    assert all(a >= b for a, b in zip(hidden, hidden[1:])), "hidden widths %s expand" % hidden

    rng = np.random.default_rng(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    X, y = load(args.per_class, rng, args.full, args.data_home)
    print("classes: %s" % ", ".join("%s %d" % (c, (y == k).sum()) for k, c in enumerate(CLASSES)), flush=True)

    # preprocessing (outside the model): signed log, standardize, select
    Z = np.sign(X.values) * np.log1p(np.abs(X.values))
    mean, std = Z.mean(axis=0), Z.std(axis=0)
    Z = (Z - mean) / np.where(std > 1e-12, std, 1.0)
    cols, names = select_features(Z, y, list(X.columns), args.features)
    Z = Z[:, cols]
    print("features (%d): %s" % (len(names), ", ".join(names)), flush=True)
    assert len(cols) >= hidden[0], "%d inputs into a first layer of %d: the network would expand" % (len(cols), hidden[0])

    # stratified 80/10/10 split
    train, val, test = [], [], []
    for k in range(len(CLASSES)):
        idx = rng.permutation(np.flatnonzero(y == k))
        a, b = int(0.8 * len(idx)), int(0.9 * len(idx))
        train.append(idx[:a]); val.append(idx[a:b]); test.append(idx[b:])
    train, val, test = (np.concatenate(s) for s in (train, val, test))

    counts = np.bincount(y[train], minlength=len(CLASSES))
    class_weight = {k: len(train) / (len(CLASSES) * counts[k]) for k in range(len(CLASSES))}

    model = dnn(len(cols), hidden, len(CLASSES))
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.summary()
    model.fit(Z[train], y[train], validation_data=(Z[val], y[val]), epochs=args.epochs, batch_size=1024,
              class_weight=class_weight, verbose=2,
              callbacks=[callbacks.EarlyStopping(patience=4, restore_best_weights=True),
                         callbacks.ReduceLROnPlateau(patience=2, factor=0.3)])

    predicted = np.argmax(model.predict(Z[test], batch_size=8192, verbose=0), axis=1)
    print("test accuracy %.4f" % (predicted == y[test]).mean())
    for k, c in enumerate(CLASSES):
        mask = y[test] == k
        print("  %-10s recall %.4f  (%d flows)" % (c, (predicted[mask] == k).mean(), mask.sum()))

    out = args.out or os.path.join(HERE, "..", "data", "kddcup99_%d_%s_%d_float64.keras" % (
        len(cols), "x".join(map(str, hidden)) if len(set(hidden)) > 1 else "%dx%d" % (hidden[0], len(hidden)),
        len(CLASSES)))
    model.save(out)
    json.dump(dict(classes=CLASSES, features=names,
                   transform="one-hot protocol_type / service / flag (column 'name=value'), then "
                             "z = (sign(x) * log1p(|x|) - mean) / std, on the features listed",
                   mean=mean[cols].tolist(), std=std[cols].tolist()),
              open(os.path.splitext(out)[0] + ".preprocess.json", "w"), indent=1)
    print("wrote %s (+ .preprocess.json)" % os.path.abspath(out))


if __name__ == "__main__":
    main()
