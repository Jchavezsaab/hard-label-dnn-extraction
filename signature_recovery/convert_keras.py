#!/usr/bin/env python3
"""Convert the keras unitary_32_32x3_10_float64 model to a torch state_dict
(models/unitary32.pth) for signature_recovery's SIGMODEL=unitary32 mode.

Keras Dense stores kernels as (in, out); torch nn.Linear wants (out, in),
so every kernel is transposed. Verifies logits agree to ~1e-12 on 1000
random inputs before saving.
"""
import os
import sys
import numpy as np

MODEL = sys.argv[1] if len(sys.argv) > 1 else "../data/unitary_32_32x3_10_float64.keras"


def main():
    import tensorflow as tf
    import torch

    km = tf.keras.models.load_model(MODEL)
    dense = [l for l in km.layers if len(l.get_weights()) == 2]
    assert len(dense) == 4, f"expected 4 Dense layers, got {len(dense)}"

    sd = {}
    for i, l in enumerate(dense):
        W, b = l.get_weights()
        sd[f"fc{i+1}.weight"] = torch.tensor(W.T, dtype=torch.float64)
        sd[f"fc{i+1}.bias"] = torch.tensor(b, dtype=torch.float64)

    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2, self.fc3, self.fc4 = (
                nn.Linear(32, 32), nn.Linear(32, 32),
                nn.Linear(32, 32), nn.Linear(32, 10))
            self.double()
        def forward(self, x):
            r = nn.functional.relu
            x = r(self.fc1(x)); x = r(self.fc2(x)); x = r(self.fc3(x))
            return self.fc4(x)

    net = Net()
    net.load_state_dict(sd)

    x = np.random.RandomState(0).normal(size=(1000, 32))
    ref = km(x.astype(np.float64)).numpy()
    got = net(torch.tensor(x).double()).detach().numpy()
    err = np.abs(ref - got).max()
    assert err < 1e-10, f"logit mismatch {err}"
    os.makedirs("models", exist_ok=True)
    torch.save(sd, "models/unitary32.pth")
    print(f"saved models/unitary32.pth; max logit err vs keras = {err:.2e}")


if __name__ == "__main__":
    main()
