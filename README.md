# hard-label-dnn-extraction
Supplementary code for the paper "Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting" (published originaly at EUROCRYPT 2024, with an extended version in Journal of Cryptology).

The code is split into two sections:
1. Signature recovery
2. Sign recovery

see the README file in each directory for a detailed explanation.

The `data/` directory contains the following .keras files for the neural networks that we used to illustrate the attack:
- `cifar10_3x256_64_10_float64.keras` This is a 'real' network used for whitebox experiments, which
was trained on the CIFAR-10 dataset, achieving 0.52 accuracy.
- `unitary_32_32x3_10_float64.keras` This is an artificial network used for blackbox experiments, with all weights set to random unitary matrices and biases chosen so that neurons are active half of the time.
- `unitary_leaky_32_32x3_10_float64.keras` Similar to the previous one, but uses leaky relus with a 0.1 leaky coefficient.

It also contains scripts for generating new networks and dual points (see `data/README.md`)

