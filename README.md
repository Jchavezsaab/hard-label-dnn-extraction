# hard-label-dnn-extraction
Supplementary code for the EUROCRYPT 2024 paper "Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting"

The code is split into two phases:
1. Signature recovery
2. Sign recovery
see the README file in each directory for a detailed explanation.

The `data/` directory contains a .keras file for the neural network that we used to illustrate the attack. This is a 'real' network which
was trained on the CIFAR-10 dataset, achieving 0.52 accuracy. It also contains arrays of precomputed dual points for this network, which
were generated using the code in `signature_recovery`.

