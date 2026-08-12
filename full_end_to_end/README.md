# Full end-to-end extraction of a black mox mdoel

This directory has a copy of the code that's designed to work with
itself and run end-to-end. It's somewhat less easy to work with as
research code, but it shows the functionality as described in the
paper.

This attack is fully end-to-end meaning that it does not touch the
real weights at any part along the attack process. It follows the
code from the paper directly:
- a large collection of dual points are identified in a fully black-box
  manner by walking the decision boundary
- these dual points are clustered to recover first-layer signatures
- then we recover the sign in a polynomial time manner
- (we then refine the first-layer signatures with a CJM'20 style metric)
- and then we repeat for later layers
- the last layer is extracted separately as described in the paper


python3 run.py [num workers] runs each of the steps in turn and writes
the intermediate outputs to out/. This calls validate.py, which looks
at the true weights and compares the extracted network with the real
one. The pipline should take a few minutes on a 32-core machine and
needs about 40 million queries.

The files in this directory are copies of the attack code in
../signature_recovery and ../sign_recovery cleaned up slightly so that
everything is self-contained and easy to see that there is no cheating
being done, with each layer building on the prior layer's stolen weights.
