This subdirectory of the code demonstrates the ability of an adversary to recover the
sign of every neuron in the network, assuming that the signatures of the target layer
have already been collected and that the signatures and signs of the previous layers are
already solved.

# Precomputed dual points

The attack performs a statistical test using a list of dual points for each neuron, which
are assumed to have already been precomputed using the code in the `signature_recovery`
directory. The dual points should be saved to separate files named `layerX_neuronY.npy`
in `data/dual_points_{model_name}`.
Each file should contain a numpy array of shape `N x M`, where `N` is a large number
of samples (preferably above 10,000) and `M` is the network's input size.

For the sample neural network, these can be downloaded from:
https://drive.google.com/file/d/1mFfKlLgE0ZnGPAYN8tPRtb2iYpP5QgfY/view?usp=sharing
The zip file must be extracted to `data/dual_points_cifar10_3x256_64_10_float64/`.

# Whitebox analysis of a single neuron

In order to simulate the sign recovery of a single neuron, run

```
python3 sign_recovery_whitebox.py --model {model_path} --layerID {layerID} --neuronID {neuronID} --filepath_load_x0 {filepath_load_x0}
```
where `model_path` is the path to the neural network being attacked, `layerID` and `neuronID` identify the target neuron, and `filepath_load_x0`
is the path to the directory containing the precomputed dual points.


CAUTION: As we say in the paper, the purpose of this script is to prove that the ideas described
in the papere are effective (i.e., that patch distances are on average larger when the target
neuron is off), not to be an actual attack. As such, it recreates the statistics that would be
collected from the real attack, but makes use of whitebox information about the model for efficiency
purposes at two specific steps:
- Computing the normal vector to the decision plane
- Measuring the distance along the decision plane until another neuron toggles

The results of the test are saved to `results/whitebox/{model_path}/{layerID}/{neuronID}`

You can also provide the following options:
    `--nExp {int}`: Max number of experiments to use. The program will exit once it has completed this many samples, but it may exit early if it
                    has achieved a 95% confidence level on the sign guess. The default value is 400.
    `--nExpMin {int}`: Minimum number of experiments to perform; the program will not exit before this many samples are completed, regardless
                    of the confidence level. The default value is 25.
    `--nToggles {int}`: Determines how many future-layer toggles have to happen before the walk stops and distance is measured. Default is 1.
    `--handlePrevLayerToggles <True|False>`: If set to True (default), the attack will recompute the optimal walking direction whenever a previous-layer
                                            neuron has toggled. Otherwise, it will discard any experiment where a previous-layer neuron was toggled.
    `--choose_dx {perfect_control_along_decision_boundary | along_decision_boundary}`:
                    Determines the walking direction for the experiments. `along_decision_boundary` (default) uses the normal vector of the critical plane projected
                    onto the decision plane (which is what we would be able to compute in a blackbox setting), whereas `perfect_control_along_decision_boundary`
                    starts with a walking direction that only changes the target neuron while keeping all the non-target neurons of the target layer fixed,
                    and then projects it onto the decision plane (this technique only applies for hidden layers 1 and 4 of our network, where the dimension of control is total).
    `--analyzeWiggleSensitivity <True | False>`: If set to True, records data on the rate of change of the target layer's vector under the chosen walking direction, for
                    ON vs OFF sides. Default is False.
    `--analyzeSpeed <True | False>`: If set to True, records data on the rate of change of each future-layer neuron under the chosen walking direction, for
                    the ON vs OFF sides. Default is False.
    `--nDebug {nDebug} <True | False>` If set to True, skips logging and several consistency checks in favor of performance. Default is False.


# Blackbox analysis

To simulate the blackbox sign recovery, run

```
python3 sign_recovery_blackbox.py --model {model_path} --layer {layerID} --neuron {neuronID} --j {num threads}
```
This performs the recovery of signs using only blackbox functionality, but does assume that perfect signatures are provided beforehand. It is only feasible for
relatively small networks (the default for `--model` is `unitary_32_8x4_4_float64`). Both `--layer` and `--neuron` admit a single number or comma-separated
numbers or ranges (e.g. `--neuron 1,2,5-7`). The `--j` flags allows one to launch the recovery for different neurons with concurrent threads (default is 1).

## Replicating our results

For the whitebox experiments, you can run
```
python batched_sign_recovery_whitebox.py
```
to replicate our results of the sign recovery of all neurons in parallel using the settings that were used for the paper.
You can edit the "Global Settings" section of this script to adjust parameters such as the number of threads, neurons to attack, etc.

For the blackbox experiments, you can run
```
python3 sign_recovery_blackbox.py --layer 1-3 --neuron 0-7 --j 8
```

After running either the whitebox or blackbox experiments (or both), run
`python create_tables.py`
to parse the results and create a summarized table per layer.