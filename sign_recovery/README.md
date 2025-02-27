This subdirectory of the code demonstrates the ability of an adversary to recover the
sign of every neuron in the network, assuming that the signatures of the target layer
have already been collected and that the signatures and signs of the previous layers are
already solved.

CAUTION: As we say in the paper, the purpose of this code is to prove that the ideas described
in the papere are effective (i.e., that patch distances are on average larger when the target
neuron is off), not to be an actual attack. As such, the code uses some whitebox functions to
determine the optimal walking direction and to detect when a future neuron has toggled. These
can in principle be implemented in a blackbox setting in polynomial time using the techniques
described in the paper.

## Precomputed dual points

The attack performs a statistical test using a list of dual points for each neuron, which
are assumed to have already been precomputed using the code in the `signature_recovery`
directory. The dual points should be saved to separate files named `layerX_neuronY.npy`.
Each file should contain a numpy array of shape `N x M`, where `N` is a large number
of samples (preferably above 10,000) and `M` is the network's input size.

For the sample neural network, these have already been computed and can be found
in `data/dual_points_cifar10_3x256_64_10_float64/`.

## Analyzing a single neuron

In order to simulate the sign recovery of a single neuron, run

```
python3 --model {model_path} --layerID {layerID} --neuronID {neuronID} --filepath_load_x0 {filepath_load_x0}
```
where `model_path` is the path to the neural network being attacked, `layerID` and `neuronID` identify the target neuron, and `filepath_load_x0`
is the path to the directory containing the precomputed dual points.

The results of the test are saved to `results/{model_path}/{layerID}/{neuronID}`

You can also provide the following options:
    `--nExp {int}`: Max number of experiments to use. The program will exit once it has completed this many samples, but it may exit early
                    has achieved a 95% confidence level on the sign guess. The default value is 400.
    `--nExpMin {int}`: Minimum number of experiments to perform; the program will not exit before this many samples are completed, regardless
                    of the confidence level. The default value is 25.
    `--nToggles {int}`: Determines how many future-layer toggles have happen before the walk stops and distance is measured. Default is 1.
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


## Replicating our results

You can run
```
python batched_sign_recovery
```
to simulate the sign recovery of all neurons in parallel using the settings that were used for the paper. You can edit the "Global Settings" section of this script
to adjust parameters such as the number of threads, neurons to attack, etc.

## Parsing the results
After running the experiments, run
`python create_tables.py`
to parse the results and create a summarized table per layer.