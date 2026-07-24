This subdirectory of the code demonstrates the ability of an adversary to recover the sign of every neuron in the network, assuming that the signatures of the target layer have already been collected and that the signatures and signs of the previous layers are already solved.

# Requirements
The code in this directory requires the following pip packages:
- numpy
- pandas
- matplotlib
- tabulate
- tensorflow

# Precomputed dual points

Our sign recovery attacks perform a statistical test using a list of dual points for each neuron, which are assumed to have already been precomputed using the code in the `signature_recovery` directory.

The dual points should be saved to separate files in `data/dual_points_{model_name}/layer{layerId}_neuron{nueronId}.npy`.
Each file should contain a numpy array of shape `N x 3 x M`, where `N` is a large number of samples (preferably above 10,000)
and `M` is the network's input size. Each entry should be a triplet of points: one just before the relu boundary, one as close
as posible to the boundary, and one just past the boundary, respectively, hence the `3`.

If you only want to test sign recovery without running signature recovery, see `generators/README.md` for a shortcut to generate or download these files.

# Whitebox analysis

In order to run the whitebox attack on a relu-based dnn, run

```
python sign_recovery_whitebox.py --model {model_name} --layerID {layerID} --neuronID {neuronID} --j {num threads}
```
where `model_name` is the name of the neural network being attacked (should be saved to `data/{model_name}.keras`, default is `cifar10_3x256_64_10_float64`),
`layerID` and `neuronID` identify the target neuron. Both `--layer` and `--neuron` admit a single number or comma-separated
numbers or ranges (e.g. `--neuron 1,2,5-7`). The `--j` flags allows one to launch the recovery for different neurons with concurrent threads (default is 1).


CAUTION: As we say in the paper, the purpose of this script is to prove that the ideas described
in the papere are effective (i.e., that patch distances are on average larger when the target
neuron is off), not to be an actual attack. As such, it recreates the statistics that would be
collected from the real attack, but makes use of whitebox information about the model for efficiency
purposes at two specific steps:
- Computing the normal vector to the decision plane
- Measuring the distance along the decision plane until another neuron toggles

The results of the test are saved to `results/whitebox/{model_path}/{layerID}/{neuronID}`

You can also provide the following options:
- `--nExpMax {int}`: Max number of experiments to use. The program will exit once it has completed this many samples, but it may exit early if it
                    has achieved a 95% confidence level on the sign guess. The default value is 400.
- `--nExpMin {int}`: Minimum number of experiments to perform; the program will not exit before this many samples are completed, regardless
                    of the confidence level. The default value is 25.
- `--nToggles {int}`: Determines how many future-layer toggles have to happen before the walk stops and distance is measured. Default is 1.
- `--handlePrevLayerToggles <True|False>`: If set to True (default), the attack will recompute the optimal walking direction whenever a previous-layer
                                            neuron has toggled. Otherwise, it will discard any experiment where a previous-layer neuron was toggled.
- `--choose_dx {perfect_control_along_decision_boundary | along_decision_boundary}`:
                    Determines the walking direction for the experiments. `along_decision_boundary` (default) uses the normal vector of the critical plane projected
                    onto the decision plane (which is what we would be able to compute in a blackbox setting), whereas `perfect_control_along_decision_boundary`
                    starts with a walking direction that only changes the target neuron while keeping all the non-target neurons of the target layer fixed,
                    and then projects it onto the decision plane (this technique only applies for hidden layers 1 and 4 of our network, where the dimension of control is total).
- `--analyzeWiggleSensitivity <True | False>`: If set to True, records data on the rate of change of the target layer's vector under the chosen walking direction, for
                    ON vs OFF sides. Default is False.
- `--analyzeSpeed <True | False>`: If set to True, records data on the rate of change of each future-layer neuron under the chosen walking direction, for
                    the ON vs OFF sides. Default is False.
- `--nDebug {nDebug} <True | False>` If set to True, skips logging and several consistency checks in favor of performance. Default is True.


# Blackbox analysis

To simulate the blackbox sign recovery on a relu dnn, run

```
python sign_recovery_blackbox.py --model {model_path} --layer {layerID} --neuron {neuronID} --j {num threads}
```
This performs the recovery of signs using only blackbox functionality, but does assume that perfect signatures are provided beforehand. It is only feasible for
relatively small networks (the default for `--model` is `unitary_32_32x3_10_float64`). Both `--layer` and `--neuron` admit a single number or comma-separated
numbers or ranges (e.g. `--neuron 1,2,5-7`). The `--j` flags allows one to launch the recovery for different neurons with concurrent threads (default is 1).

# Leaky Relus

You can also simulate a blackbox sign recovery on a leaky-relu dnn by running 
```
python sign_recovery_leaky.py --model {model_name} --j {num_threads}
```
The `model_name` must be the name of a model in the `data` directory (default is `unitary_leaky_32_32x3_10_float64`),
and `j` is the number of parallel threads (default 1). This attack does not require any precomputed dual points (they are computed on the fly) and is much more efficient, so it always runs for all layers and all neurons.

The attack assumes that the signatures have already been recovered, but is fully blackbox other than that.

# Replicating our results

## Whitebox
For the whitebox experiments, you can run:
```
python sign_recovery_whitebox.py --model cifar10_3x256_64_10_float64 --layer 1 --neuron 0-255 --nExpMax 10000 --nExpMin 100 --choose_dx perfect_control_along_decision_boundary --analyzeWiggleSensitivity True --analyzeSpeed True --j 10
```
```
python sign_recovery_whitebox.py --model cifar10_3x256_64_10_float64 --layer 2,3 --neuron 0-255 --nExpMax 10000 --nExpMin 1000 --choose_dx along_decision_boundary --analyzeWiggleSensitivity True --analyzeSpeed True --j 10
```
```
python sign_recovery_whitebox.py --model cifar10_3x256_64_10_float64 --layer 4 --neuron 0-63 --nExpMax 10000 --nExpMin 100 --choose_dx perfect_control_along_decision_boundary --analyzeWiggleSensitivity True --analyzeSpeed True --j 10
```
to replicate our results of the sign recovery of all neurons in parallel (adjust `--j 10` to your liking) using the settings that were used for the paper.

## Blackbox
For the blackbox experiments, you can run
```
python sign_recovery_blackbox.py --model unitary_32_32x3_10_float64 --layer 1-3 --neuron 0-7 --j 8
```

## Parsing
After running either the whitebox or blackbox experiments (or both), run
```
python create_tables.py
```
to parse the results and create a summarized table per layer.

## Leaky Relu
For the leaky relu network, run
```
python sign_recovery_leaky.py --model unitary_leaky_32_32x3_10_float64 --j 8
```
This will attack all neurons at once and print a summary of the results.