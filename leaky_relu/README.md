# hard-label-dnn-extraction in leaky relu setting:
This directory demonstrates the attack on networks that use leaky relus.

## Generating a leaky relu dnn
Run
`python leaky_dnn_generator.py`
to create an idealized dnn with leaky relus.
The model's structure and name can be edited in the script.

## Signature Recovery

## Sign Recovery
To simulate sign recovery, run:
`python sign_recovery_leaky.py --model {model_name} --j {num_threads}`

The `model_name` must be the name of a model in the `data` directory (default is `unitary_leaky_8_8x4_4_float64`),
and `j` is the number of parallel threads (default 1).

The attack assumes that the signatures have already been recovered, but is fully blackbox other than that.
It works by finding random points at a decision boundary, which it computes on the fly.

