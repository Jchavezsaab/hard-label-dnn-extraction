# ---------------------------------------------------
# Prepare environment
# ---------------------------------------------------
import pathlib
import os
import sys
from sign_recovery_common import parseArguments, parseRange

if __name__ == "__main__":

    args = parseArguments(sys.argv[1:])

    try:
        neurons = parseRange(args.neuron)
        layers = parseRange(args.layer)
    except:
        print("Failed to parse neuron/layer range")
        exit(-1)

    pathlib.Path(f"../data/results/{args.model}").mkdir(parents=True, exist_ok=True)

    for layer in range(1,5):
        print(f"LAYER {layer}")
        print("neuron\texperiments\tvotes+\tvotes-\tconfidence\tfailed?")
        count = 0
        for neuron in range(8):
            try:
                with open(f"../data/results/{args.model}/{layer}_{neuron}.txt") as f:
                    for x in f: line = x.strip() #Gets the last line in the file
                    line = line.split("Experiments ")[1]
                    N, line = line.split("/", 1)
                    M, line = line.split(",", 1)
                    line = line.split("votes+ ")[1]
                    vp, line = line.split(",",1)
                    line = line.split("votes- ")[1]
                    vm, line = line.split(",",1)
                    conf = line.split(" ")[-1]
                    if int(vp) > int(vm): res = ""
                    else: res = "FAIL"
                    print(f"{neuron}\t{N}/{M}\t\t{vp}\t{vm}\t{conf}\t\t{res}")
                    count += 1
            except Exception as e:
                print(e)
        print()
