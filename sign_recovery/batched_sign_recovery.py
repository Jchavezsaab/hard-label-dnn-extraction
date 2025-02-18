# Imports
import sys
import os
from os.path import dirname, abspath
SCRIPT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# perform imports from parent directory
import sign_recovery

# ========== Global Settings ========== #
model_name               = "cifar10_3x256_64_10_float64" # Name of the model to be analyzed (for labeling of output files)
model_path               = f"../data/{model_name}.keras" # Path to the .keras file containing the model
duals_path                 = f"../data/critical_points_{model_name}" # Path to precomputed dual points
LAYERIDS                 = 1, # layer IDs to analyze
NEURONIDS                = range(256) # neuron IDs to analyze
analyzeWiggleSensitivity  = 'True' # Record the sensitivity ti the wiggle at the target layer
analyzeSpeed              = 'True' # Record the rate of change of future layer neurons
nDebug                    = 'True' # Set to True to skip logfile
nThreads                  = 10 # Number of threads to be used for analyzing multiple neurons in parallel
# ==================================== #

# Set up multiprocessing
if nThreads > 1:
    import multiprocessing 
    def error_handler(e): 
        print(dir(e), "\n")
        print("-->{}<--".format(e.__cause__))
    pool = multiprocessing.Pool(nThreads)

for layerID in LAYERIDS:
    # Parameter adjustments for each layer
    if layerID == 1: 
        nExpMin   = 100
        nExp      = 10_000
        choose_dx = 'perfect_control_along_decision_boundary'
        NEURONIDS = [x for x in NEURONIDS if x < 256]
    elif (layerID == 2) or (layerID==3): 
        nExpMin   = 1000
        nExp      = 10_000
        choose_dx = 'along_decision_boundary'
        NEURONIDS = [x for x in NEURONIDS if x < 256]
    elif layerID == 4: 
        nExpMin   = 1
        nExp      = 100
        choose_dx = 'perfect_control_along_decision_boundary'
        NEURONIDS = [x for x in NEURONIDS if x < 64]
    else: 
        raise Exception(f"Layer ID  is {layerID}, but has to be 1, 2, 3, or 4.")

    # Run script for each neuron
    for neuronID in NEURONIDS:
        current_filename          = os.path.basename(__file__)
        runID_base                = current_filename.replace('.py', '')
        runID = f'{runID_base}_neuron{neuronID}'
        # =========== Load dual points ==============
        filepath_load_x0 = f"{duals_path}/layer{layerID}_neuron{neuronID}.npy"
        # =========== RUN ==============
        args = f"""--model {model_path} --layerID {layerID} --neuronID {neuronID} --nExp {nExp} --analyzeWiggleSensitivity {analyzeWiggleSensitivity} --analyzeSpeed {analyzeSpeed} --handlePrevLayerToggles True --nToggles 1 --nDebug {nDebug} --filepath_load_x0 {filepath_load_x0} --nExpMin {nExpMin} --choose_dx {choose_dx}"""
        if nThreads > 1: pool.apply_async(func=sign_recovery.main, args=[args.split(' ')], error_callback=error_handler)
        else: 
            args = args.split(' ')
            sign_recovery.main(args)
if nThreads > 1: 
    pool.close()
    pool.join()
