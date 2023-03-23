#!/usr/bin/env python3
import sys
sys.path.append('./_model')

import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--resdir', help='Directory of IRL results.', required=True)
parser.add_argument('--tfile', help='Trajectory file.', required=True)

args = parser.parse_args()
print(args)

trajectory = np.load(args.tfile)
locations = trajectory["locationHistory"]
directions = trajectory["directionHisory"]
print(locations.shape)
print(directions.shape)

sys.exit()

resfile = f'{args.resdir}/latest'
with open(resfile, 'r') as infile:
    results = json.load(infile)
    actionDim = results["Problem"]["Action Vector Size"]
    featureDim = results["Problem"]["Feature Vector Size"]
    stateDim = results["Problem"]["State Vector Size"]
    rewardHp = results["Solver"]["Training"]["Best Reward Params"]["Hyperparameters"]
    policyHp = results["Solver"]["Training"]["Best Policies"]["Hyperparameters"]

#print(rewardHp)

import korali
k = korali.Engine()
e = korali.Experiment()
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = 1
e["Problem"]["Testing Batch Size"] = 4

e["Problem"]["Input"]["Data"] = [[np.random.uniform(size=featureDim).tolist()], [np.random.uniform(size=featureDim).tolist()],[np.random.uniform(size=featureDim).tolist()], [np.random.uniform(size=featureDim).tolist()]]
e["Problem"]["Input"]["Size"] = featureDim
e["Problem"]["Solution"]["Size"] = 1

### Using a neural network solver (deep learning) for training

e["Solver"]["Type"] = "DeepSupervisor"
e["Solver"]["Mode"] = "Testing"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Learning Rate"] = 1e-4

#e["Solver"]["Current Generation"] = 0
#e["Solver"]["Optimizer"]["Current Value"] = rewardHp
e["Solver"]["Hyperparameters"] = rewardHp

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 8

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 8

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"

### Configuring output

e["Random Seed"] = 0xC0FFEE
e["File Output"]["Enabled"] = True
e["File Output"]["Use Multiple Files"] = True
e["Console Output"]["Frequency"] = 1
e["File Output"]["Path"] = '_korali_result_reward_evaluation'

e["Solver"]["Termination Criteria"]["Max Generations"] = 100
k.run(e)

testInferredSet = [ x for x in e["Solver"]["Evaluation"] ]
print(testInferredSet)

