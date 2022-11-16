import argparse
import sys
import math
sys.path.append('_model')
from environment import *

### Parsing arguments
parser = argparse.ArgumentParser()

parser.add_argument('--visualize', help='whether to plot the swarm or not, default is 0', required=False, type=int, default=0)
parser.add_argument('--N', help='number of fish', required=True, type=int)
parser.add_argument('--NT', help='number of timesteps to simulate', required=True, type=int)
parser.add_argument('--NN', help='number of nearest neighbours used for state', required=True, type=int)
parser.add_argument('--dim', help='Dimensions.', required=True, type=int, default=2)
parser.add_argument('--run', help='Run tag.', required=False, type=int, default=0)

args = vars(parser.parse_args())

### check arguments
numIndividuals          = int(args["N"])
numTimesteps            = int(args["NT"])
numNearestNeighbours    = int(args["NN"])
run                     = int(args["run"])

assert (numIndividuals > 0) 
assert (numTimesteps > 0) 
assert (numNearestNeighbours > 0) 
assert (numIndividuals > numNearestNeighbours)

### Define Korali Problem
import korali
k = korali.Engine()
e = korali.Experiment()

### Define results folder and loading previous results, if any
resultFolder = f'_result_vracer_{run}/'

found = e.loadState(resultFolder + '/latest')
if found == True:
	print(f"[run-vracer] Old run {resultFolder} exists, abort..")
    sys.exit()

### Define Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lambda x : environment( args, x )
e["Problem"]["Agents Per Environment"] = numIndividuals

### Define Agent Configuration 
e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

# States (distance and angle to nearest neighbours)
for i in range(numNearestNeighbours):
  e["Variables"][i]["Name"] = "Distance " + str(i)
  e["Variables"][i]["Type"] = "State"
  e["Variables"][i+numNearestNeighbours]["Name"] = "Angle " + str(i)
  e["Variables"][i+numNearestNeighbours]["Type"] = "State"

# Direction update left/right
e["Variables"][2*numNearestNeighbours]["Name"] = "Phi"
e["Variables"][2*numNearestNeighbours]["Type"] = "Action"
e["Variables"][2*numNearestNeighbours]["Lower Bound"] = 0
e["Variables"][2*numNearestNeighbours]["Upper Bound"] = -np.pi
e["Variables"][2*numNearestNeighbours]["Initial Exploration Noise"] = np.pi

# Direction update up/down
e["Variables"][2*numNearestNeighbours+1]["Name"] = "Theta"
e["Variables"][2*numNearestNeighbours+1]["Type"] = "Action"
e["Variables"][2*numNearestNeighbours+1]["Lower Bound"] = 0
e["Variables"][2*numNearestNeighbours+1]["Upper Bound"] = -np.pi/2
e["Variables"][2*numNearestNeighbours+1]["Initial Exploration Noise"] = np.pi/2

### Set Experience Replay, REFER and policy settings
e["Solver"]["Experience Replay"]["Start Size"] = 131072
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1

e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
  
### Configure the neural network and its hidden layers
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = False
e["Solver"]["L2 Regularization"]["Importance"] = 0.0

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Set file output configuration
e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e6
e["Solver"]["Experience Replay"]["Serialize"] = False
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 20
e["File Output"]["Path"] = resultFolder


### Run Experiment
k.run(e)
