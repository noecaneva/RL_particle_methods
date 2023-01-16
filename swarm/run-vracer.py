import argparse
import sys
import math
sys.path.append('_model')
from environment import *
import numpy as np

### Parsing arguments
parser = argparse.ArgumentParser()

parser.add_argument('--visualize', help='whether to plot the swarm or not, default is 0', required=False, type=int, default=0)
parser.add_argument('--N', help='Swarm size.', required=True, type=int)
parser.add_argument('--NT', help='Number of steps', required=True, type=int)
parser.add_argument('--NN', help='Number of nearest neighbours', required=True, type=int)
parser.add_argument('--NL', help='Number of nodes in hidden layer', required=True, type=int, default=32)
parser.add_argument('--reward', help='Reward type (local / global)', required=True, type=str, default="global")
parser.add_argument('--exp', help='Number of experiences.', required=True, type=int, default=1000000)
parser.add_argument('--dim', help='Dimensions.', required=True, type=int, default=3)
parser.add_argument('--run', help='Run tag.', required=False, type=int, default=0)

args = vars(parser.parse_args())

### check arguments
numIndividuals          = int(args["N"])
numTimesteps            = int(args["NT"])
numNearestNeighbours    = int(args["NN"])
numNodesLayer           = int(args["NL"])
exp                     = int(args["exp"])
dim                     = int(args["dim"])
run                     = int(args["run"])
visualize               = int(args["visualize"])

assert (numIndividuals > 0) 
assert (numTimesteps > 0) 
assert (exp > 0) 
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

# Define max Angle of rotation during a timestep
maxAngle=swarm.maxAngle
nrVectorStates=swarm.nrVectorStates

### Define Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lambda x : environment( args, x )
e["Problem"]["Agents Per Environment"] = numIndividuals

### Define Agent Configuration 
e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Testing" if visualize else "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

numStates = nrVectorStates*numNearestNeighbours if dim == 2 else 3*numNearestNeighbours
# States (distance and angle to nearest neighbours)
for i in range(numStates):
  e["Variables"][i]["Name"] = "State " + str(i)
  e["Variables"][i]["Type"] = "State"

#for i in range(numNearestNeighbours):
#  e["Variables"][i+numNearestNeighbours]["Name"] = "Phi " + str(i)
#  e["Variables"][i+numNearestNeighbours]["Type"] = "State"

#for i in range(numNearestNeighbours):
#  e["Variables"][i+anumNearestNeighbours]["Name"] = "Phi " + str(i)
#  e["Variables"][i+numNearestNeighbours]["Type"] = "State"


# TODO: in /_model/swarm.py angles towards nearest neighbours need to be
# calculated correctly
if dim == 3:
    for i in range(numNearestNeighbours):
      e["Variables"][i+2*numNearestNeighbours]["Name"] = "Theta " + str(i)
      e["Variables"][i+2*numNearestNeighbours]["Type"] = "State"

# Direction update left/right
e["Variables"][numStates]["Name"] = "Phi"
e["Variables"][numStates]["Type"] = "Action"
e["Variables"][numStates]["Lower Bound"] = -maxAngle
e["Variables"][numStates]["Upper Bound"] = +maxAngle
e["Variables"][numStates]["Initial Exploration Noise"] = maxAngle/2.

# Direction update up/down
if dim == 3:
    e["Variables"][numStates+1]["Name"] = "Theta"
    e["Variables"][numStates+1]["Type"] = "Action"
    e["Variables"][numStates+1]["Lower Bound"] = -np.pi/2
    e["Variables"][numStates+1]["Upper Bound"] = +np.pi/2
    e["Variables"][numStates+1]["Initial Exploration Noise"] = np.pi/4

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
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = numNodesLayer

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/SoftReLU"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = numNodesLayer

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/SoftReLU"

### Set file output configuration
e["Solver"]["Termination Criteria"]["Max Experiences"] = exp
e["Solver"]["Experience Replay"]["Serialize"] = False
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 25
e["File Output"]["Path"] = resultFolder
e["Solver"]["Testing"]["Sample Ids"] = [ 0 ] 

### Run Experiment
k.run(e)
