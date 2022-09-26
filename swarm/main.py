import argparse
import sys
sys.path.append('_model')
from swarm import *
from plotter3D import *
import math
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--visualize', help='whether to plot the swarm or not', required=True, type=int)
    parser.add_argument('--numIndividuals', help='number of fish', required=True, type=int)
    parser.add_argument('--numTimesteps', help='number of timesteps to simulate', required=True, type=int)
    parser.add_argument('--numNearestNeighbours', help='number of nearest neighbours used for state/reward', required=True, type=int)
    parser.add_argument('--numdimensions', help='number of dimensions of the simulation', required=True, type=int)
    parser.add_argument('--centered', help='if plotting should the camera be centered or not', required=True, type=int)
    parser.add_argument('--movementType', help='Type of movement, 0 is hardcoded, 1 is random, 2 is according to the related papers', required=True, type=int)

    args = vars(parser.parse_args())

    numIndividuals       = args["numIndividuals"]
    numTimeSteps         = args["numTimesteps"]
    numNearestNeighbours = args["numNearestNeighbours"]
    numdimensions        = args["numdimensions"]
    followcenter         = args["centered"]
    movementType         = args["movementType"]

    assert numIndividuals > numNearestNeighbours, print("numIndividuals must be bigger than numNearestNeighbours")
    
    sim  = swarm( numIndividuals, numNearestNeighbours,  numdimensions, movementType)
    step = 0
    done = False
    action = np.zeros(shape=(sim.dim), dtype=float)
    vec = np.random.normal(0.,1.,sim.dim)
    mag = np.linalg.norm(vec)
    action = vec/mag
 #   action[0]=1.
    while (step < numTimeSteps) and not done:
        print("timestep {}/{}".format(step+1, numTimeSteps))
        # if enable, plot current configuration
        if args["visualize"]:
            Path("./_figures").mkdir(parents=True, exist_ok=True)
            if(sim.dim == 3):
                plotSwarm3D( sim, step, followcenter)
            else:
                plotSwarm2D( sim, step, followcenter)

        # compute pair-wise distances and view-angles
        done = sim.preComputeStates()

        if(movementType == 2):
            sim.move_calc()

        # update swimming directions
        for i in np.arange(sim.N):
            print("agent {}/{}".format(i+1, sim.N))
            # for Newton policy state is the directions to the nearest neighbours
            state  = sim.getState(i)
            # print("state:", state)
            # set action
            # action = sim.fishes[i].newtonPolicy( state )
            # print("action:", action)
            if(movementType == 0):
                if math.isclose( np.linalg.norm(action),  1.0 ):
                    sim.fishes[i].wishedDirection = action
            elif(movementType == 1):
                    vec = np.random.normal(0.,1.,sim.dim)
                    mag = np.linalg.norm(vec)
                    sim.fishes[i].wishedDirection = vec/mag
            elif(movementType == 2):
                break
            else:
                raise Exception("Unknown movement type please choose 0, 1 or 2")

            # get reward (Careful: assumes sim.state(i) was called before)
            reward = sim.getReward( i )
            print("reward:", reward)
            # rotation in wished direction
            sim.fishes[i].updateDirection()
            # update positions
            sim.fishes[i].updateLocation()

        step += 1
