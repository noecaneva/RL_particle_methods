import argparse
import sys
sys.path.append('_model')
import json
from swarm import *
from plotter3D import *
import math
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--numIndividuals', help='number of fish', required=False, type=int, default=10)
    parser.add_argument('--numTimesteps', help='number of timesteps to simulate', required=False, type=int, default=500)
    parser.add_argument('--numNearestNeighbours', help='number of nearest neighbours used for state/reward', required=False, type=int, default=3)
    parser.add_argument('--numdimensions', help='number of dimensions of the simulation', required=False, type=int, default=2)
    parser.add_argument('--initialization', help='how the fishes should be initialized. 0 for grid, 1 for on circle or sphere, 2 for within a circle or a sphere', required=False, type=int, default=1)
    parser.add_argument('--psi', help='gives the initial polarization of the fish', required=False, type=float, default=-1.)
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=10)
    parser.add_argument('--numTrajectories', help='number of trajectories to produce', required=False, type=int, default=1)

    args = vars(parser.parse_args())

    numIndividuals       = args["numIndividuals"]
    numTimeSteps         = args["numTimesteps"]
    numNearestNeighbours = args["numNearestNeighbours"]
    numdimensions        = args["numdimensions"]
    initializationType   = args["initialization"]
    psi                  = args["psi"]
    seed                 = args["seed"]
    numTrajectories      = args["numTrajectories"]

    assert numIndividuals > numNearestNeighbours, print("numIndividuals must be bigger than numNearestNeighbours")

    Path("./_trajectories").mkdir(parents=True, exist_ok=True)
    fname = f'_trajectories/observations_simple_{numIndividuals}_{numNearestNeighbours}.json'
    
    observations = {}
    obsstates = []
    obsactions = []
    obsrewards = []
    try:
        f = open(fname)
        observations = json.load(f)
        obsstates = observations["States"]
        obsactions = observations["Actions"]
        obsrewards = observations["Rewards"]
        print(f'{len(obsstates)} trajectories loaded')
    except:
        print(f'File {fname} not found, init empty obs file')

    count = len(obsstates)
    while count < numTrajectories:
        sim  = swarm( numIndividuals, numNearestNeighbours,  numdimensions, 2, initializationType, _psi=psi, seed=seed+count )
        action = np.zeros(shape=(sim.dim), dtype=float)
        vec = np.random.normal(0.,1.,sim.dim)
        mag = np.linalg.norm(vec)
        action = vec/mag
       
        states = []
        actions = []
        rewards = []
        
        step = 0

        while (step < numTimeSteps):
            sim.preComputeStates()
            sim.angularMoments.append(sim.computeAngularMom())
            sim.polarizations.append(sim.computePolarisation())
            sim.move_calc()
             
            state = [ list(sim.getState(i)) for i in range(numIndividuals) ]
            action = [ [sim.fishes[i].getAction()] for i in range(numIndividuals) ]
            reward = list(sim.getGlobalReward())[0]
            print("trajectory {}: timestep {}/{} reward {}".format(count, step+1, numTimeSteps, reward))

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            for i in np.arange(sim.N):
                sim.fishes[i].curDirection = sim.fishes[i].applyrotation(sim.fishes[i].curDirection, action[i])
                sim.fishes[i].updateLocation()

            step += 1
        
        if reward >= 0.9:
            obsstates.append(states)
            obsactions.append(actions)
            obsrewards.append(rewards)

        count = len(obsstates)

    observations["States"] = obsstates
    observations["Actions"] = obsactions
    observations["Rewards"] = obsrewards

    with open(fname,'w') as f:
        json.dump(observations, f)

    print(f"Saved {len(obsstates)} trajectories")
