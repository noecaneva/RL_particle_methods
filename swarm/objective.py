import sys
sys.path.append('_model')
from swarm import *
from plotter3D import *
import math
from pathlib import Path
import numpy as np

def objectivefunction(rRepulsion, delrOrientation, delrAttraction,psi):
    N = 50
    numdimensions = 3
    numNearestNeighbours = 3
    movementType = 2 # 0 is hardcoded, 1 is random, 2 is according to the related papers
    initializationType = 2 # 0 for grid, 1 for on circle or sphere, 2 for within a circle or a sphere
    numTimeSteps = 50
    visualize = True

    sim  = swarm( N, numNearestNeighbours,  numdimensions, movementType, initializationType, _rRepulsion=rRepulsion, _delrOrientation=delrOrientation, _delrAttraction=delrAttraction,_psi=psi)
    step = 0
    done = False
    action = np.zeros(shape=(sim.dim), dtype=float)
    vec = np.random.normal(0.,1.,sim.dim)
    mag = np.linalg.norm(vec)
    action = vec/mag
    
    while (step < numTimeSteps):
        print("timestep {}/{}".format(step+1, numTimeSteps))
        # if enable, plot current configuration
        if visualize:
            Path("./_figures").mkdir(parents=True, exist_ok=True)
            followcenter=True
            if(sim.dim == 3):
                plotSwarm3D( sim, step, followcenter, step, numTimeSteps)
            else:
                plotSwarm2D( sim, step, followcenter, step, numTimeSteps)

        # compute pair-wise distances and view-angles
        done = sim.preComputeStates()

        if(movementType == 2):
            sim.move_calc()

        # update swimming directions
        for i in np.arange(sim.N):
            #print("agent {}/{}".format(i+1, sim.N))
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
                    action = vec/mag
                    sim.fishes[i].wishedDirection = action
            elif(movementType != 2):
                raise Exception("Unknown movement type please choose 0, 1 or 2")

            # rotation in wished direction
            sim.fishes[i].updateDirection()
            # update positions
            sim.fishes[i].updateLocation()

        step += 1


    avgAngMom = np.mean(np.array(sim.angularMoments))
    avgPol = np.mean(np.array(sim.polarizations))
    return avgAngMom, avgPol

a,b = objectivefunction(0.1, 1.5, 3., 1.)
print(a, b)