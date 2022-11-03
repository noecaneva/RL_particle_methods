import sys
sys.path.append('_model')
from swarm import *
from plotter3D import *
import math
from pathlib import Path
import numpy as np

def objectivefunction(p, dim, momentum):
    
    rRepulsion      = p["Parameters"][0]
    delrOrientation = p["Parameters"][1]
    delrAttraction  = p["Parameters"][2]
    alpha           = p["Parameters"][3]

    N           = 100
    totalTime   = 100
    
    numNearestNeighbours    = 3 # unused
    movementType            = 2 # 0 is hardcoded, 1 is random, 2 is according to the related papers
    initializationType      = 1 # random uniform in circle
    pctAvg                  = 0.95
    visualize               = False
    
    # Init simulation
    sim  = swarm( N, numNearestNeighbours, dim, movementType,
        initializationType, _rRepulsion=rRepulsion, _delrOrientation=delrOrientation,
            _delrAttraction=delrAttraction, _alpha=alpha)
    
    step = 0
    numTimeSteps = round(totalTime/sim.fishes[0].dt)
    lastElements = round(numTimeSteps * pctAvg)
    
    # Error handling if it is not possible to initialize a swarm that is well connected
    if(sim.tooManyInits):
        avgAngMom = -1.
        avgPol = -1.
    
    else:
        while (step < numTimeSteps):
            # compute pair-wise distances and view-angles
            done = sim.preComputeStates()
            sim.move_calc()
            for i in range(N):
                # rotation in wished direction
                sim.fishes[i].updateDirection()
                # update positions
                sim.fishes[i].updateLocation()

            step += 1

        avgAngMom = np.mean(np.array(sim.angularMoments)[-lastElements:])
        avgPol = np.mean(np.array(sim.polarizations)[-lastElements:])

    print(f"avg polarization: {avgPol}")
    print(f"avg angular momentum: {avgAngMom}")

    if momentum:
        p["F(x)"] = avgAngMom
    else:
        p["F(x)"] = avgPol
