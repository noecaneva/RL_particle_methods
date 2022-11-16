from swarm import *
from pathlib import Path

def environment( args, s ):

    # set set parameters and initialize environment
    numIndividuals       = args["N"]
    numTimesteps         = args["NT"]
    numNearestNeighbours = args["NN"]
    dim                  = args["dim"]
   
    movementType            = 2 # 0 is hardcoded, 1 is random, 2 is according to the related papers
    initializationType      = 1 # random uniform in circle
   
    sim = swarm( N=numIndividuals, numNN=numNearestNeighbours, numdimensions=dim, initType=initializationType, movementType=movementType )
 
    # compute pair-wise distances and view-angles
    done = sim.preComputeStates()

    # set initial state
    states = []
    for i in np.arange(sim.N):
        # get state
        state = sim.getState( i )
        states.append( state )
    
    #print("states:", states)
    s["State"] = states

    ## run simulation
    step = 0
    if done: 
        print("Initial configuration is terminal state...")

    while (step < numTimesteps) and (not done):
        if args["visualize"]:
            Path("./_figures").mkdir(parents=True, exist_ok=True)
            # fixed camera
            # plotSwarm( sim, step )
            # camera following center of swarm
            plotSwarmCentered( sim, step )

        # Getting new action
        s.update()

        ## apply action, get reward and advance environment
        actions = s["Action"]
        #print("actions:", actions)
        for i in np.arange(sim.N):
            # compute wished direction based on action
            phi = actions[i][0]
            theta = actions[i][1]
            x = np.cos(phi)*np.sin(theta)
            y = np.sin(phi)*np.sin(theta)
            z = np.cos(theta)
            sim.fishes[i].wishedDirection = [ x, y, z ]
            # rotation in wished direction
            sim.fishes[i].updateDirection()
            # update positions
            sim.fishes[i].updateLocation()

        # compute pair-wise distances and view-angles
        #print("precomp")
        done = sim.preComputeStates()
        #print("done")
        
        # set state
        states  = []
        rewards = sim.getGlobalReward()
        for i in np.arange(sim.N):
            # get state
            states.append(sim.getState( i ))

        #print("states:", state)
        s["State"] = states
        #print("rewards:", rewards)
        s["Reward"] = rewards

        step += 1

    # Setting termination status
    if done:
        s["Termination"] = "Terminal"
    else:
        s["Termination"] = "Truncated"
