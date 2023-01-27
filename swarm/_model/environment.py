from swarm import *
from pathlib import Path

def environment( args, s ):

    # set set parameters and initialize environment
    numIndividuals       = args.N
    numTimesteps         = args.NT
    numNearestNeighbours = args.NN
    dim                  = args.dim
    globalreward         = True if args.reward == "global" else False
   
    movementType        = 2 # 0 is hardcoded, 1 is random, 2 is according to the related papers
    initializationType  = 1 # random uniform in circle
    alpha               = 4.49 # vision of fish in radian

    sampleId = s["Sample Id"]
    seeds = [10, 30, 70] #, 120, 130, 170, 190, 220, 230, 240, 250, 260, 270, 280,  290, 300]
    seed = seeds[sampleId % len(seeds)]
   
    sim = swarm( N=numIndividuals, numNN=numNearestNeighbours,
        numdimensions=dim, initType=initializationType, movementType=movementType, _alpha=alpha, seed=seed)
 
    # compute pair-wise distances and view-angles
    done = sim.preComputeStates()

    # set initial state
    numVectorsInState = sim.nrVectorStates
    states  = np.zeros((sim.N, numNearestNeighbours * numVectorsInState))
    for i in np.arange(sim.N):
        # get state
        states[i,:] = sim.getState( i )

    #print("states:", states)
    s["State"] = states.tolist()
    s["Features"] = states.tolist()

    ## run simulation
    step = 0
    if done: 
        print("Initial configuration is terminal state...")

    while (step < numTimesteps) and (not done):
        if args.visualize:
            Path("./_figures").mkdir(parents=True, exist_ok=True)
            # fixed camera
            # plotSwarm( sim, step )
            # camera following center of swarm
            # plotSwarmCentered( sim, step )
            followcenter=True
            if(sim.dim == 3):
                plotSwarm3D( sim, step, followcenter, step, numTimesteps)
            else:
                print(f"Visualizing step {step}")
                plotSwarm2D( sim, step, followcenter, step, numTimesteps)

        sim.angularMoments.append(sim.computeAngularMom())
        sim.polarizations.append(sim.computePolarisation())

	# Getting new action
        s.update()

        ## apply action, get reward and advance environment
        actions = s["Action"]
        #print("actions:", actions)

        if dim == 2:
            for i in np.arange(sim.N):
                # compute wished direction based on action
                phi = actions[i][0]
                currentDir = sim.fishes[i].curDirection
                sim.fishes[i].curDirection = sim.fishes[i].applyrotation(currentDir, phi)
                # update positions
                sim.fishes[i].updateLocation()
        else:
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
        states  = np.zeros((sim.N, numNearestNeighbours * sim.nrVectorStates))
        rewards = sim.getGlobalReward() if globalreward else sim.getLocalReward()
        for i in np.arange(sim.N):
            # get state
            states[i,:] = sim.getState( i )

        #print("states:", state)
        s["State"] = states.tolist()
        s["Features"] = states.tolist()
        #print("rewards:", rewards)
        s["Reward"] = (rewards / numTimesteps).tolist()

        step += 1

    # Setting termination status
    if done:
        s["Termination"] = "Terminal"
    else:
        s["Termination"] = "Truncated"
