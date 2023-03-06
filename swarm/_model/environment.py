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
    storeGoodEpisode = s["Custom Settings"]["Store Good Episodes"]

    locationHistory = []
    directionHistory = []

    if dim == 2:
        seeds = [1, 2, 3, 4, 5]
    else:
        seeds = [1, 2, 3, 4, 5]

    seed = seeds[sampleId % len(seeds)]
   
    sim = swarm( N=numIndividuals, numNN=numNearestNeighbours,
        numdimensions=dim, initType=initializationType, movementType=movementType, _alpha=alpha, seed=seed)
 
    # compute pair-wise distances and view-angles
    done = sim.preComputeStates()

    # set initial state
    if(dim == 2):
        numVectorsInState = 2
        sim.numStates = numNearestNeighbours * numVectorsInState
        states  = np.zeros((sim.N, sim.numStates))
    else:
        numVectorsInState = 5
        sim.numStates = numNearestNeighbours * numVectorsInState
        states  = np.zeros((sim.N, sim.numStates))
    for i in np.arange(sim.N):
        # get state
        states[i,:] = sim.getState( i )

    print("states:", states.tolist())
    s["State"] = states.tolist()

    ## run simulation
    step = 0
    cumReward = 0
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

        print("BEFORE UPDATE__________________")
        print("s[State]: ",s["State"][0][0])


   	    # Getting new action
        s.update()
        print("AFTER UPDATE__________________")

        ## apply action, get reward and advance environment
        actions = s["Action"]

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
                
                sim.fishes[i].curDirection = sim.fishes[i].applyrotation(sim.fishes[i].curDirection, actions[i])
                sim.fishes[i].updateLocation()

        # compute pair-wise distances and view-angles
        done = sim.preComputeStates()
        
        # states  = np.zeros((sim.N, sim.numStates))
        if(dim == 2):
            numVectorsInState = 2
            sim.numStates = numNearestNeighbours * numVectorsInState
            states  = np.zeros((sim.N, sim.numStates))
        else:
            numVectorsInState = 5
            sim.numStates = numNearestNeighbours * numVectorsInState
        
        rewards = sim.getGlobalReward() if globalreward else sim.getLocalReward()
        for i in np.arange(sim.N):
            states[i,:] = sim.getState( i )

        print("states later on:", states)
        s["State"] = states.tolist()
        s["Features"] = states.tolist()
        rewards = (rewards / numTimesteps).tolist()
        s["Reward"] = rewards

        step += 1
        cumReward += rewards[0]

        if storeGoodEpisode == "True":
            locations = []
            directions = []
            for fish in sim.fishes:
                locations.append(fish.location)
                directions.append(fish.curDirection)

            locationHistory.append(locations)
            directionHistory.append(directions)

    if cumReward > 0.8:
        fname = f"trajectory_{sampleId}.npz"
        print(f"Dumping trajectory with cumulative reward {cumReward} to file {fname}")
        locationHistory = np.array(locationHistory)
        directionHistory = np.array(directionHistory)
        np.savez(fname, cumReward=cumReward, locationHistory=locationHistory, directionHisory=directionHistory)

    # Setting termination status
    if done:
        s["Termination"] = "Terminal"
    else:
        s["Termination"] = "Truncated"
