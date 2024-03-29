from swarm import *
from pathlib import Path
from plotter3D import plotSwarm3DEnv

episodeId = 0

def environment( args, s ):

    global episodeId

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

    centerHistory = []
    avgDistHistory = []
    locationHistory = []
    directionHistory = []

    seed = episodeId % 3
    #numVectorsInState = args.dim
    numVectorsInState = 5
   
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

    s["State"] = states.tolist()

    ## run simulation
    step = 0
    cumReward = 0
    if done: 
        print("Initial configuration is terminal state...")

    while (step < numTimesteps) and (not done):

        if storeGoodEpisode == "True":
            locations = []
            directions = []
            for fish in sim.fishes:
                locations.append(fish.location.copy())
                directions.append(fish.curDirection.copy())

            centerHistory.append(sim.computeCenter())
            avgDistHistory.append(sim.computeAvgDistCenter(centerHistory[-1]))
            locationHistory.append(locations.copy())
            directionHistory.append(directions.copy())

        if args.visualize:
            Path("./_figures").mkdir(parents=True, exist_ok=True)
            # fixed camera
            # plotSwarm( sim, step )
            # camera following center of swarm
            # plotSwarmCentered( sim, step )
            followcenter=True
            if(sim.dim == 3):
                print(f"Visualizing step {step}")
                plotSwarm3D( sim, step, followcenter, step, numTimesteps)
            else:
                print(f"Visualizing step {step}")
                plotSwarm2D( sim, step, followcenter, step, numTimesteps)

        s.update()
   	# Getting new action

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

        sim.angularMoments.append(sim.computeAngularMom())
        sim.polarizations.append(sim.computePolarisation())

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

        s["State"] = states.tolist()
        s["Features"] = states.tolist()
        rewards = (rewards / numTimesteps).tolist()
        s["Reward"] = rewards

        step += 1
        cumReward += rewards[0]


    if cumReward > 0.8:
        fname = f"trajectory_{episodeId}.npz"
        print(f"Dumping trajectory with cumulative reward {cumReward} to file {fname}")
        #print(f"locationHistory size {locationHistory.shape}")
        #print(f"directionHistory size {directionHistory.shape}")
        np.savez(fname, cumReward=cumReward, locationHistory=locationHistory, directionHisory=directionHistory, centerHistory=centerHistory, avgDistHistory=avgDistHistory)
        plotSwarm3DEnv(episodeId, True, True, sim.N, locationHistory, directionHistory, centerHistory, avgDistHistory, sim.angularMoments, sim.polarizations)

    episodeId += 1

    # Setting termination status
    if done:
        s["Termination"] = "Terminal"
    else:
        s["Termination"] = "Truncated"
