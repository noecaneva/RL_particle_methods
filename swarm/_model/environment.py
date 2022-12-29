from swarm import *
from pathlib import Path

def environment( args, s ):

    # set set parameters and initialize environment
    numIndividuals       = args["N"]
    numTimesteps         = args["NT"]
    numNearestNeighbours = args["NN"]
    dim                  = args["dim"]
    globalreward         = True if args["reward"] == "global" else False
   
    movementType        = 2 # 0 is hardcoded, 1 is random, 2 is according to the related papers
    initializationType  = 1 # random uniform in circle
    alpha               = 4.49 # vision of fish in radian
   
    sim = swarm( N=numIndividuals, numNN=numNearestNeighbours,
    numdimensions=dim, initType=initializationType, movementType=movementType, _alpha=alpha)
 
    # compute pair-wise distances and view-angles
    done = sim.preComputeStates()

    # set initial state
    numVectorsInState = sim.nrVectorStates
    if(sim.toyexample):
        states  = np.zeros((sim.N, sim.toystate))
    else:
        states  = np.zeros((sim.N, numNearestNeighbours * numVectorsInState))
    
    print("State of the simulation")
    print()
    
    print("STATES WILL FOLLOW")
    for i in np.arange(sim.N):
        # get state
        states[i,:] = sim.getState( i )
        print(states[i,:])

    # print("states:", states)
    # print(states[0])
    s["State"] = states.tolist()

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
            # plotSwarmCentered( sim, step )
            followcenter=True
            if(sim.dim == 3):
                plotSwarm3D( sim, step, followcenter, step, numTimesteps)
            else:
                plotSwarm2D( sim, step, followcenter, step, numTimesteps)

        # Getting new action
        s.update()

        ## apply action, get reward and advance environment
        actions = s["Action"]
        #print("actions:", actions)

        if dim == 2:
            for i in np.arange(sim.N):
                # compute wished direction based on action
                print("Swimmer ", i)
                phi = actions[i][0]
                print("phi :", phi)
                x = np.cos(phi)
                y = np.sin(phi)
                sim.fishes[i].wishedDirection = [ x, y ]
                print("sim.fishes[i].wishedDirection: ", sim.fishes[i].wishedDirection)
                print("sim.fishes[i].curDirection: ", sim.fishes[i].curDirection)
                # rotation in wished direction
                sim.fishes[i].updateDirection()
                print("sim.fishes[i].curDirection: ", sim.fishes[i].curDirection)
                print("sim.fishes[i].location: ", sim.fishes[i].location)
                # update positions
                sim.fishes[i].updateLocation()
                print("sim.fishes[i].location: ", sim.fishes[i].location)

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

        print("States Will Follow of the simulation")
                
        # set state
        states  = np.zeros((sim.N, numNearestNeighbours * sim.nrVectorStates))
        if (globalreward and not sim.toyexample):
            rewards = sim.getGlobalReward()
        elif(not globalreward and not sim.toyexample):
            rewards = sim.getLocalReward()
        if (sim.toyexample):
            rewards = sim.getToyReward()
        for i in np.arange(sim.N):
            # get state
            states[i,:] = sim.getState( i )
            print("states[i,:] : ",states[i,:])

        #print("states:", state)
        s["State"] = states.tolist()
        print("rewards:", rewards)
        s["Reward"] = (rewards / numTimesteps).tolist()

        step += 1

    # Setting termination status
    if done:
        s["Termination"] = "Terminal"
    else:
        s["Termination"] = "Truncated"
