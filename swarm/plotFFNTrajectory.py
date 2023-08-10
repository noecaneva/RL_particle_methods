import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.append('_model')
from swarm import *

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.models import load_model

def plotSwarm3D(obj, locations, directions, seed):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    locations = locations[-1,:,:]
    directions = directions[-1,:,:]
    N, _ = locations.shape
    
    cmap = plt.cm.inferno
    norm = Normalize(vmin=0, vmax=N)

    colors = []
    norm = Normalize(vmin=0, vmax=N)
    csel = plt.cm.inferno(norm(np.arange(N)))
    for i in range(N):
        colors.append(csel[i])
    for i in range(N):
        colors.append(csel[i])
        colors.append(csel[i])

    ax.quiver(locations[:,0],locations[:,1],locations[:,2], directions[:,0], directions[:,1], directions[:,2], colors=colors, normalize=True, length=1.)
    
    fig.tight_layout()
    plt.savefig(f'swarm_ffn_o{obj}_{seed}.pdf', dpi=400)
    plt.close('all')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfile', help='stored h5 file', type=str, required=False)
    #parser.add_argument('--i', help='dim idx', required=False, type=int, default=0)
    #parser.add_argument('--j', help='dim idx', required=False, type=int, default=1)
    parser.add_argument('--NN', default=7, type=int, required=False)
    parser.add_argument('--NF', default=25, type=int, required=False)
    parser.add_argument('--D', default=3, type=int, required=False)
    parser.add_argument('--obj', default=0, type=int, required=False)
    parser.add_argument('--seed', default=1337, type=int, required=False)

    args = parser.parse_args()
    
    numVectorsInState = 5
    print(f"Loading model from {args.modelfile}")
    model = load_model(args.modelfile)
    model.summary()

    sim = swarm( N=args.NF, numNN=args.NN, numdimensions=args.D, initType=1, movementType=2, seed=args.seed)
        
    locations = []
    momentum = []
    polarization = []
    directions = []

    cumReward = 0
    for t in range(1000): 

        done = sim.preComputeStates()
        states  = np.zeros((sim.N,  args.NN * numVectorsInState))
        for fish in range(args.NF):
            states[fish,:] = sim.getState(fish)

        fishdirs = []
        fishlocs = []

        actions = model.predict(states)
        for fish in np.arange(sim.N):
            sim.fishes[fish].curDirection = sim.fishes[fish].applyrotation(sim.fishes[fish].curDirection, actions[fish,:])
            sim.fishes[fish].updateLocation()

            fishdirs.append(sim.fishes[fish].curDirection)
            fishlocs.append(sim.fishes[fish].location)

        momentum.append(sim.computeAngularMom())
        polarization.append(sim.computePolarisation())
        locations.append(np.array(fishlocs))
        directions.append(np.array(fishdirs))

        # compute pair-wise distances and view-angles
        rewards = sim.getGlobalReward(args.obj)

        cumReward += rewards[0]
        print(f"t={t}: {cumReward/t}")

    print(cumReward)
    locations = np.array(locations)
    directions = np.array(directions)
    NT, NF, D = locations.shape

    print(f"plotting..")
    plotSwarm3D(args.obj, locations, directions, args.seed)

    fig, axs = plt.subplots(1, D, gridspec_kw={'width_ratios': [1, 1, 1]}) #, 'height_ratios': [1]})
    colors = plt.cm.Oranges(np.linspace(0, 1, NF))

    axs[0].plot(polarization, color='steelblue')
    axs[0].plot(momentum, color='coral')
    axs[0].set_box_aspect(1)
    axs[0].set_yticks([0.1, 0.5, 0.9])
    axs[0].set_ylim([0.0, 1.0])
    for d in range(D-1):
        for fish in range(NF):
          traj = locations[:,fish, :]
          axs[1+d].plot(traj[:,d], traj[:,d+1], color=colors[fish])
        axs[1+d].set_aspect('equal') #, 'box')
        axs[1+d].set_box_aspect(1)

    fig.tight_layout()
    figname = f'ffn_traj_o{args.obj}_{args.seed}.pdf'
    print(f"saving figure {figname}..")
    plt.savefig(figname)
    print(f"done!")
