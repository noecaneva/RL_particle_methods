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
    parser.add_argument('--reps', default=20, type=int, required=False)

    args = parser.parse_args()
    
    numVectorsInState = 5
    print(f"Loading model from {args.modelfile}")
    model = load_model(args.modelfile)
    model.summary()

    reps = args.reps
    T = 1000
    momentum = np.zeros((reps,T))
    polarization = np.zeros((reps,T))
    rewards = np.zeros((reps,T))
    for r in range(reps):

        sim = swarm( N=args.NF, numNN=args.NN, numdimensions=args.D, initType=1, movementType=2, seed=args.seed+r)
            
        cumReward = 0
        for t in range(T): 

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

            # compute pair-wise distances and view-angles
            rewards[r,t] = sim.getGlobalReward(args.obj)[0]
            momentum[r,t] = sim.computeAngularMom()
            polarization[r,t] = sim.computePolarisation()

            cumReward += rewards[r,t]
            print(f"rep {r}/{reps} t={t}: {cumReward/t}")


    CI = 0.8
    t = np.arange(T)
    cumrewards = np.cumsum(rewards, axis=0)
    mediancumrewards = np.median(cumrewards, axis=0)
    cILower = np.percentile(cumrewards, 50-50*CI, axis=0)
    cIUpper = np.percentile(cumrewards, 50+50*CI, axis=0)
 
    medianmomentum = np.median(momentum, axis=0)
    cILowerMom = np.percentile(momentum, 50-50*CI, axis=0)
    cIUpperMom = np.percentile(momentum, 50+50*CI, axis=0)
 
    medianpolarization = np.median(polarization, axis=0)
    cILowerPol = np.percentile(polarization, 50-50*CI, axis=0)
    cIUpperPol = np.percentile(polarization, 50+50*CI, axis=0)

    fig, axs = plt.subplots(1, 1, gridspec_kw={'width_ratios': [1]})
    axs.plot(medianmomentum, color='coral')
    axs.fill_between(t, cILowerMom, cIUpperMom, alpha=0.5, color='coral')
    axs.plot(medianpolarization, color='steelblue')
    axs.fill_between(t, cILowerPol, cIUpperPol, alpha=0.5, color='steelblue')
    axs.set_yticks([0.1, 0.5, 0.9])
    axs.set_ylim([0.0, 1.0])
    plt.savefig(f"ffn_mom_pol_200_o{args.obj}.pdf")
    plt.close('all')

    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
    axs[0].plot(medianmomentum, color='coral')
    axs[0].fill_between(t, cILowerMom, cIUpperMom, alpha=0.5, color='coral')
    axs[0].plot(medianpolarization, color='steelblue')
    axs[0].fill_between(t, cILowerPol, cIUpperPol, alpha=0.5, color='steelblue')
    axs[1].plot(t,mediancumrewards, color='blue')
    axs[1].fill_between(t, cILower, cIUpper, alpha=0.5, color='blue')
    plt.savefig(f"ffn_mom_pol_rew_200_o{args.obj}.pdf")
    plt.close('all')
