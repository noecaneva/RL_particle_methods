#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import json
import shutil as sh
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

sys.path.append('_model')
from swarm import *

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

parser = argparse.ArgumentParser()
parser.add_argument(
    '--epochs',
    help='Maximum Number of generations to run',
    default=500,
    type=int,
    required=False)
parser.add_argument(
    '--NN',
    default=7,
    type=int,
    required=False)
parser.add_argument(
    '--obj',
    default=0,
    type=int,
    required=False)    
parser.add_argument(
    '--learningRate',
    help='Learning rate for the selected optimizer',
    default=0.0001,
    type=float,
    required=False)
parser.add_argument('--file', help='trajectory file', required=True, type=str)
args = parser.parse_args()

print("Running FNN solver with arguments:")
print(args)

basedir = f"./shuffle_training_ffn_o{args.obj}/"
filepath = basedir + 'cp-{epoch:04d}.ckpt'
print(basedir)
print(filepath)

# Opening JSON file
f = open(args.file)

# returns JSON object as 
# a dictionary
data = json.load(f)
locations = np.array(data["Locations"])
directions = np.array(data["Directions"])
actionsRaw = np.array(data["Actions"])
N, NT, NF, D = locations.shape
#locations = locations.reshape((-1,D))
print(locations.shape)
print(directions.shape)
_, _, _, NA = actionsRaw.shape
print(actionsRaw.shape)
print(f"loaded {N} trajectories of length {NT} in {D}d, number of actions {NA}, swarm size {NF}")

print("preprocessing data..")
states = []
actions = []
for k in range(10): #N
    for t in range(NT): #NT
        sim = swarm( N=NF, numNN=args.NN,
            numdimensions=D, initType=1, movementType=2)

        sim.initFromLocationsAndDirections(locations[k,t,:,:], directions[k,t,:,:])
        sim.preComputeStates()
        for fish in range(NF):
            states.append(sim.getState(fish))
            actions.append(actionsRaw[k,t,fish,:])

states = np.array(states)
actions = np.array(actions)
np.random.shuffle(states)
np.random.shuffle(actions)
print("new states shape")
print(states.shape)
_, NS = states.shape
print("new actions shape")
print(actions.shape)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    save_weights_only=True,
    save_freq=100)

v_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    monitor='val_loss',
    save_best_only=True,
    mode='auto'
)

model = Sequential()
model.add(Dense(256, activation='tanh'))
model.add(Dense(256,activation='tanh'))
model.add(Dense(NA,activation=None))

opt = tf.keras.optimizers.Adam(learning_rate=args.learningRate)
model.compile(loss='mse', optimizer='adam')
history = model.fit(x=states, y=actions, epochs=args.epochs, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[v_checkpoint])
model.summary()
model.save(f'{basedir}final_model_o{args.obj}')
