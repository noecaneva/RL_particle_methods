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
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument(
    '--steps',
    help='Maximum Number of generations to run',
    default=2000,
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
parser.add_argument(
    '--train',
    action='store_true',
    required=False)    
parser.add_argument(
    '--modelfile',
    help='stored h5 file',
    type=str,
    required=False)    
parser.add_argument(
    '--file',
    help='trajectory file',
    required=True,
    type=str)

print("Running FNN solver with arguments:")
args = parser.parse_args()
print(args)

basedir = f"./training_ffn_o{args.obj}/"
filepath = basedir + 'cp-{epoch:04d}.ckpt'
print(basedir)

if args.train == False:
    print(f"Loading model from {args.modelfile}")
    model = load_model(args.modelfile)
    model.summary()

states = []
actions = []
testStates = []
testActions = []

# Opening JSON file
f = open(args.file)

# returns JSON object as 
# a dictionary
data = json.load(f)
locations = np.array(data["Locations"])
directions = np.array(data["Directions"])
actionsRaw = np.array(data["Actions"])
N, NT, NF, D = locations.shape

print(locations.shape)
print(directions.shape)
_, _, _, NA = actionsRaw.shape
print(actionsRaw.shape)
print(f"loaded {N} trajectories of length {NT} in {D}d, number of actions {NA}, swarm size {NF}")

print("preprocessing data..")
for k in range(N):
    for t in range(NT): #NT
        sim = swarm( N=NF, numNN=args.NN,
            numdimensions=D, initType=1, movementType=2)

        sim.initFromLocationsAndDirections(locations[k,t,:,:], directions[k,t,:,:])
        sim.preComputeStates()
        for fish in range(NF):
            if k < 3:
                testStates.append(sim.getState(fish))
                testActions.append(actionsRaw[k,t,fish,:])
            else:
                states.append(sim.getState(fish))
                actions.append(actionsRaw[k,t,fish,:])

states = np.array(states)
actions = np.array(actions)
testStates= np.array(testStates)
testActions = np.array(testActions)

#np.random.shuffle(states)
#np.random.shuffle(actions)
print("new states shape")
print(states.shape)
#_, NS = states.shape
print("new actions shape")
print(actions.shape)

print("test states shape")
print(testStates.shape)
print("test actions shape")
print(testActions.shape)

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

if args.train == True:

    model = Sequential()
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(128,activation='tanh'))
    model.add(Dense(NA,activation=None))
    model.compile(loss='mse', optimizer='adam')

    opt = tf.keras.optimizers.Adam(learning_rate=args.learningRate)
    history = model.fit(x=states, y=actions, epochs=args.steps, steps_per_epoch=1, batch_size=128, shuffle=True, validation_split=0.2, callbacks=[v_checkpoint])
    model.summary()
    model.save(f'{basedir}/final_o{args.obj}.h5')
    np.save(f'{basedir}/history_o{args.obj}.npy',history.history)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print(history.history)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mse')
    plt.ylabel('loss')
    plt.xlabel('steps')
    plt.legend(['train', 'validation']) #, loc='upper left')
    plt.savefig(f'{basedir}/history_mse_o{args.obj}.pdf')

loss = model.evaluate(testStates, testActions, verbose=2)
prediction = model.predict(testStates)
print(prediction)
print(prediction.shape)
