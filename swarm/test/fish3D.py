import sys
sys.path.append('../_model')

import fish

location = [0,0,0]
idir = [0,0,1]
psi = -1.

f = fish.fish(location, idir, 3, psi)
angle = [0., 0.1]
wdir = f.applyrotation(f.curDirection,angle)
f.wishedDirection = wdir
ac = f.getAction()
print(ac)


f = fish.fish(location, idir, 3, psi)
angle = [0., 0.05]
wdir = f.applyrotation(f.curDirection,angle)
f.wishedDirection = wdir
ac = f.getAction()
print(ac)

f = fish.fish(location, idir, 3, psi)
angle = [0.1, 0.]
wdir = f.applyrotation(f.curDirection,angle)
f.wishedDirection = wdir
ac = f.getAction()
print(ac)

f = fish.fish(location, idir, 3, psi)
angle = [0.05, 0.]
wdir = f.applyrotation(f.curDirection,angle)
f.wishedDirection = wdir
ac = f.getAction()
print(ac)


f = fish.fish(location, idir, 3, psi)
angle = [0.1, 0.1]
wdir = f.applyrotation(f.curDirection,angle)
f.wishedDirection = wdir
ac = f.getAction()
print(ac)

f = fish.fish(location, idir, 3, psi)
angle = [0.01, 0.01]
wdir = f.applyrotation(f.curDirection,angle)
f.wishedDirection = wdir
ac = f.getAction()
print(ac)
