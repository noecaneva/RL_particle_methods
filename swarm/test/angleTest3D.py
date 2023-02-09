import sys
import numpy as np

locations = np.array([[0.,0., 0.],[0.,1., 0.],[1.,0., 1.0]])
curDirections = np.array([[-1.,-1., 0.],[1., 0., 0.],[0.,-1., 1.0]])
N = 3
dimension = 3

# normalize swimming directions NOTE the directions should already be normalized
curDirections = curDirections/(np.sqrt(np.einsum('ij,ij->i', curDirections, curDirections))[:,np.newaxis])


## create containers for direction, distance, and angle
directionsOtherFish    = np.empty(shape=(N,N, dimension ), dtype=float)
distances              = np.empty(shape=(N,N),    dtype=float)
angles                 = np.empty(shape=(N,N),    dtype=float)
dotprod                = np.empty(shape=(N,N),    dtype=float)
detprod                = np.empty(shape=(N,N),    dtype=float)
anglesVel              = np.empty(shape=(N,N),    dtype=float)

## use numpy broadcasting to compute direction, distance, and angles
directionsOtherFish    = locations[np.newaxis, :, :] - locations[:, np.newaxis, :]
distances     = np.sqrt( np.einsum('ijk,ijk->ij', directionsOtherFish, directionsOtherFish) )
# Filling diagonal to avoid division by 0
np.fill_diagonal( distances, 1.0 )
# normalize direction
normalDirectionsOtherFish = directionsOtherFish / distances[:,:,np.newaxis]

print(curDirections)
print(normalDirectionsOtherFish)

curDirectionsXY = curDirections[:,:-1]
normalDirectionsOtherFishXY = normalDirectionsOtherFish[:,:,:-1]

print(curDirectionsXY)
print(normalDirectionsOtherFishXY)

dotprod = np.einsum( 'ijk, ijk->ij', curDirectionsXY[:,np.newaxis,:], normalDirectionsOtherFishXY )
# reverse order along the third axis
detprod = -np.flip(normalDirectionsOtherFishXY,2)
# invert sign of second element along third axis
detprod = np.einsum('ijk, ijk->ijk',np.array([1.,-1.])[np.newaxis,np.newaxis,:],detprod)
detprod = np.einsum( 'ijk, ijk->ij', curDirectionsXY[:,np.newaxis,:], detprod)
angles = np.arctan2(detprod, dotprod)

print("With the respect of the position")
print(angles*180/np.pi)
print((angles*180/np.pi)[0])
print()
print()

print("dir other fish")
print(directionsOtherFish)
print("cur dir")
print(curDirections)
dz = directionsOtherFish[:,:,-1]-curDirections[:,-1]
#print(d)
#print(d[:,:,-1])
print("z angle")
print(np.arcsin(dz)*180/np.pi)

curDirectionsZ = curDirections[:,-1]
directionsOtherFishZ = directionsOtherFish[:,:,-1]

#print(curDirectionsZ)
#print(np.arcsin(curDirectionsZ)*180/np.pi)
#print(directionsOtherFishZ)
#print(np.arcsin(directionsOtherFishZ)*180/np.pi)
#rotZ = np.arcsin(directionsOtherFishZ)-np.arcsin(curDirectionsZ[:,np.newaxis])
#print(rotZ*180/np.pi)
#print(rotZ)
#print(np.arcsin(directionsOtherFishZ)*180/np.pi-np.arcsin(curDirectionsZ[:,np.newaxis])*180/np.pi)
#print(rotZ)
#print(directionsOtherFishZ - curDirectionsZ[:,np.newaxis])
#print(np.arcsin((directionsOtherFishZ - curDirectionsZ[:,np.newaxis]))*180/np.pi)
