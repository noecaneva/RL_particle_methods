import numpy as np

locations = np.array([[0.,0.],[0.,1.],[1.,0.]])
curDirections = np.array([[-1.,-1.],[1.,0.],[0.,-1.]])
N = 2
dimension = 2
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

dotprod = np.einsum( 'ijk, ijk->ij', curDirections[:,np.newaxis,:], normalDirectionsOtherFish )
# reverse order along the third axis
detprod = np.flip(normalDirectionsOtherFish,2)
# invert sign of second element along third axis
detprod = np.einsum('ijk, ijk->ijk',np.array([1.,-1.])[np.newaxis,np.newaxis,:],detprod)
detprod = np.einsum( 'ijk, ijk->ij', curDirections[:,np.newaxis,:], detprod)
angles = np.arctan2(detprod, dotprod)
print("With the respect of the position")
print(angles*180/np.pi)
print((angles*180/np.pi)[0])
print()
print()
print("With respect of the velocities")
print(curDirections)
dotprod = np.einsum( 'ijk, ijk->ij', curDirections[:,np.newaxis], curDirections[np.newaxis,:])
detprod = np.flip(curDirections,1)
print(detprod.shape)
print(detprod)
detprod = np.einsum('ij, ij->ij',np.array([1.,-1.])[np.newaxis,:],detprod)
print(detprod)
detprod = np.einsum( 'ijk, ijk->ij', curDirections[:,np.newaxis], detprod[np.newaxis,:])
anglesVel  = np.arctan2(detprod, dotprod)
print(anglesVel*180/np.pi)

print(angles*180/np.pi)
mask = angles == 0.
angles = np.ma.array(angles, mask=mask)
angles = angles.filled(-np.pi)
print(angles*180/np.pi)
