import numpy as np

locations = np.array([[0.,0.],[0.,2.]])
curDirections = np.array([[1.,1.],[1.,0.]])
N = 2
# normalize swimming directions NOTE the directions should already be normalized
normalCurDirections = curDirections/(np.sqrt(np.einsum('ij,ij->i', curDirections, curDirections))[:,np.newaxis])

## create containers for direction, distance, and angle
directionsOtherFish    = np.empty(shape=(N,N, 3 ), dtype=float)
distances              = np.empty(shape=(N,N),    dtype=float)
angles                 = np.empty(shape=(N,N),    dtype=float)

## use numpy broadcasting to compute direction, distance, and angles
directionsOtherFish    = locations[np.newaxis, :, :] - locations[:, np.newaxis, :]
distances     = np.sqrt( np.einsum('ijk,ijk->ij', directionsOtherFish, directionsOtherFish) )
# Filling diagonal to avoid division by 0
np.fill_diagonal( distances, 1.0 )
# normalize direction
normalDirectionsOtherFish = directionsOtherFish / distances[:,:,np.newaxis]

test = normalCurDirections[:,np.newaxis,:]
dotprod = np.einsum( 'ijk, ijk->ij', test, normalDirectionsOtherFish )
# reverse order along the third axis
predetprod = np.flip(normalDirectionsOtherFish,2)
# invert sign of second element along third axis
predetprod = np.einsum('ijk, ijk->ijk',np.array([1.,-1.])[np.newaxis,np.newaxis,:],predetprod)
detprod = np.einsum( 'ijk, ijk->ij', normalCurDirections[:,np.newaxis,:], predetprod)
angles = np.arctan2(detprod, dotprod)
print(angles*180/np.pi)
print(dotprod)