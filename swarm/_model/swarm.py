import random
import numpy as np
from itertools import product 
import time
import math

from fish import *

class swarm:
    def __init__(self, N, numNN, numdimensions, movementType, seed=42, _rRepulsion = 0.8, _rOrientation=1.5, _rAttraction=3, _alpha=2):
        #number of dimensions of the swarm
        self.dim = numdimensions
        # number of fish
        self.N = N
        # number of nearest neighbours
        self.numNearestNeighbours = numNN
        # type of movement the fish follow
        self.movType  = movementType
        self.rRepulsion = _rRepulsion
        self.rOrientation = _rOrientation
        self.rAttraction = _rAttraction
        self.alpha = _alpha
        # create fish at random locations
        # self.fishes = self.randomPlacementNoOverlap( seed )
        if(self.dim == 2):
            self.fishes = self.place_on_circle( self.rAttraction)
        elif(self.dim == 3):
            self.fishes = self.place_on_sphere(2.)

    """ random placement on a grid """
    # NOTE the other two papers never start on grids but they start on sphere like structures
    def randomPlacementNoOverlap(self, seed):
        # number of gridpoints per dimension for initial placement
        M = int( pow( self.N, 1/self.dim ) )
        V = M+1
        # grid spacing ~ min distance between fish
        # NOTE this 0.7 comes from a few test Daniel run
        dl = 0.7
        # maximal extent
        L = V*dl
        
        # generate random permutation of [1,..,V]x[1,..,V]x[1,..,V]
        perm = list(product(np.arange(0,V),repeat=self.dim))
        assert self.N < len(perm), "More vertices required to generate random placement"
        random.Random( seed ).shuffle(perm)
    
        # place fish
        fishes = np.empty(shape=(self.N,), dtype=fish)
        location = np.zeros(shape=(self.N,self.dim), dtype=float)

        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim)

        if(self.dim == 3):
            for i in range(self.N):
                location = np.array([perm[i][0]*dl, perm[i][1]*dl, perm[i][2]*dl]) - L/2
                initdirect=reffish.randUnitDirection()
                fishes[i] = fish(location, initdirect, self.dim)
        if(self.dim == 2):
            for i in range(self.N):
                location = np.array([perm[i][0]*dl, perm[i][1]*dl]) - L/2
                initdirect=reffish.randUnitDirection()
                fishes[i] = fish(location, initdirect, self.dim)
        
        # return array of fish
        return fishes

    def place_on_circle(self, circleRay):
        assert self.dim == 2, print("This function should only be used in 2 dimensions")

        fishes = np.empty(shape=(self.N,), dtype=fish)
        location = np.zeros(shape=(self.N,self.dim), dtype=float)
        
        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim)

        delalpha = 360./self.N
        for i in range(self.N):
            location = np.array([circleRay*np.cos(delalpha*i), circleRay*np.sin(delalpha*i)])
            initdirect=reffish.randUnitDirection()
            fishes[i] = fish(location, initdirect, self.dim)
        
        return fishes

    def place_on_sphere(self, raySphere):
        assert self.dim == 3, print("This function should only be used in 3 dimensions")

        fishes = np.empty(shape=(self.N,), dtype=fish)
        location = np.zeros(shape=(self.N,self.dim), dtype=float)
        
        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim)

        # # placement according to https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf

        # radtodeg =180/np.pi

        # N_count = 0
        # a = 4*np.pi*raySphere*raySphere/self.N
        # d = np.sqrt(a)
        # M_theta = math.ceil(np.pi/d)
        # d_theta = np.pi/M_theta
        # d_phi = a/d_theta
        # print("M_theta is", M_theta)
        # for m in range(M_theta):
        #     theta = np.pi*(m + 0.5)/M_theta
        #     M_phi = math.ceil(2*np.pi*np.sin(theta)/d_phi)
        #     print("M_phi is", M_phi)
        #     for n in range(M_phi):
        #         phi = 2 * np.pi * n /M_phi
        #         initdirect=reffish.randUnitDirection()
        #         location = np.array([raySphere*np.sin(theta*radtodeg)*np.cos(phi*radtodeg),raySphere*np.sin(theta*radtodeg)*np.sin(phi*radtodeg),raySphere*np.cos(theta*radtodeg)])

        #         if(N_count == self.N):
        #             break
        #         fishes[N_count] = fish(location, initdirect, self.dim)

        #         N_count += 1
        # print("Ncount is  ", N_count)
        # print("self.N is ", self.N)

        #placement according to  https://medium.com/@vagnerseibert/distributing-points-on-a-sphere-6b593cc05b42
        
        radtodeg =180/np.pi

        location = []

        for i in range(self.N):
            k = i + 0.5

            phi = np.cos((1. - 2. * k / self.N))
            theta = np.pi * (1 + np.sqrt(5)) * k

            x = raySphere * np.cos(theta) * np.sin(phi)
            y = raySphere * np.sin(theta) * np.sin(phi)
            z = raySphere * np.cos(phi)

            location = np.array([x,y,z])
            initdirect= location/np.linalg.norm(location) #reffish.randUnitDirection()

            fishes[i] = fish(location, initdirect, self.dim)

        return fishes

    """ compute distance and angle matrix (very slow version) """
    def preComputeStatesNaive(self):
        # create containers for distances, angles and directions
        distances  = np.full( shape=(self.N,self.N), fill_value=np.inf, dtype=float)
        angles     = np.full( shape=(self.N,self.N), fill_value=np.inf, dtype=float)
        directions = np.zeros(shape=(self.N,self.N,self.dim), dtype=float)
        # boolean indicating if two fish are touching
        terminal = False
        # iterate over grid and compute angle / distance matrix
        for i in np.arange(self.N):
            for j in np.arange(self.N):
                if i != j:
                    # direction vector and current direction
                    u = self.fishes[j].location - self.fishes[i].location
                    v = self.fishes[i].curDirection
                    # set distance
                    distances[i,j] = np.linalg.norm(u)
                    # set direction
                    directions[i,j,:] = u
                    # set angle
                    cosAngle = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
                    angles[i,j] = np.arccos(cosAngle)
            # Termination state in case distance matrix has entries < cutoff
            if (distances[i,:] < self.fishes[i].sigmaPotential ).any():
                terminal = True

        self.distancesMat = distances
        self.anglesMat    = angles
        self.directionMat = directions

        return terminal

    """ compute distance and angle matrix """
    def preComputeStates(self):
        ## create containers for location, swimming directions, and 
        locations     = np.empty(shape=(self.N, self.dim ), dtype=float)
        curDirections = np.empty(shape=(self.N, self.dim ), dtype=float)
        cutOff        = np.empty(shape=(self.N, ),  dtype=float)

        ## fill matrix with locations / current swimming direction
        for i,fish in enumerate(self.fishes):
            locations[i,:]     = fish.location
            curDirections[i,:] = fish.curDirection
            cutOff[i]          = fish.sigmaPotential
        # NOTE the swimming directions (but not the whished direction) are normalized here
        # normalize swimming directions
        normalCurDirections = curDirections / np.linalg.norm( curDirections, axis=1 )[:, np.newaxis]

        ## create containers for direction, distance, and angle
        directions    = np.empty(shape=(self.N,self.N, self.dim ), dtype=float)
        distances     = np.empty(shape=(self.N,self.N),    dtype=float)
        angles        = np.empty(shape=(self.N,self.N),    dtype=float)

        # QUESTION do not understand what is going on here what does the newaxis do. Why is locations suddently with
        # 3 columns
        ## use numpy broadcasting to compute direction, distance, and angles
        directions    = locations[np.newaxis, :, :] - locations[:, np.newaxis, :]
        distances     = np.sqrt( np.einsum('ijk,ijk->ij', directions, directions) )
        # print(distances)
        # NOTE directions get normalized here
        # normalize direction
        normalDirections = directions / distances[:,:,np.newaxis]
        angles = np.arccos( np.einsum( 'ijk, ijk->ij', normalCurDirections[:,np.newaxis,:], normalDirections ) )
        
        ## set diagonals entries
        np.fill_diagonal( distances, np.inf )
        np.fill_diagonal( angles,    np.inf )

        ## fill values to class member variable
        self.directionMat = directions
        self.distancesMat = distances
        self.anglesMat    = angles
        
        # Note if this boolean returns true the simulation is stopped
        # return if any two fish are closer then the cutOff
        return ( self.distancesMat < cutOff[:,np.newaxis] ).any()

    def getState( self, i ):
        # get array for agent i
        distances = self.distancesMat[i,:]
        angles    = self.anglesMat[i,:]
        directions= self.directionMat[i,:,:]
        # sort and select nearest neighbours
        idSorted = np.argsort( distances )
        idNearestNeighbours = idSorted[:self.numNearestNeighbours]
        self.distancesNearestNeighbours = distances[ idNearestNeighbours ]
        self.anglesNearestNeighbours    = angles[ idNearestNeighbours ]
        self.directionNearestNeighbours = directions[idNearestNeighbours,:]
        # the state is the distance (or direction?) and angle to the nearest neigbours
        return np.array([ self.distancesNearestNeighbours, self.anglesNearestNeighbours ]).flatten().tolist() # or np.array([ directionNearestNeighbours, anglesNearestNeighbours ]).flatten()

    def getReward( self, i ):
        # Careful: assumes sim.getState(i) was called before
        return self.fishes[i].computeReward( self.distancesNearestNeighbours )


    # Careful assumes that precomputestates has already been called.
    ''' according to https://doi.org/10.1006/jtbi.2002.3065 and/or https://hal.archives-ouvertes.fr/hal-00167590 '''
    def move_calc(self):
        for i,fish in enumerate(self.fishes):
            deviation = self.anglesMat[i,:]
            distances = self.distancesMat[i,:]
            visible = abs(deviation) <= ( self.alpha / 2. ) # check if the angle is within the visible range alpha

            rRepell  = self.rRepulsion   * ( 1 + fish.epsRepell  )
            rOrient  = self.rOrientation * ( 1 + fish.epsOrient  )
            rAttract = self.rAttraction  * ( 1 + fish.epsAttract )

            repellTargets  = self.fishes[(distances < rRepell)]
            orientTargets  = self.fishes[(distances >= rRepell) & (distances < rOrient) & visible]
            attractTargets = self.fishes[(distances >= rOrient) & (distances <= rAttract) & visible]
            self.fishes[i].computeDirection(repellTargets, orientTargets, attractTargets)

    ''' utility to compute polarisation (~alignement) '''
    def computePolarisation(self):
        polarisationVec = np.zeros(shape=(self.dim,), dtype=float)
        for fish in self.fishes:
            polarisationVec += fish.curDirection
        polarisation = np.linalg.norm(polarisationVec) / self.N
        return polarisation

    ''' utility to compute center of swarm '''
    def computeCenter(self):
        center = np.zeros(shape=(self.dim,), dtype=float)
        for fish in self.fishes:
            center += fish.location
        center /= self.N
        return center

    ''' utility to compute angular momentum (~rotation) '''
    def computeAngularMom(self):
        center = self.computeCenter()
        if(self.dim == 3):
            angularMomentumVec = np.zeros(shape=(self.dim,), dtype=float)
            for fish in self.fishes:
                distance = fish.location-center
                distanceNormal = distance / np.linalg.norm(distance) 
                angularMomentumVecSingle = np.cross(distanceNormal,fish.curDirection)
                angularMomentumVec += angularMomentumVecSingle
            angularMomentum = np.linalg.norm(angularMomentumVec) / self.N
        elif(self.dim == 2):
            #in this case the cross product yealds a scalar
            angularMomentumVec = np.zeros(shape=(1,), dtype=float)
            for fish in self.fishes:
                distance = fish.location-center
                distanceNormal = distance / np.linalg.norm(distance) 
                angularMomentumVecSingle = np.cross(distanceNormal,fish.curDirection)
                angularMomentumVec += angularMomentumVecSingle
            angularMomentum = np.linalg.norm(angularMomentumVec) / self.N
        return angularMomentum
