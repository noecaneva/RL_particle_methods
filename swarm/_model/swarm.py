import random
import numpy as np
from itertools import product 
import time
import math

from fish import *

class swarm:
    def __init__(self, N, numNN, numdimensions, movementType, initType, _psi=-1,
    _nu = 1.,seed=43, _rRepulsion = 0.6542305401553551, _delrOrientation=2.0085540025782747, _delrAttraction=15.891970757799829, 
    _alpha=4.4868223590698655, _initcircle = +7.0, _f=0.3, _height= +3., _emptzcofactor=+0.5):
        random.seed(seed)
        self.seed=seed
        #number of dimensions of the swarm
        self.dim = numdimensions
        # number of fish
        self.N = N
        # number of nearest neighbours
        self.numNearestNeighbours = numNN
        # type of movement the fish follow
        self.movType  = movementType
        self.rRepulsion = _rRepulsion
        self.rOrientation = _rRepulsion+_delrOrientation
        self.rAttraction = _rRepulsion+_delrOrientation+_delrAttraction
        self.alpha = _alpha
        self.initializationType = initType
        self.initialCircle = _initcircle
        # Maximal number of initializations
        self.tooManyInits=False
        self.maxInits=5000
        self.speed=3.
        self.angularMoments = []
        self.polarizations = []
        # In case we have a cylinder we want to control its height
        self.height = _height
        #extra parameter to control polarization see Gautrais et al. "Initial polarization"
        self.psi = _psi
        # parameter to weigh how important the attraction is over the orientation
        self.nu = _nu
        # Circle parameter as described in Gautrais et al. page 420 top right
        self.f = _f
        # self.initialCircle=np.cbrt(self.N)*self.f*self.rAttraction
        # In case of a ring disposition what % of the initial
        # circle shuld be empty
        self.emptycorecofactor = _emptzcofactor
        self.emptyray = self.initialCircle * self.emptycorecofactor
        lonefish = True
        trycounter = 0
        while(lonefish):
            # placement on a grid
            if(self.initializationType == 0):
                self.fishes = self.randomPlacementNoOverlap(seed)
            elif (self.initializationType in np.array([1, 2])):
                if(self.dim == 2):
                    #self.fishes = self.initOnRing()
                    self.fishes = self.initInSphereorCyl()
                elif(self.dim == 3):
                    self.fishes = self.initInSphereorCyl()
            else:
                print("Unknown initialization type, please choose a number between 0 and 2")
                exit(0)
            
            lonefish = self.noperceivefishinit(self.fishes)
            trycounter += 1 
            # print("number of initializations: ", trycounter)
            if(trycounter == self.maxInits):
                print("over ", trycounter, " initializations")
                self.printstate()
                self.tooManyInits=True
                lonefish = False
        self.angularMoments.append(self.computeAngularMom())
        self.polarizations.append(self.computePolarisation())
        


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
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim, self.psi)

        if(self.dim == 3):
            for i in range(self.N):
                location = np.array([perm[i][0]*dl, perm[i][1]*dl, perm[i][2]*dl]) - L/2
                initdirect=reffish.randUnitDirection()
                fishes[i] = fish(location, initdirect, self.dim, self.psi, speed=self.speed)
        if(self.dim == 2):
            for i in range(self.N):
                location = np.array([perm[i][0]*dl, perm[i][1]*dl]) - L/2
                initdirect=reffish.randUnitDirection()
                fishes[i] = fish(location, initdirect, self.dim, self.psi, speed=self.speed)
        
        # return array of fish
        return fishes

    # different explanations of how to generate random unit distr https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
    # on the unit speher$
    """Generate N random uniform points within a sphere"""
    def initInSphereorCyl(self):
        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim, self.psi)
        fishes = np.empty(shape=(self.N, ), dtype=fish)
        # based on https://stackoverflow.com/questions/9048095/create-random-number-within-an-annulus
        # Normalizing costant
        r_max = self.initialCircle
        normFac = 1./(r_max*r_max)

        if self.initializationType==1:
            for i in range(self.N):
                
                initdirect= reffish.randUnitDirection() #vec/np.linalg.norm(vec)
                if(self.dim == 2):
                    r = np.sqrt(random.uniform(0,1)/normFac)
                    theta = np.random.uniform() * 2 * np.pi
                    vec = np.array([r * np.cos(theta), r * np.sin(theta)])
 
                if(self.dim == 3):
                    phi = random.uniform(0,2*np.pi)
                    costheta = random.uniform(-1,1)
                    u = random.uniform(0,1)
                    theta = np.arccos( costheta )
                    r = r_max * np.cbrt(u)
                    
                    x = r * np.sin( theta ) * np.cos( phi )
                    y = r * np.sin( theta ) * np.sin( phi )
                    z = r * np.cos( theta )
                    vec = np.array([x, y, z])

                fishes[i] = fish(vec, initdirect, self.dim, self.psi, speed=self.speed)
        
        if self.initializationType==2:
            for k in range(self.N):
                r = np.sqrt(2*random.uniform(0,1)/normFac + r_min*r_min)
                theta = np.random.uniform() * 2 * np.pi
                z = np.random.uniform(-height/2, +height/2)
                location = np.array([r * np.cos(theta), r * np.sin(theta), z])
                projxy = np.array([location[0],location[1],0])
                initdirect = projxy/np.linalg.norm(projxy)
                initdirect = reffish.applyrotation(initdirect, np.pi/2, twodproj=True)
                fishes[k] = fish(location, initdirect, self.dim, self.psi, speed=self.speed)

        return fishes

    '''sample uniform random points within a ring with an empty core'''
    def initOnRing(self):
        assert self.dim == 2, print("This function should only be used in 2 dimensions")
        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim, self.psi)

        # based on https://stackoverflow.com/questions/9048095/create-random-number-within-an-annulus
        # Normalizing costant
        r_max = self.initialCircle
        r_min = self.emptyray
        normFac = 2./(r_max*r_max - r_min*r_min)

        fishes = np.empty(shape=(self.N, ), dtype=fish)

        if self.initializationType==1 :
            for k in range(self.N):
                r = np.sqrt(2*random.uniform(0,1)/normFac + r_min*r_min)
                theta = np.random.uniform() * 2 * np.pi
                location = np.array([r * np.cos(theta), r * np.sin(theta)])
                initdirect=reffish.randUnitDirection() #location/np.linalg.norm(location)
                fishes[k] = fish(location, initdirect, self.dim, self.psi, speed=self.speed)

        elif self.initializationType==2:
            for k in range(self.N):
                r = np.sqrt(2*random.uniform(0,1)/normFac + r_min*r_min)
                theta = np.random.uniform() * 2 * np.pi
                location = np.array([r*np.cos(theta), r*np.sin(theta)])
                initdirect = location/np.linalg.norm(location)
                initdirect = reffish.applyrotation(initdirect, np.pi/2)
                fishes[k] = fish(location, initdirect, self.dim, self.psi, speed=self.speed)
        return fishes

    """Boolean function that checks that in the fishes list all fishes perceive at least one other fish"""
    def noperceivefishinit(self, fishes):
        for i, fish in enumerate(fishes):
            directions, distances, angles, cutOff = self.retpreComputeStates(fishes)
            repellTargets, orientTargets, attractTargets = self.retturnrep_or_att(i, fish, angles, distances)

            # Check if the the repellTargets, orientTargets, attractTargets are empty
            if(not any(repellTargets) and not any(orientTargets) and not any(attractTargets)):
                return True

        return False

    """ compute and return distance, direction and angle matrix """
    def retpreComputeStates(self, fishes):
        ## create containers for location, swimming directions, and 
        locations     = np.empty(shape=(self.N, self.dim ), dtype=float)
        curDirections = np.empty(shape=(self.N, self.dim ), dtype=float)
        cutOff        = np.empty(shape=(self.N, ),  dtype=float)

        ## fill matrix with locations / current swimming direction
        for i,fish in enumerate(fishes):
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
        
        #print(normalDirections.shape)
        #np.fill_diagonal( normalDirections, 1.0 )
        
        angles = np.arccos( np.einsum( 'ijk, ijk->ij', normalCurDirections[:,np.newaxis,:], normalDirections ) )
        
        ## set diagonals entries
        np.fill_diagonal( distances, np.inf )
        np.fill_diagonal( angles,    np.inf )

        return directions, distances, angles, cutOff

    """ compute distance and angle matrix """
    def preComputeStates(self):
        ## fill values to class member variable
        self.directionMat,  self.distancesMat, self.anglesMat, cutOff = self.retpreComputeStates(self.fishes)
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

    """for fish i returns the repell, orient and attractTargets"""
    def retturnrep_or_att(self, i, fish, anglesMat, distancesMat):
        deviation = anglesMat[i,:]
        distances = distancesMat[i,:]
        visible = abs(deviation) <= ( self.alpha / 2. ) # check if the angle is within the visible range alpha

        rRepell  = self.rRepulsion   * ( 1 + fish.epsRepell  )
        rOrient  = self.rOrientation * ( 1 + fish.epsOrient  )
        rAttract = self.rAttraction  * ( 1 + fish.epsAttract )

        repellTargets  = self.fishes[(distances < rRepell)]
        orientTargets  = self.fishes[(distances >= rRepell) & (distances < rOrient) & visible]
        attractTargets = self.fishes[(distances >= rOrient) & (distances <= rAttract) & visible]

        return repellTargets, orientTargets, attractTargets


    # Careful assumes that precomputestates has already been called.
    ''' according to https://doi.org/10.1006/jtbi.2002.3065 and/or https://hal.archives-ouvertes.fr/hal-00167590 '''
    def move_calc(self):
        for i,fish in enumerate(self.fishes):
            repellTargets, orientTargets, attractTargets = self.retturnrep_or_att(i, fish, self.anglesMat, self.distancesMat)
            self.fishes[i].computeDirection(repellTargets, orientTargets, attractTargets, self.nu)
        self.angularMoments.append(self.computeAngularMom())
        self.polarizations.append(self.computePolarisation())

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

    '''utility to compute average distance to center of the fishes'''
    def computeAvgDistCenter(self, center):
        # center = self.computeCenter()
        avg = 0.
        for fish in self.fishes:
            avg += np.linalg.norm(fish.location - center)
        avg /= self.N
        return avg


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

    def printstate(self):
        # N, numNN, numdimensions, movementType, initType, _psi, seed=43, _rRepulsion = 0.1, _delrOrientation=1.5, _delrAttraction=3, _alpha=1.5*np.pi, _initcircle = 1.
        print("N :", self.N)
        print("numNN :", self.numNearestNeighbours)
        print("numdimensions :", self.dim)
        print("initType :", self.initializationType)
        print("psi :", self.psi)
        print("seed :", self.seed)
        print("rRepulsion :", self.rRepulsion)
        print("rOrientation :", self.rOrientation)
        print("rAttraction :", self.rAttraction)
        print("alpha :", self.alpha)
        print("initcircle :", self.initialCircle)

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
