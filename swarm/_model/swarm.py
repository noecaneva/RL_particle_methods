import random
import numpy as np
from itertools import product 
import time
import math

from fish import *

class swarm:
    def __init__(self, N, numNN, numdimensions, movementType, initType, _psi,
    _nu = 1.,seed=43, _rRepulsion = +8.185e-01, _delrOrientation=+7.697e+00, _delrAttraction=+1.523e+01, 
    _alpha=1.5*np.pi, _initcircle = 1., _f=+4.565e-01):
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
        #extra parameter to control polarization see Gautrais et al. "Initial polarization"
        self.psi = _psi
        # parameter to weigh how important the attraction is over the orientation
        self.nu = _nu
        # Circle parameter as described in Gautrais et al. page 420 top right
        self.f = _f
        # self.initialCircle=np.cbrt(self.N)*self.f*self.rAttraction
        lonefish = True
        trycounter = 0
        while(lonefish):
            # placement on a grid
            if(self.initializationType == 0):
                self.fishes = self.randomPlacementNoOverlap(seed)

            # Placement on boarder of circle or a sphere
            elif(self.initializationType == 1):
                    if(self.dim == 2):
                        self.fishes = self.milling_on_circle(self.initialCircle)
                    elif(self.dim == 3):
                        self.fishes = self.place_on_sphere(self.initialCircle)

            #Placement within a circle or a sphere
            elif(self.initializationType == 2):
                if(self.dim == 2):
                    self.fishes = self.randInCircle(self.initialCircle)
                elif(self.dim == 3):
                    self.fishes = self.initInSphere(self.initialCircle)     
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

    """ random placement on a circle"""
    def place_on_circle(self, circleRay):
        assert self.dim == 2, print("This function should only be used in 2 dimensions")

        fishes = np.empty(shape=(self.N,), dtype=fish)
        location = np.zeros(shape=(self.N,self.dim), dtype=float)
        
        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim, self.psi)

        delalpha = 2*np.pi/self.N
        for i in range(self.N):
            location = np.array([circleRay*np.cos(delalpha*i), circleRay*np.sin(delalpha*i)])
            initdirect=reffish.randUnitDirection()
            initdirect = location/np.linalg.norm(location)
            fishes[i] = fish(location, initdirect, self.dim, self.psi, speed=self.speed)
        
        return fishes

    """ enforce milling behaviour """
    def milling_on_circle(self, circleRay):
        assert self.dim == 2, print("This function should only be used in 2 dimensions")

        fishes = np.empty(shape=(self.N,), dtype=fish)
        location = np.zeros(shape=(self.N,self.dim), dtype=float)
        
        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim, self.psi)

        delalpha = 2*np.pi/self.N
        for i in range(self.N):
            location = np.array([circleRay*np.cos(delalpha*i), circleRay*np.sin(delalpha*i)])
            initdirect=reffish.randUnitDirection()
            initdirect = location/np.linalg.norm(location)
            initdirect = reffish.applyrotation(initdirect, np.pi/2)
            fishes[i] = fish(location, initdirect, self.dim, self.psi, speed=self.speed)
        
        return fishes

    """ random placement on a sphere, somethimes the number of points placed is inferior to what is given"""
    def place_on_sphere(self, raySphere):
        assert self.dim == 3, print("This function should only be used in 3 dimensions")

        fishes = np.empty(shape=(self.N,), dtype=fish)
        location = np.zeros(shape=(self.N,self.dim), dtype=float)
        
        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim, self.psi)

        # placement according to https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf

        N_count = 0
        a = 4*np.pi/self.N
        d = np.sqrt(a)
        M_theta = math.ceil(np.pi/d)
        d_theta = np.pi/M_theta
        d_phi = a/d_theta
        #print("M_theta is", M_theta)
        for m in range(M_theta):
            theta = np.pi*(m + 0.5)/M_theta
            M_phi = math.ceil(2*np.pi*np.sin(theta)/d_phi)
            #print("M_phi is", M_phi)
            for n in range(M_phi):
                phi = 2 * np.pi * n /M_phi
                initdirect=reffish.randUnitDirection()
                x = raySphere*np.sin(theta)*np.cos(phi)
                y = raySphere*np.sin(theta)*np.sin(phi)
                z = raySphere*np.cos(theta)

                location = np.array([x, y, z])

                if(N_count == self.N):
                    break
                fishes[N_count] = fish(location, initdirect, self.dim, self.psi, speed=self.speed)

                N_count += 1
        
        print("Ncount is  ", N_count)
        print("self.N is ", self.N)
        self.N = N_count
        print("self.N was changed to N_count")

        # Alternative solution can be found in https://medium.com/@vagnerseibert/distributing-points-on-a-sphere-6b593cc05b42
    
        return fishes[:N_count]

    """ random placement within a circle"""
    #taken from https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle
    def radius(self, k, n, b):
        if k > n - b:
            return 1.0
        else:
            return np.sqrt(k - 0.5) / np.sqrt(n - (b + 1) / 2)

    # More or less equidistant points within a sphere
    def sunflower(self, raySphere, alpha=0, geodesic=False):

        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim, self.psi)

        fishes = np.empty(shape=(self.N, ), dtype=fish)
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        angle_stride = 2 * np.pi * phi if geodesic else 2 * np.pi / phi ** 2
        b = round(alpha * np.sqrt(self.N))  # number of boundary points

        for k in range(1, self.N + 1):
            r = raySphere*self.radius(k, self.N, b)
            theta = k * angle_stride
            location = np.array([r * np.cos(theta), r * np.sin(theta)])
            initdirect=reffish.randUnitDirection() #location/np.linalg.norm(location)
            fishes[k-1] = fish(location, initdirect, self.dim, self.psi, speed=self.speed)

        return fishes
    
    # sample uniform random points within a circle
    def randInCircle(self, raySphere):
         # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim, self.psi)

        fishes = np.empty(shape=(self.N, ), dtype=fish)

        for k in range(self.N):
            r = raySphere*np.sqrt(np.random.uniform())
            theta = np.random.uniform() * 2 * np.pi
            location = np.array([r * np.cos(theta), r * np.sin(theta)])
            initdirect=reffish.randUnitDirection() #location/np.linalg.norm(location)
            fishes[k] = fish(location, initdirect, self.dim, self.psi, speed=self.speed)

        return fishes       

    # different explanations of how to generate random unit distr https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
    # on the unit speher$
    """Generate N random uniform points within a sphere"""
    def initInSphere(self, raySphere):

        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim, self.psi)
        fishes = np.empty(shape=(self.N, ), dtype=fish)

        for i in range(self.N):
            u = random.uniform(0, 1)
            vec = np.random.normal(loc=0, scale=1, size=3)
            vec /= np.linalg.norm(vec)
            # np.cbrt is cube root
            c = np.cbrt(u)
            vec *= (c*raySphere)
            initdirect= reffish.randUnitDirection() #vec/np.linalg.norm(vec)
            fishes[i] = fish(vec, initdirect, self.dim, self.psi, speed=self.speed)

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
