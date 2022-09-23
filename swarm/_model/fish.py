import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import vonmises
from scipy.stats import truncnorm

from plotter2D import *
from plotter3D import *

# parameters for truncated gaussians (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html), taken from https://www.sciencedirect.com/science/article/pii/0304380094900132 
observedMean  = np.array([ 0.7, 1.0, 1.3 ])
observedSigma = np.array([ 0.3, 0.4, 0.5 ])
lowerBound = 0
upperBound = np.inf
observedA = (lowerBound-observedMean)/observedSigma
observedB = (upperBound-observedMean)/observedSigma


class fish:
    def __init__(self, location, numdimensions, individualStd=0.05, speed=1, maxAngle=30./180.*np.pi, eqDistance=0.1, potentialStrength=100, potential="Observed" ):
        self.dim = numdimensions
        self.location = location
        self.curDirection = self.randUnitDirection()
        self.wishedDirection = self.curDirection

        # individual variation
        self.individualStd = individualStd
        individualNoise = np.zeros(4) #np.random.normal(0.0, self.individualStd, 4)
        # motion parameters
        self.speed       = speed    * ( 1 + individualNoise[0] )
        self.maxAngle    = maxAngle * ( 1 + individualNoise[1] ) #is this correct? TODO
        self.dt          = 0.1
        self.sigmaMotion = 0.1 #what is sigmamoition QUESTION
        # potential
        self.potential = potential
        ## parameters for potentials
        # max value of reward
        self.epsilon        = potentialStrength * ( 1 + individualNoise[2] ) #what is epsilon QUESTION
        # distance below which reward becomes penality
        self.sigmaPotential = eqDistance        * ( 1 + individualNoise[3] ) #what is sigmapotential QUESTION

    ''' get uniform random unit vector on sphere '''      
    def randUnitDirection(self):
        vec = np.random.normal(0.,1.,self.dim)
        mag = np.linalg.norm(vec)
        return vec/mag

    ''' according to https://doi.org/10.1006/jtbi.2002.3065 and/or https://hal.archives-ouvertes.fr/hal-00167590 '''
    def computeDirection(self, repellTargets, orientTargets, attractTargets):
        newWishedDirection = np.zeros(self.dim)
        # zone of repulsion - highest priority
        if repellTargets.size > 0:
            for fish in repellTargets:
                diff = fish.location - self.location
                # QUESTION where does speed*dt<=rRepulsion come from?
                assert np.linalg.norm(diff) > 1e-12, print(diff, "are you satisfying speed*dt<=rRepulsion?")
                assert np.linalg.norm(diff) < 1e12,  print(diff)
                newWishedDirection -= diff/np.linalg.norm(diff)
        else:
            orientDirect = np.zeros(self.dim)
            attractDirect = np.zeros(self.dim)
            # zone of orientation
            if orientTargets.size > 0:
              for fish in orientTargets:
                  orientDirect += fish.curDirection/np.linalg.norm(fish.curDirection)
            # zone of attraction
            if attractTargets.size > 0:
              for fish in attractTargets:
                  diff = fish.location - self.location
                  attractDirect += diff/np.linalg.norm(diff)
            
            # NOTE control if the magnitude does not matter of whisheddirection
            newWishedDirection = orientDirect+attractDirect
        
        # QUESTION where does this 1e-12 come from?
        if np.linalg.norm(newWishedDirection) < 1e-12:
          newWishedDirection = self.curDirection
        
        ## NOTE here the stocastic rotation of the direction of the fish gets calculated and applied
        # In the gautrais paper a rotational diffusion coefficent is introduced in order to make sure
        # that the angular stochastic deviation stays below the maximally permitted angle in turning 
        # time sense.
        # QUESTION I want to make the whole thing so that we can just change a parameter d that makes it work in 3D
        # and in 2D. Here is it bad for performance if we get an if/else branch.
        # also generally what happens here exactly? how does it work?
        ## stochastic effect, replicates "spherically wrapped Gaussian distribution"
        # get random unit direction orthogonal to newWishedDirection
        # compute random angle from wrapped Gaussian ~ van Mises distribution
        print("Do we actually come here?______________________________________________________________________")
        randAngle = vonmises.rvs(1/self.sigma**2)
        self.wishedDirection  = self.applyrotation(newWishedDirection, randAngle)
        print("During computeDirection whisheddir")
        print(len(self.wishedDirection))



    ''' rotate direction of the swimmer ''' 
    def updateDirection(self):
        u = self.curDirection
        v = self.wishedDirection
        assert np.isclose( np.linalg.norm(u), 1.0 ), "Current direction {} not normalized".format(u)
        # Here we control that the wished direction is normalized so we have to have it normalized somewhere
        assert np.isclose( np.linalg.norm(v), 1.0 ), "Wished direction {} not normalized".format(v)

        # numerical safe computation of cos and angle
        cosAngle = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
        # values outside get projected onto the edges
        cosAngle = np.clip(cosAngle, -1, 1)
        angle    = np.arccos(cosAngle)
        # Maxangle is the max rotation that can be done in the timestep dt. In our case we fix it in the beginning so
        # there might be an issue
        if angle < self.maxAngle:
            self.curDirection = self.wishedDirection
        # handle antiparallel case
        # this means that u is in the opposite direction of v.
        elif np.isclose(angle, np.pi):
            self.curDirection = self.applyrotation(self.curDirection, self.maxAngle)
        else:
            # Why not use u and v here for the cross?
            # QUESTION how do we know we are rotating in the correct direction? Do I need to keep the assert in line 148
            self.curDirection = self.applyrotation_2vec(self.curDirection, self.wishedDirection, self.maxAngle,  cosAngle)
        
        # normalize
        self.curDirection /= np.linalg.norm(self.curDirection)
        # NOTE only curdirection gets normalized, whisheddirection does not

    ''' update the direction according to x += vt ''' 
    def updateLocation(self):
        self.location += self.speed*self.dt*self.curDirection

    ''' reward assumes pair-wise potentials ''' 
    def computeReward(self, nearestNeighbourDistance ):
        reward = 0.0
        for i,r in enumerate(nearestNeighbourDistance):
            # Lennard-Jones potential
            if self.potential == "Lennard-Jones":
                x = self.sigmaPotential / r
                reward -= 4*self.epsilon*( x**12 - x**6 )
            # Harmonic potential
            elif self.potential == "Harmonic":
                reward += self.epsilon - 4*self.epsilon/self.sigmaPotential**2*(156/2**(7/3)-42/2**(4/3))*(r-2**(1/6)*self.sigmaPotential)**2
            # Observations (https://www.sciencedirect.com/science/article/pii/0304380094900132)
            elif self.potential == "Observed":
                if i>2:
                    assert 0, print("The 'Observed' reward only supports up to 3 nearest Neighbours")
                # rTest = np.linspace(-10,10,1001)
                # plt.plot(rTest, truncnorm.pdf(rTest, a=observedA[i], b=observedB[i], loc=observedMean[i], scale=observedSigma[i]))
                reward += truncnorm.pdf(r, a=observedA[i], b=observedB[i], loc=observedMean[i], scale=observedSigma[i])
            else:
                assert 0, print("Please chose a pair-potential that is implemented")
        # plt.show()
        # print(nearestNeighbourDistance, reward)
        return reward

    ''' newton policy computes direction as gradient of potential ''' 
    def newtonPolicy(self, nearestNeighbourDirections ):
        action = np.zeros(self.dim)
        for direction in nearestNeighbourDirections:
            r = np.linalg.norm(direction)
            # Lennard-Jones potential
            if self.potential == "Lennard-Jones":
                x = self.sigmaPotential / r
                action -= 4*self.epsilon*( -12*x**12/r + 6*x**6/r )*direction/r
            # Harmonic potential
            elif self.potential == "Harmonic":
                action += 4*self.epsilon/self.sigmaPotential**2*(156/2**(7/3)-42/2**(4/3))*(r-2**(1/6)*self.sigmaPotential)*direction/r
            elif self.potential == "Observed":
                assert 0, print("please do first implement the policy for the 'Observed' reward")
            else:
                assert 0, print("Please chose a pair-potential that is implemented")
        action = action / np.linalg.norm(action)
        return action

    ''' general calculation in order to apply a rotation to a vector returns the rotated vector'''
    def applyrotation(self, vectortoapply, angletoapply):
        if(self.dim == 3):
            randVector = self.randUnitDirection()
            rotVector = np.cross(vectortoapply,randVector)
            while np.isclose(np.linalg.norm(rotVector), 0.0):
                randVector = self.randUnitDirection()
                rotVector = np.cross(vectortoapply,randVector)
            rotVector /= np.linalg.norm(rotVector)
            # rotvector is orthogonal to the random and the newWishedvector

            # create rotation
            rotVector *= angletoapply
            r = Rotation.from_rotvec(rotVector)
            # return a vector on which the rotation has been applied
            return r.apply(vectortoapply)

        elif(self.dim == 2):
            # In this case to make the rotation work we pad a zero rotate and than extract
            # the first two values in the end
            rotVector = np.array([0., 0., 1.])
            # compute random angle from wrapped Gaussian ~ van Mises distribution

            # create rotation
            rotVector *= angletoapply
            r = Rotation.from_rotvec(rotVector)
            # apply rotation to padded wisheddirection
            exp_newwishedir = np.pad(vectortoapply, (0, 1), 'constant')
            exp_wisheddir = r.apply(exp_newwishedir)
            return exp_wisheddir[:2]

    ''' apply a rotation to a vector to turn it by maxangle into the direction of the second vectorreturns the rotated vector'''
    def applyrotation_2vec(self, vectortoapply, vector_final, angletoapply, cosAngle):
        if(self.dim == 3):
            rotVector = np.cross(vectortoapply,vector_final)
            assert np.linalg.norm(rotVector) > 0, "Rotation vector {} from current {} and wished direction {} with angle {} is zero".format(rotVector, self.curDirection, self.wishedDirection, cosAngle)
            rotVector /= np.linalg.norm(rotVector)
            rotVector *= angletoapply
            r = Rotation.from_rotvec(rotVector)
            return r.apply(self.curDirection)

        elif(self.dim == 2):
            # In this case to make the rotation work we pad the 2 vectors with a 0 in z and then do exactly the same
            # at the end though we'll only take the first 2 entries
            # the first two values in the end
            rotVector = np.cross(np.pad(vector_final, (0, 1), 'constant'),np.pad(vector_final, (0, 1), 'constant'))
            assert np.linalg.norm(rotVector) > 0, "Rotation vector {} from current {} and wished direction {} with angle {} is zero".format(rotVector, self.curDirection, self.wishedDirection, cosAngle)
            rotVector /= np.linalg.norm(rotVector)
            rotVector *= angletoapply
            r = Rotation.from_rotvec(rotVector)
            return r.apply(self.curDirection)[:2]