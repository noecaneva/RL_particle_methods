import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import vonmises
from scipy.stats import truncnorm
from scipy.optimize import minimize

from plotter2D import *
from plotter3D import *

# parameters for truncated gaussians (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html), taken from https://www.sciencedirect.com/science/article/pii/0304380094900132 
observedMean  = np.array([ 0.7, 1.0, 1.3 ])
observedSigma = np.array([ 0.3, 0.4, 0.5 ])
lowerBound = 0
upperBound = np.inf
observedA = (lowerBound-observedMean)/observedSigma
observedB = (upperBound-observedMean)/observedSigma
np.random.seed(50)

def rotation3D(th, old, new, ax1, ax2):
    r1 = Rotation.from_rotvec(th[0]*ax1)
    r2 = Rotation.from_rotvec(th[1]*ax2)
    tmp = r1.apply(old)
    tmp = r2.apply(tmp)
    return np.linalg.norm(new-tmp)

class fish:
    def __init__(self, location, initialDirection, numdimensions, _psi, individualStd=0.05, speed=3, maxAngle=4./180.*np.pi, eqDistance=0.1, potentialStrength=100, _sigma = 0.8, potential="Observed", randomMov = False ):
        self.dim = numdimensions
        self.location = location
        #self.history = [self.location]
        self.oldDirection = initialDirection
        self.curDirection = initialDirection
        self.wishedDirection = self.curDirection
        self.normalDist=False
        self.randomMov=randomMov
        self.epsRepell=0.0
        self.epsOrient=0.0
        self.epsAttract=0.0
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
        # psi for the initial polarization
        self.psi = _psi
        # Diffusion coefficent to control the distribuiton of the stochastic noise on the
        # direction
        self.D_r = (self.maxAngle*self.speed/1.96)*(self.maxAngle*self.speed/1.96)/(2*self.dt)
        # simga for the normal distribuiton of the angle
        self.sigma = np.sqrt(2.*self.D_r*self.dt)
        self.sigma = 0.05
        self.distanceToNearestNeighbour = []

    ''' get uniform random unit vector on sphere '''
    # psi = -1 means the resulting vector is completely random
    def randUnitDirection(self):
        assert abs(self.psi) <= 1.0, f"psi should be between -1.0 and 1.0 (is {self.psi})"
        if(self.dim == 2):
            vx = np.random.uniform(self.psi, 1.)
            vy = np.sqrt(1-vx*vx) * np.sign(np.random.uniform() - 0.5)
            vec = np.array([vx, vy])

        elif(self.dim == 3):
            vx = np.random.uniform(self.psi, 1.)
            u = np.random.uniform(0, 2*np.pi)
            cofac = np.sqrt(1. - vx*vx)
            vy = cofac *np.sin(u)
            vz = cofac *np.cos(u)
            vec = np.array([vx, vy, vz])

        else:
            print("unknown number of dimensions please choose 2 or 3")
            exit(0)
        return vec/np.linalg.norm(vec) # Normalization necessary


    def setAxis(self):

        if self.dim == 2:
            self.rotAxis1 = np.array([0.,0.,1.])
            self.rotAxis2 = None


        if self.dim == 3:
            tmp = np.zeros(3)
            tmp = self.curDirection[:2]
            rotVector1 = np.cross(self.curDirection, tmp)
            normRot1 = np.linalg.norm(rotVector1)
            if normRot1 > 1e-12:
                rotVector1 /= np.linalg.norm(rotVector1)
            else:
                rotVector1 = np.array([1.,0.,0.])
            
            rotVector2 = np.cross(self.curDirection, rotVector1)
            rotVector2 /= np.linalg.norm(rotVector2)

            self.rotAxis1 = rotVector1
            self.rotAxis2 = rotVector2


    
    ''' according to https://doi.org/10.1006/jtbi.2002.3065 and/or https://hal.archives-ouvertes.fr/hal-00167590 '''
    def computeDirection(self, repellTargets, orientTargets, attractTargets, nu): #TODO: add againomputeDirection(self, repellTargets, orientTargets, attractTargets, nu):
        newWishedDirection = np.zeros(self.dim)
        # zone of repulsion - highest priority
        if repellTargets.size > 0:
            for fish in repellTargets:
                diff = fish.location - self.location
                assert np.linalg.norm(diff) > 1e-12, print(diff, "are you satisfying speed*dt<=rRepulsion?")
                assert np.linalg.norm(diff) < 1e12,  print(diff)
                newWishedDirection -= diff/np.linalg.norm(diff)
            newWishedDirection /= np.linalg.norm(newWishedDirection)
        else:
            orientDirect = np.zeros(self.dim)
            attractDirect = np.zeros(self.dim)
            # zone of orientation
            if orientTargets.size > 0:
                for fish in orientTargets:
                    orientDirect += fish.curDirection/np.linalg.norm(fish.curDirection)
                orientDirect/=np.linalg.norm(orientDirect)
            # zone of attraction
            if attractTargets.size > 0:
                for fish in attractTargets:
                    diff = fish.location - self.location
                    attractDirect += diff/np.linalg.norm(diff)
                attractDirect/=np.linalg.norm(attractDirect)
            # NOTE control if the magnitude does not matter of whisheddirection
            if (orientTargets.size > 0 and attractTargets.size > 0):
                newWishedDirection = 0.5*(orientDirect+nu*attractDirect)
            elif(orientTargets.size > 0 and attractTargets.size == 0):
                newWishedDirection = orientDirect
            elif(orientTargets.size == 0 and attractTargets.size > 0):
                newWishedDirection = nu*attractDirect
            
        
        if np.linalg.norm(newWishedDirection) < 1e-12:
          newWishedDirection = self.curDirection
        
        newWishedDirection /= np.linalg.norm(newWishedDirection)
        
        ## NOTE here the stocastic rotation of the direction of the fish gets calculated and applied
        # In the gautrais paper a rotational diffusion coefficent is introduced in order to make sure
        # that the angular stochastic deviation stays below the maximally permitted angle in turning 
        # time sense.
        ## stochastic effect, replicates "spherically wrapped Gaussian distribution"
        # get random unit direction orthogonal to newWishedDirection
        # compute random angle from wrapped Gaussian ~ van Mises distribution
        if (self.normalDist):
            randAngle = np.random.normal(0., self.sigma, 1, size=self.dim-1)
        else:
            randAngle = vonmises.rvs(1/self.sigma**2, size=self.dim-1)
        self.wishedDirection  = self.applyrotation(newWishedDirection, randAngle) #TODO: add again
        #self.wishedDirection = newWishedDirection #self.applyrotation(newWishedDirection, randAngle)
        # print(len(self.wishedDirection)) this is 2

    def getAction(self):
        assert np.isclose( np.linalg.norm(self.oldDirection), 1.0 ), f"[fish] Current direction {self.oldDirection} not normalized"
        assert np.isclose( np.linalg.norm(self.curDirection), 1.0 ), f"[fish] Old direction {self.curDirection} not normalized"

        if self.dim == 2:
            action = np.zeros(1)
            normal = np.cross(self.oldDirection, self.wishedDirection)
            rotangle = np.clip(np.arccos(np.dot(self.oldDirection, self.wishedDirection)), a_min=-self.maxAngle, a_max=self.maxAngle)
            action[0] = rotangle * np.sign(normal)

        else:
            obj = lambda th : rotation3D(th, self.oldDirection, self.wishedDirection, self.rotAxis1, self.rotAxis2)
            #res = minimize(obj, [0.,0.], method='Nelder-Mead', tol=1e-6)
            res = minimize(obj, [0.,0.], tol=1e-6)
            action = res.x
            action = np.clip(action, a_min=-self.maxAngle, a_max=self.maxAngle)
            #assert res.success == True, f"[fish] Optimization not succesfull {res}"
            #assert res.fun < 1e-3, f"[fish] Action could not be identified {res}"

            """
            r0 = np.linalg.norm(old)
            r1 = np.linalg.norm(newDirection)
            
            th0 = np.arccos(old[2]/r0)
            th1 = np.arccos(newDirection[2]/r1)

            phi0 = np.arctan2(old,old)
            phi1 = np.arctan2(newDirection[1],newDirection[0])

            dth = th1-th0
            dph = phi1-phi0


            dph = dph % (2*np.pi)
            dth = dth % (2*np.pi)
 
            if dth > np.pi:
                dth -= 2*np.pi
            elif dth < -np.pi:
                dth += 2*np.pi
            
            if dph > np.pi:
                dph -= 2*np.pi
            elif dph < -np.pi:
                dph += 2*np.pi
            
            action[0] = dph
            action[1] = dth
            """

        return action


    ''' rotate direction of the swimmer ''' 
    def updateDirection(self):
        u = self.curDirection
        v = self.wishedDirection
        assert np.isclose( np.linalg.norm(u), 1.0 ), "Current direction {} not normalized".format(u)
        if(not np.isclose( np.linalg.norm(u), 1.0 )):
            print(u, v)
        # Here we control that the wished direction is normalized so we have to have it normalized somewhere
        assert np.isclose( np.linalg.norm(v), 1.0 ), "Wished direction {} not normalized".format(v)

        # numerical safe computation of cos and angle
        cosAngle = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))

        # values outside get projected onto the edges
        cosAngle = np.clip(cosAngle, -1, 1)
        angle    = np.arccos(cosAngle)
        # Maxangle is the max rotation that can be done in the timestep dt. In our case we fix it in the beginning so
        # there might be an issue
        if angle < self.maxAngle or self.randomMov:
            self.curDirection = self.wishedDirection
        # handle antiparallel case
        # this means that u is in the opposite direction of v.
        elif np.isclose(angle, np.pi):
            # assert(False)
            self.curDirection = self.applyrotation(self.curDirection, self.maxAngle)
        else:
            self.curDirection = self.applyrotation_2vec(self.curDirection, self.wishedDirection, self.maxAngle)
        
        # normalize
        self.curDirection /= np.linalg.norm(self.curDirection)
        # NOTE only curdirection gets normalized, whisheddirection does not

    ''' update the direction according to x += vt ''' 
    def updateLocation(self):
        self.location += self.speed*self.dt*self.curDirection
        #self.history.append(self.location)

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

    ''' general calculation in order to apply a rotation to a vector returns the rotated vector'''
    def applyrotation(self, vectortoapply, angletoapply, twodproj=False):
        if(self.dim == 2):
            # In this case to make the rotation work we pad a zero rotate and than extract
            # the first two values in the end
            rotVector = np.array([0., 0., 1.])
            # create rotation
            rotVector *= angletoapply
            r = Rotation.from_rotvec(rotVector)
            # apply rotation to padded wisheddirection
            exp_newwishedir = np.pad(vectortoapply, (0, 1), 'constant')
            exp_wisheddir = r.apply(exp_newwishedir)
            whisheddir = exp_wisheddir[:2]/np.linalg.norm(exp_wisheddir[:2])
            return whisheddir
        
        elif(self.dim == 3):

            """
            r = np.linalg.norm(vectortoapply)
            th = np.arccos(vectortoapply[2]/r)
            phi = np.sign(vectortoapply[1])*np.arccos(vectortoapply[0]/np.linalg.norm(vectortoapply[:2]))

            th += angletoapply[1]
            phi += angletoapply[0]

            x = r*np.sin(th)*np.cos(phi)
            y = r*np.sin(th)*np.sin(phi)
            z = r*np.cos(th)
            wisheddir = np.array([x,y,z])
            wisheddir /= np.linalg.norm(wisheddir)
            """

            r1 = Rotation.from_rotvec(angletoapply[0]*self.rotAxis1)
            r2 = Rotation.from_rotvec(angletoapply[1]*self.rotAxis2)
            tmp = r1.apply(vectortoapply)
            wisheddir = r2.apply(tmp)
            assert np.isclose(np.linalg.norm(wisheddir), 1.0), f"[fish] Wished dir {wisheddir} not normalized {np.linalg.norm(wisheddir)}"

            return wisheddir


    ''' apply a rotation to a vector to turn it by maxangle into the direction of the second vectorreturns the rotated vector'''
    def applyrotation_2vec(self, curDirection, wishedDirection, maxAngle):

        if(self.dim == 2):
            # In this case to make the rotation work we pad the 2 vectors with a 0 in z and then do exactly the same
            # at the end though we'll only take the first 2 entries
            # the first two values in the end
            exp_curDirection = np.pad(curDirection, (0, 1), 'constant')
            exp_wishedDirection = np.pad(wishedDirection, (0, 1), 'constant')
            rotVector = np.cross(exp_curDirection , exp_wishedDirection)
            assert np.linalg.norm(rotVector) > 0, "Rotation vector {} from current {} and wished direction {} with angle {} is zero".format(rotVector, curDirection, wishedDirection, cosAngle)
            rotVector /= np.linalg.norm(rotVector)
            rotVector *= maxAngle

            r = Rotation.from_rotvec(rotVector)

            newDirection = r.apply(exp_curDirection)[:2]
            newDirection /= np.linalg.norm(newDirection)
            newTheta = np.arccos(np.dot(curDirection, newDirection))
        
            return newDirection

        elif(self.dim == 3):
            rotVector = np.cross(curDirection, wishedDirection)
            assert np.linalg.norm(rotVector) > 0, "Rotation vector {} from current {} and wished direction {} with angle {} is zero".format(rotVector, curDirection, wishedDirection, cosAngle)
            rotVector /= np.linalg.norm(rotVector)
            rotVector *= maxAngle
            
            r = Rotation.from_rotvec(rotVector)
            
            newDirection = r.apply(curDirection) 
            newDirection /= np.linalg.norm(newDirection)
            newTheta = np.arccos(np.dot( curDirection, newDirection))

            return newDirection
