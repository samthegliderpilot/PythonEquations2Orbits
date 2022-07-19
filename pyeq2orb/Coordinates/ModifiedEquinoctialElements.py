import sympy as sy
import math
from .KeplerianModule import KeplerianElements
from CartesianModule import Cartesian, MotionCartesian

class ModifiedEquinoctialElements:
    def __init__(self, p, f, g, h, k, l, mu) :
        self.SemiParameter = p
        self.EccentricitySinTermF = f
        self.EccentricityCosTermG = g
        self.InclinationSinTermH = h
        self.InclinationCosTermK = k
        self.TrueLongitude = l
        self.GravitationalParameter = mu

    @staticmethod
    def FromCartesian(x, y, z, vx, vy, vz) :
        pass
    
    @property
    def W(self) :
        return 1+self.EccentricitySinTermF*sy.cos(self.TrueLongitude)+self.EccentricityCosTermG*sy.sin(self.TrueLongitude)

    @property
    def Q(self) :
        return self.W

    @property
    def SSquared(self) :
        return 1+self.InclinationSinTermH**2+self.InclinationCosTermK**2
    
    @property
    def Alpha(self) :
        return self.InclinationSinTermH**2-self.InclinationCosTermK**2

    def ToKeplerian(self) -> KeplerianElements :
        l = self.TrueLongitude
        f = self.EccentricitySinTermF
        g = self.EccentricityCosTermG
        h = self.InclinationSinTermH
        k = self.InclinationCosTermK
        p = self.SemiParameter

        a = p/(1-f*f-g*g)
        e = sy.sqrt(f*f+g*g)
        i = sy.atan2(2*sy.sqrt(h*h+k*k), a-h*h-k*k)
        w = sy.atan2(g*h-f*k, f*h+g*k)
        raan = sy.atan2(k, h)
        ta = l-raan+w
        return KeplerianElements(a, e, i, w, raan, ta, self.GravitationalParameter)
    
    def ToCartesian(self) -> MotionCartesian :
        l = self.TrueLongitude
        f = self.EccentricitySinTermF
        g = self.EccentricityCosTermG
        h = self.InclinationSinTermH
        k = self.InclinationCosTermK
        p = self.SemiParameter
        mu = self.GravitationalParameter
        w = self.W

        r = p/w
        sSquared = self.SSquared
        w = self.W
        alp = self.Alpha
        alpSq = alp**2
        
        rx = (r/sSquared)*(sy.cos(l) + alpSq*sy.cos(l) + 2*h*k*sy.sin(l))
        ry = (r/sSquared)*(sy.sin(l) + alpSq*sy.sin(l) + 2*h*k*sy.cos(l))
        rz = 2*(r/sSquared)*(h*sy.sin(l) - k*sy.cos(l))

        sqrtMuOverP = sy.sqrt(mu/p)
        vx = -1*(1/sSquared) *sqrtMuOverP*(sy.sin(l) + alpSq*sy.sin(l)-2*h*k*sy.cos(l) + g - 2*f*h*k + alpSq*g)
        vy = -1*(1/sSquared) *sqrtMuOverP*(-1*sy.cos(l) + alpSq*sy.cos(l)+2*h*k*sy.sin(l) + f + 2*g*h*k+alpSq*f)
        vz = (2/(sSquared))*sqrtMuOverP*(h*sy.cos(l) + k*sy.sin(l) + f*h+g*k)

        return MotionCartesian(Cartesian(rx, ry, rz), Cartesian(vx, vy, vz))

    def ToCartesianArray(self) :
        motion = self.ToCartesian()
        return [[motion.Position.X,motion.Position.Y,motion.Position.Z], [motion.Velocity.X,motion.Velocity.Y,motion.Velocity.Z]]

def CreateSymbolicElements(elementOf = None) -> ModifiedEquinoctialElements : #TODO kargs of mu and element of
    if(elementOf == None) : 
        p = sy.Symbol('p', positive=True)
        f = sy.Symbol('f', real=True)
        g = sy.Symbol('g', real=True)
        h = sy.Symbol('h', real=True)
        k= sy.Symbol('k', real=True)
        l = sy.Symbol('l', real=True)
    else :
        p = sy.Function('p', positive=True)(elementOf)
        f = sy.Function('f', real=True)(elementOf)
        g = sy.Function('g', real=True)(elementOf)
        h = sy.Function('h', real=True)(elementOf)
        k= sy.Function('k', real=True)(elementOf)
        l = sy.Function('l', real=True)(elementOf)
    mu = sy.Symbol(r'\mu', positive=True, real=True)

    return ModifiedEquinoctialElements(p, f, g, h, k, l, mu)


def PerturbationMatrix(elements : ModifiedEquinoctialElements, w = None, sSquared = None) -> sy.Matrix:
    l = elements.TrueLongitude
    f = elements.EccentricitySinTermF
    g = elements.EccentricityCosTermG
    h = elements.InclinationSinTermH
    k = elements.InclinationCosTermK
    p = elements.SemiParameter
    mu = elements.GravitationalParameter

    if(w==None) :
        w = elements.W

    if(sSquared == None) :
        sSquared = elements.SSquared
    sqrtpOverMu=sy.sqrt(p/mu)
    a = sy.Matrix([[0, (2*p/w)*sqrtpOverMu, 0],
                [sqrtpOverMu*sy.sin(l), sqrtpOverMu*(1/w)*((w+1)*sy.cos(l)+f), -1*sqrtpOverMu*(g/w)*(h*sy.sin(l)-k*sy.cos(l))],
                [-1*sqrtpOverMu*sy.cos(l), sqrtpOverMu*(1/w)*((w+1)*sy.sin(l)+g), sqrtpOverMu*(f/w)*(h*sy.sin(l)-k*sy.cos(l))],
                [0,0,sqrtpOverMu*(sSquared*sy.cos(l)/(2*w))],
                [0,0,sqrtpOverMu*(sSquared*sy.sin(l)/(2*w))],
                [0,0,sqrtpOverMu*(1/w)*(h*sy.sin(l)-k*sy.cos(l))]])

    return a


def TwoBodyGravityForceOnElements(elements : ModifiedEquinoctialElements, w=None) ->sy.Matrix:
    if(w==None) :
        w = elements.W
    return sy.Matrix([[0],[0],[0],[0],[0],[sy.sqrt(elements.GravitationalParameter*elements.SemiParameter)*(w/elements.SemiParameter)**2]])




