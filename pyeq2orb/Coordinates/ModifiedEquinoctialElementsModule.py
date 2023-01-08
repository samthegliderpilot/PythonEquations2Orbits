import sympy as sy
import math
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from typing import List

class ModifiedEquinoctialElements:
    def __init__(self, p, f, g, h, k, l, mu) :
        self.SemiParameter = p
        self.EccentricityCosTermF = f
        self.EccentricitySinTermG = g
        self.InclinationCosTermH = h
        self.InclinationSinTermK = k
        self.TrueLongitude = l
        self.GravitationalParameter = mu

    @staticmethod
    def FromCartesian(x, y, z, vx, vy, vz) :
        pass #TODO
    
    @property
    def W(self) :
        return 1+self.EccentricityCosTermF*sy.cos(self.TrueLongitude)+self.EccentricitySinTermG*sy.sin(self.TrueLongitude)

    @property
    def Q(self) :
        return self.W

    @property
    def SSquared(self) :
        return 1+self.InclinationCosTermH**2+self.InclinationSinTermK**2
    
    @property
    def AlphaSquared(self) :
        return self.InclinationCosTermH**2-self.InclinationSinTermK**2

    @property
    def Radius(self) :
        return self.SemiParameter / self.W

    def ToKeplerian(self) -> KeplerianElements :
        l = self.TrueLongitude
        f = self.EccentricityCosTermF
        g = self.EccentricitySinTermG
        h = self.InclinationCosTermH
        k = self.InclinationSinTermK
        p = self.SemiParameter

        a = p/(1-f**2-g**2)
        e = sy.sqrt(f*f+g*g)
        # it is not clear to me if the atan2 in the MME PDF passes in x,y or y,x (y,x is right for sy)
        i = sy.atan2(1-h*h-k*k, 2*sy.sqrt(h*h+k*k))
        w = sy.atan2(f*h+g*k, g*h-f*k)
        raan = sy.atan2(h, k)
        ta = l-raan+w
        return KeplerianElements(a, e, i, w, raan, ta, self.GravitationalParameter)
    
    def ToMotionCartesian(self) -> MotionCartesian :
        l = self.TrueLongitude
        f = self.EccentricityCosTermF
        g = self.EccentricitySinTermG        
        h = self.InclinationCosTermH
        k = self.InclinationSinTermK
        p = self.SemiParameter
        mu = self.GravitationalParameter
        w = self.W

        r = self.Radius
        sSquared = self.SSquared
        w = self.W
        alpSq = self.AlphaSquared
        
        cosL = sy.cos(l)
        sinL = sy.sin(l)
        rx = (r/sSquared)*(cosL + alpSq*cosL + 2*h*k*sinL)
        ry = (r/sSquared)*(sinL - alpSq*sinL + 2*h*k*cosL)
        rz = 2*(r/sSquared)*(h*sinL - k*cosL)

        sqrtMuOverP = sy.sqrt(mu/p)
        vx = -1*(1/sSquared) *sqrtMuOverP*(   sinL + alpSq*sinL - 2*h*k*cosL + g - 2*f*h*k + alpSq*g)
        vy = -1*(1/sSquared) *sqrtMuOverP*(-1*cosL + alpSq*cosL + 2*h*k*sinL - f + 2*g*h*k + alpSq*f)
        vz = (2/(sSquared))*sqrtMuOverP*(h*cosL + k*sinL + f*h + g*k)

        return MotionCartesian(Cartesian(rx, ry, rz), Cartesian(vx, vy, vz))

    def ToCartesianArray(self) :
        motion = self.ToCartesian()
        return [[motion.Position.X,motion.Position.Y,motion.Position.Z], [motion.Velocity.X,motion.Velocity.Y,motion.Velocity.Z]]

    def ToPseuodNormalizedCartesian(self) :
        # sorry for the copy/paste of the above
        p = self.SemiParameter
        f = self.EccentricityCosTermF
        g = self.EccentricitySinTermG        
        k = self.InclinationSinTermK
        h = self.InclinationCosTermH
        tl = self.TrueLongitude
        mu = self.GravitationalParameter
        alp2 = h*h-k*k
        s2 = 1+h*h+k*k
        w = 1+f*sy.cos(tl)+g*sy.sin(tl)
        rM = p/w
        
        x = (sy.cos(tl) + alp2*sy.cos(tl) + 2*h*k *sy.sin(tl))
        y = (sy.sin(tl) + alp2*sy.sin(tl) + 2*h*k *sy.cos(tl))
        z = (1/s2)*(h*sy.sin(tl) - k*sy.cos(tl))

        # not being rigirous about the normalizing here, just removing various parameters
        vx = (-1) * (sy.sin(tl) + alp2*sy.sin(tl) - 2*h*k*sy.cos(tl) + g - 2*f*h*k + alp2*g)
        vy = (-1) * (-1*sy.cos(tl) + alp2*sy.cos(tl) + 2*h*k*sy.sin(tl) - f - 2*g*h*k + alp2*f)
        vz = (1/s2) * (h*sy.cos(tl) + k*sy.sin(tl) + f*h + g*k)

        return MotionCartesian(Cartesian(x, y, z), Cartesian(vx, vy, vz))   

    def ToArray(self) -> List :
        return [self.SemiParameter, self.EccentricityCosTermF, self.EccentricitySinTermG, self.InclinationCosTermH, self.InclinationSinTermK, self.TrueLongitude]

    @staticmethod
    def FromMotionCartesian(motion, gravitationalParameter) :
        # TODO: something that avoids keplerian elements
        return ConvertKeplerianToEquinoctial(KeplerianElements.FromMotionCartesian(motion, gravitationalParameter))

    @staticmethod
    def CreateEphemeris(equinoctialElementsList) -> List[MotionCartesian] :
        motions = []
        for equi in equinoctialElementsList :
            motions.append(equi.ToMotionCartesian())
        return motions

    def CreatePerturbationMatrix(self) ->sy.Matrix :
        eqElements=self
        mu = eqElements.GravitationalParameter
        pEq = eqElements.SemiParameter        
        fEq = eqElements.EccentricityCosTermF
        gEq = eqElements.EccentricitySinTermG        
        kEq = eqElements.InclinationCosTermH
        hEq = eqElements.InclinationSinTermK
        lEq = eqElements.TrueLongitude
        w = 1+fEq*sy.cos(lEq)+gEq*sy.sin(lEq)
        s2 = 1+hEq**2+kEq**2
        sqrtPOverMu=sy.sqrt(pEq/mu)
        # note that in teh 3rd row, middle term, I added a 1/w because I think it is correct even though the MME pdf doesn't have it
        B = sy.Matrix([[0, (2*pEq/w)*sqrtPOverMu, 0],
                    [sqrtPOverMu*sy.sin(lEq), sqrtPOverMu*(1/w)*((w+1)*sy.cos(lEq)+fEq), -1*sqrtPOverMu*(gEq/w)*(hEq*sy.sin(lEq)-kEq*sy.cos(lEq))],
                    [-1*sqrtPOverMu*sy.cos(lEq), sqrtPOverMu*(1/w)*((w+1)*sy.sin(lEq)+gEq), sqrtPOverMu*(fEq/w)*(hEq*sy.sin(lEq)-kEq*sy.cos(lEq))],
                    [0,0,sqrtPOverMu*(s2*sy.cos(lEq)/(2*w))],
                    [0,0,sqrtPOverMu*(s2*sy.sin(lEq)/(2*w))],
                    [0,0,sqrtPOverMu*(hEq*sy.sin(lEq)-kEq*sy.cos(lEq))/w]])
        return B        

def ConvertKeplerianToEquinoctial(keplerianElements : KeplerianElements, nonModedLongitude = True) ->ModifiedEquinoctialElements :
    a = keplerianElements.SemiMajorAxis
    e = keplerianElements.Eccentricity
    i = keplerianElements.Inclination
    w = keplerianElements.ArgumentOfPeriapsis
    raan = keplerianElements.RightAscensionOfAscendingNode
    ta = keplerianElements.TrueAnomaly

    per = a*(1.0-e**2)
    f = e*sy.cos(w+raan)
    g = e*sy.sin(w+raan)
    
    h = sy.tan(i/2)*sy.cos(raan)
    k = sy.tan(i/2)*sy.sin(raan)
    if nonModedLongitude :
        l = w+raan+ta
    else :
        l = ((w+raan+ta) % (2*math.pi)).simplify()

    return ModifiedEquinoctialElements(per, f, g, h, k, l, keplerianElements.GravitationalParameter)

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


def TwoBodyGravityForceOnElements(elements : ModifiedEquinoctialElements, w=None) ->sy.Matrix:
    if(w==None) :
        w = elements.W
    return sy.Matrix([[0],[0],[0],[0],[0],[sy.sqrt(elements.GravitationalParameter*elements.SemiParameter)*(w/elements.SemiParameter)**2]])




class EquinoctialElementsHalfI :
    def __init__(self, a, h, k, p, q, longitude, mu) :
        self.SemiMajorAxis = a
        self.EccentricitySinTermH = h
        self.EccentricityCosTermJ = k
        self.InclinationSinTermP = p
        self.InclinationCosTermQ = q
        self.Longitude = longitude # consider how things would work if longitude was left ambigious
        self.GravitationalParameter = mu

    def ConvertToModifiedEquinoctial(self) ->ModifiedEquinoctialElements:
        eSq = self.EccentricityCosTermJ**2+self.EccentricitySinTermH**2
        param = self.SemiMajorAxis * (1.0 - eSq)
        return ModifiedEquinoctialElements(param, self.EccentricityCosTermJ, self.EccentricitySinTermH, self.InclinationCosTermQ, self.InclinationSinTermP, self.Longitude, self.GravitationalParameter)

    @staticmethod
    def FromModifiedEquinoctialElements(mee : ModifiedEquinoctialElements) :
        eSq = mee.EccentricityCosTermF**2+mee.EccentricitySinTermG**2
        sma = mee.SemiParameter/(1.0-eSq)
        return EquinoctialElementsHalfI(sma, mee.EccentricitySinTermG, mee.EccentricityCosTermF, mee.InclinationSinTermK, mee.InclinationCosTermH, mee.TrueLongitude, mee.GravitationalParameter)

    def CreateSymbolicElements(elementOf = None)  : #TODO kargs of mu and element of
        if(elementOf == None) : 
            a = sy.Symbol('a', real=True)
            h = sy.Symbol('h', real=True)
            k = sy.Symbol('k', real=True)
            p = sy.Symbol('p', real=True)
            q= sy.Symbol('q', real=True)
            l = sy.Symbol('l', real=True)
        else :
            a = sy.Function('a', positive=True)(elementOf)
            h = sy.Function('h', real=True)(elementOf)
            k = sy.Function('k', real=True)(elementOf)
            p = sy.Function('p', real=True)(elementOf)
            q= sy.Function('q', real=True)(elementOf)
            l = sy.Function('l', real=True)(elementOf)
        mu = sy.Symbol(r'\mu', positive=True, real=True)

        return EquinoctialElementsHalfI(a, h, k, p, q, l, mu)

