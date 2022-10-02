import sympy as sy
import math
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
import sympy as sy
from typing import List

class EquinoctialElements:
    def __init__(self, periapsis, f, g, h, k, l, mu) :
        self.PeriapsisRadius = periapsis
        self.EccentricityCosTermF = g
        self.EccentricitySinTermG = f        
        self.InclinationSinTermK = h
        self.InclinationCosTermH = k
        self.TrueLongitude = l
        self.GravitationalParameter = mu
    
    def ToKeplerian(self) -> KeplerianElements :
        per = self.PeriapsisRadius
        g = self.EccentricitySinTermG
        f = self.EccentricityCosTermF
        k = self.InclinationSinTermK
        h = self.InclinationCosTermH
        tl = self.TrueLongitude

        e = sy.sqrt(f**2+g**2)
        a = per/(1-f*f-g*g)
        i = 2*sy.atan(sy.sqrt(h*h+k*k))#, (1-h*h-k*k))
        raan = sy.atan2(k, h)
        w = sy.atan2(g*h-f*k, f*h+g*k)
        ta = tl - sy.atan2(f, g) #TODO: atan2's for the angles?

        return KeplerianElements(a, e, i, w, raan, ta, self.GravitationalParameter)

    def ToMotionCartesian(self) -> MotionCartesian :
        p = self.PeriapsisRadius
        g = self.EccentricitySinTermG
        f = self.EccentricityCosTermF
        k = self.InclinationSinTermK
        h = self.InclinationCosTermH
        tl = self.TrueLongitude
        mu = self.GravitationalParameter
        e = (f**2+g**2)**(1.0/2.0)
        a = p/(1-f*f-g*g)
        i = 2*sy.atan(h**2+k**2)**(1.0/2.0)
        alp2 = h*h-k*k
        s2 = 1+h*h+k*k
        w = 1+f*sy.cos(tl)+g*sy.sin(tl)
        rM = p/w
        
        x = (rM/s2)*(sy.cos(tl) + alp2*sy.cos(tl) + 2*h*k *sy.sin(tl))
        y = (rM/s2)*(sy.sin(tl) + alp2*sy.sin(tl) + 2*h*k *sy.cos(tl))
        z = (2*rM/s2)*(h*sy.sin(tl) - k*sy.cos(tl))

        vx = (-1/s2)*sy.sqrt(mu/p) * (sy.sin(tl) + alp2*sy.sin(tl) - 2*h*k*sy.cos(tl) + g - 2*f*h*k + alp2*g)
        vy = (-1/s2)*sy.sqrt(mu/p) * (-1*sy.cos(tl) + alp2*sy.cos(tl) + 2*h*k*sy.sin(tl) - f - 2*g*h*k + alp2*f)
        vz = (2/s2) * (sy.sqrt(mu/p)) * (h*sy.cos(tl) + k*sy.sin(tl) + f*h + g*k)

        return MotionCartesian(Cartesian(x, y, z), Cartesian(vx, vy, vz))
        #return self.ToKeplerian().ToInertialMotionCartesian() # TODO: something that avoids keplerian elements

    def ToArray(self) -> List :
        return [self.PeriapsisRadius, self.EccentricityCosTermF, self.EccentricitySinTermG, self.InclinationCosTermH, self.InclinationSinTermK, self.TrueLongitude]

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

def ConvertKeplerianToEquinoctial(keplerianElements : KeplerianElements) ->EquinoctialElements :
    a = keplerianElements.SemiMajorAxis
    e = keplerianElements.Eccentricity
    i = keplerianElements.Inclination
    w = keplerianElements.ArgumentOfPeriapsis
    raan = keplerianElements.RightAscensionOfAscendingNode
    ta = keplerianElements.TrueAnomaly

    per = a*(1.0-e)
    f = e*sy.cos(w+raan)
    g = e*sy.sin(w+raan)
    
    h = sy.tan(i/2)*sy.cos(raan)
    k = sy.tan(i/2)*sy.sin(raan)
    l = w+raan+ta

    return EquinoctialElements(per, f, g, h, k, l, keplerianElements.GravitationalParameter)


def CreateSymbolicElements(elementOf = None) -> EquinoctialElements :
    if(elementOf == None):
        p = sy.Symbol('p', positive=True)
        f = sy.Symbol('f', real=True)
        g = sy.Symbol('g', real=True)
        h = sy.Symbol('h', real=True)
        k= sy.Symbol('k', real=True)
        l = sy.Symbol('L', real=True)
    else :
        p = sy.Function('p', positive=True)(elementOf)
        f = sy.Function('f', real=True)(elementOf)
        g = sy.Function('g', real=True)(elementOf)
        h = sy.Function('h', real=True)(elementOf)
        k= sy.Function('k', real=True)(elementOf)
        l = sy.Function('L', real=True)(elementOf)

    mu = sy.Symbol(r'\mu', positive=True)
    return EquinoctialElements(p, f, g, h, k, l, mu)

