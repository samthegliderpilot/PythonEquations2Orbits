import sympy as sy
import math
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
import sympy as sy
from typing import List

class EquinoctialElements:
    def __init__(self, a, h, k, p, q, f, mu) :
        self.SemiMajorAxis = a
        self.EccentricitySinTermH = h
        self.EccentricityCosTermK = k
        self.InclinationSinTermP = p
        self.InclinationCosTermQ = q
        self.TrueLongitude = f
        self.GravitationalParameter = mu
    
    def ToKeplerian(self) -> KeplerianElements :
        a = self.SemiMajorAxis
        h = self.EccentricitySinTermH
        k = self.EccentricityCosTermK
        p = self.InclinationSinTermP
        q = self.InclinationCosTermQ
        ta = self.TrueLongitude

        e = (h**2+k**2)**(1.0/2.0)
        i = 2*sy.atan(p**2+q**2)**(1.0/2.0)
        raan = sy.atan(p/q)
        w = sy.atan(h/k)-sy.atan(p/q)
        ta = self.TrueLongitude - w - raan

        return KeplerianElements(a, e, i, w, raan, ta, self.GravitationalParameter)

    def ToMotionCartesian(self) -> MotionCartesian :
        return self.ToKeplerian().ToInertialMotionCartesian() # TODO: something that avoids keplerian elements

    def ToArray(self) -> List :
        return [self.SemiMajorAxis, self.EccentricitySinTermH, self.EccentricityCosTermK, self.InclinationSinTermP, self.InclinationCosTermQ, self.TrueLongitude]

    @staticmethod
    def FromMotionCartesian(motion, gravitationalParameter) :
        # TODO: something that avoids keplerian elements
        return ConvertKeplerianToEquinoctial(KeplerianElements.FromMotionCartesian(motion, gravitationalParameter))

def ConvertKeplerianToEquinoctial(keplerianElements : KeplerianElements) ->EquinoctialElements :
    a = keplerianElements.SemiMajorAxis
    e = keplerianElements.Eccentricity
    i = keplerianElements.Inclination
    w = keplerianElements.ArgumentOfPeriapsis
    raan = keplerianElements.RightAscensionOfAscendingNode
    ta = keplerianElements.TrueAnomaly

    h = e*sy.sin(w+raan)
    k = e*sy.cos(w+raan)
    p = sy.tan(i/w)*sy.sin(raan)
    q = sy.tan(i/w)*sy.cos(raan)
    f = w+raan+ta

    return EquinoctialElements(a, h, k, p, q, f, keplerianElements.GravitationalParameter)


def CreateSymbolicElements(elementOf = None) -> EquinoctialElements :
    if(elementOf == None):
        a = sy.Symbol('a', positive=True)
        h = sy.Symbol('h', real=True)
        k = sy.Symbol('k', real=True)
        p = sy.Symbol('p', real=True)
        q= sy.Symbol('q', real=True)
        l = sy.Symbol('L', real=True)
    else :
        a = sy.Function('a', positive=True)(elementOf)
        h = sy.Function('h', real=True)(elementOf)
        k = sy.Function('k', real=True)(elementOf)
        p = sy.Function('p', real=True)(elementOf)
        q= sy.Function('q', real=True)(elementOf)
        l = sy.Function('L', real=True)(elementOf)

    mu = sy.Symbol(r'\mu', positive=True)
    return EquinoctialElements(a, h, k, p, q, l, mu)

