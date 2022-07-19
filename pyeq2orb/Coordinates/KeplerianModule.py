from CartesianModule import Cartesian
import sympy as sy
import math as math
from abc import ABC
from fractions import Fraction
from .RotationMatrix import RotAboutY, RotAboutX, RotAboutZ

class KeplerianElements() :
    def __init__(self, sma, ecc, inc, aop, raan, ta, mu) :
        self.SemimajorAxis=sma
        self.Eccentricity=ecc
        self.Inclination=inc
        self.RightAscensionOfAscendingNode=raan
        self.ArgumentOfPeripsis=aop
        self.TrueAnomaly = ta
        self.GravatationalParameter=mu

    @staticmethod
    def FromCartesian(x, y, z, vx, vy, vz, mu) :
        r = Cartesian(x, y, z)
        rMag = r.Magnitude()
        v = Cartesian(vx, vy, vz)
        vMag = v.Magnitude()
        h = r.Cross(v)
        hMag = h.Magnitude()
        
        k = Cartesian(0, 0, 1)
        n = k.Cross(h)
        eVec = (r*(vMag*vMag-mu/rMag)-(v*(r.Dot(v))))/mu
        e = eVec.Magnitude()

        energy = vMag*vMag/2-mu/rMag
        if(e != 1) :
            a = -mu/(2*energy)
        else :
            raise Exception("Eccentricity is parabolic, Keplerianl elements do not work")
        i = sy.acos(h.Z/hMag)
        raan = sy.acos(n.X/n.Magnitude)
        if(n.Y < 0) :
            raan = 2*math.pi - raan
        
        aop = sy.acos(n.dot(e)/(n.Magnitude*e.Magnitude()))
        if(e.Y < 0) :
            aop = 2*math.pi - aop
        
        ta = sy.acos(e.Dot(r)/(e.Magnitude()*rMag))
        if(r.Dot(v) < 0) :
            ta = 2*math.pi - ta

        return KeplerianElements(a, e, i, aop, raan, ta, mu)

    def ToArrayOfElements(self) :
        return [self.SemimajorAxis, self.Eccentricity, self.Inclination, self.ArgumentOfPeripsis, self.RightAscensionOfAscendingNode, self.TrueAnomaly]

    @property
    def SemiminorAxis(self) :        
        return self.SemimajorAxis * sy.sqrt(1.0-self.Eccentricity**2)
    
    @property
    def MeanMotion(self) :
        return sy.sqrt(self.GravatationalParameter/(self.SemimajorAxis**3))
    
    @property
    def Parameter(self) :
        return self.SemimajorAxis*(1-self.Eccentricity**2)

    @property
    def ArgumentOfLatitude(self) :
        return self.ArgumentOfPeripsis+self.RightAscensionOfAscendingNode

    @property
    def BattinH(self):
        return self.MeanMotion*self.SemiminorAxis*self.SemimajorAxis

    @property
    def Radius(self) :
        return self.Parameter/(1+self.Eccentricity*sy.cos(self.TrueAnomaly))

    def ToSympyMatrix(self) :
        sy.Matrix([[self.SemimajorAxis, self.Eccentricity, self.Inclination, self.ArgumentOfPeripsis, self.RightAscensionOfAscendingNode, self.TrueAnomaly]]).transpose()

    def PerifocalToInertialRotationMatrix(self) :
        return RotAboutZ(-1*self.RightAscensionOfAscendingNode) * RotAboutX(-1*self.Inclination) * RotAboutZ(-1*self.ArgumentOfPeripsis)

    def ToPerifocalCartesian(self, theParameter = None) :
        p=theParameter
        if(p == None) :
            p = self.Parameter
        ta = self.TrueAnomaly
        rDenom = 1+self.Eccentricity*sy.cos(ta)
        
        mu = self.GravatationalParameter
        r = Vector([p*(sy.cos(ta)/rDenom), p*(sy.sin(ta)/rDenom), 0])
        e = self.Eccentricity
        firstPart = sy.sqrt(mu/p)
        v = Vector([-1*firstPart *sy.sin(e), firstPart*(e+sy.cos(ta)),0.0])
        return [r,v]

    def ToInertialMotionCartesian(self):
        [r,v] = self.ToPerifocalCartesian()
        rotMatrix = self.PerifocalToInertialRotationMatrix()
        return [rotMatrix*r, rotMatrix*v]

class GaussianEquationsOfMotion :
    def __init__(self, elements : KeplerianElements, accelerationVector : Cartesian) :
        self.Elements = elements
        # Battin page 488
        # note that battin has f = true anomaly, theta = argument of lattitude
        mu = elements.GravatationalParameter
        a = elements.SemimajorAxis
        b = elements.SemiminorAxis
        e = elements.Eccentricity
        i = elements.Inclination
        #raan = elements.RightAscensionOfAscendingNode
        #aop = elements.ArgumentOfPeripsis
        ta = elements.TrueAnomaly
        u = elements.ArgumentOfLatitude

        r = elements.Radius
        b = elements.SemiminorAxis        
        n = elements.MeanMotion
        h = n*a*b
        p = b**2/a
        cTa = sy.cos(ta)
        sTa = sy.sin(ta)
        cU = sy.cos(u)
        sU = sy.sin(u)

        ar = accelerationVector.X
        ah = accelerationVector.Y
        aTh = accelerationVector.Z 

        self.SemimajorAxisDot = (2.0*(a**2.0)/h) * (e*sTa*ar + (p/r)*aTh)
        self.EccentricityDot = (1/h)*(p*sTa*ar+((p+r)*cTa+r*e)*aTh)
        self.InclinationDot = (r*cU/h)*ah
        self.RightAscensionOfAscendingNodeDot = (r*sU/h*sy.sin(i))*ah        
        self.ArgumentOfPeriapsisDot = (1/(h*e))*(-1*p*cTa*ar+(p+r)*sTa*aTh)-r*sTa*sy.cos(i)*ah/(h*sy.sin(i))                

        # take your pick
        #self.TrueAnomalyDot = h/(r**2)-(1/e*h)*(p*cTa*ar-(p+r)*sTa*aTh)
        self.TrueAnomalyDot = h/(r**2)+(1/(e*h))*(p*cTa*ar-(p+r)*sTa*aTh)
        self.MeanAnomalyDot = n + b/(a*h*e)*((p*cTa-2*r*e)*ar - (p+r)*sTa*aTh)
        self.EccentricAnomalyDot = n*a/r + (1/(n*a*e))*((cTa-e)*ar-(1+r/a)*sTa*aTh)

    def ToSympyMatrixEquation(self) :
        lhs = sy.Matrix([[self.Elements.SemimajorAxis, self.Elements.Eccentricity, self.Elements.Inclination, self.Elements.ArgumentOfPeripsis, self.Elements.RightAscensionOfAscendingNode, self.Elements.TrueAnomaly]]).transpose()
        rhs = sy.Matrix([[self.SemimajorAxisDot, self.EccentricityDot, self.InclinationDot, self.ArgumentOfPeriapsisDot, self.RightAscensionOfAscendingNodeDot, self.TrueAnomalyDot]]).transpose()
        return sy.Eq(lhs, rhs)


def CreateSymbolicElements(elementsFunctionOf = None) -> KeplerianElements :
    if(elementsFunctionOf == None) :
        a = sy.Symbol("a", real=True)
        ecc = sy.Symbol('e', nonnegative=True)
        inc = sy.Symbol('i', real=True)
        raan = sy.Symbol(r'\Omega', real=True)
        aop = sy.Symbol(r'\omega', real=True)
        ta = sy.Symbol(r'\nu', real=True)
    else:
        a = sy.Function("a", real=True)(elementsFunctionOf)
        ecc = sy.Function('e', nonnegative=True)(elementsFunctionOf)
        inc = sy.Function('i', real=True)(elementsFunctionOf)
        raan = sy.Function(r'\Omega', real=True)(elementsFunctionOf)
        aop = sy.Function(r'\omega', real=True)(elementsFunctionOf)
        ta = sy.Function(r'\nu', real=True)(elementsFunctionOf)
    mu = sy.Symbol(r'\mu', positive=True)
    return KeplerianElements(a, ecc, inc, aop, raan, ta, mu)
