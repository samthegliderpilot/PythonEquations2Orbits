from __future__ import annotations
from .CartesianModule import Cartesian, MotionCartesian #type:ignore
import sympy as sy
import math as math
from .RotationMatrix import RotAboutXValladoConvention, RotAboutY, RotAboutX, RotAboutZ, RotAboutZValladoConvention #type:ignore
from typing import Optional, List, Union
from pyeq2orb.Utilities.Typing import SymbolOrNumber

def MeanAnomalyFromEccentricAnomaly(eccentricAnomaly: SymbolOrNumber, eccentricity : SymbolOrNumber) ->SymbolOrNumber:
    if eccentricAnomaly % math.pi == 0:
        return eccentricAnomaly
    ma = eccentricAnomaly - eccentricity*sy.sin(eccentricAnomaly)
    if hasattr(ma, "evalf"):
        ma = ma.evalf()
    ma = ma.evalf()
    if ma.is_Float:
        ma = float(ma)%(2*math.pi)
    return ma

def EccentricAnomalyFromMeanAnomaly(meanAnomaly : float, eccentricity : float, tolerance : float = 0.000000001) : #TODO: Parabolic and hyperbolic
    if meanAnomaly % math.pi == 0:
        return meanAnomaly
    ma = meanAnomaly % (2*math.pi)
    if ma > math.pi:
        eccentricAnomaly = ma - eccentricity
    else :
        eccentricAnomaly = ma + eccentricity
    
    lastGuess = ma*2 # just something large enough to not trigger on the first iteration
    while abs(lastGuess - eccentricAnomaly) > tolerance:
        lastGuess = eccentricAnomaly
        eccentricAnomaly = lastGuess + (ma-lastGuess+eccentricity*math.sin(lastGuess))/(1-eccentricity*math.cos(lastGuess))
    return eccentricAnomaly

def TrueAnomalyFromEccentricAnomaly(eccentricAnomaly: SymbolOrNumber, eccentricity: SymbolOrNumber) ->SymbolOrNumber:
    if eccentricAnomaly % math.pi == 0:
        return eccentricAnomaly

    if eccentricity < 1.0 :
        sinEa = sy.sin(eccentricAnomaly) * sy.sqrt(1-eccentricity**2)/(1-eccentricity*sy.cos(eccentricAnomaly))
        cosEa = (sy.cos(eccentricAnomaly) - eccentricity)/(1-eccentricity*sy.cos(eccentricAnomaly))
        ta = sy.atan2(sinEa, cosEa)
    elif eccentricity == 1.0 :
        raise NotImplementedError()
    elif float(eccentricity) > 0.0:
        sinTa = -1*sy.sinh(eccentricAnomaly) * sy.sqrt((eccentricity**2)-1)/(1-eccentricity*sy.cosh(eccentricAnomaly))
        cosTa = (sy.cosh(eccentricAnomaly) - eccentricity)/(1-eccentricity*sy.cosh(eccentricAnomaly))
        ta = sy.atanh(cosTa/sinTa)        
    if hasattr(ta, "evalf"):
        ta = ta.evalf()
    ta = ta.evalf()
    if ta.is_Float:
        ta = float(ta)%(2*math.pi)
    return ta

def EccentricAnomalyFromTrueAnomaly(trueAnomaly: SymbolOrNumber, eccentricity : SymbolOrNumber) ->SymbolOrNumber:
    if trueAnomaly % math.pi == 0:
        return trueAnomaly

    eccentricityAsFloat =-1.0
    if isinstance(eccentricity, float) or hasattr(eccentricity, "is_float") :
        eccentricityAsFloat = float(eccentricity)
    if eccentricityAsFloat < 1.0 :# minus 1 default case is caught here too
        sinEa = sy.sin(trueAnomaly)*sy.sqrt(1-eccentricity**2)/(1+eccentricity*sy.cos(trueAnomaly))
        cosEa = (eccentricity+sy.cos(trueAnomaly))/(1+eccentricity*sy.cos(trueAnomaly))
        ea = sy.atan2(sinEa, cosEa)
    elif eccentricityAsFloat == 1.0:
        ea = sy.atan(trueAnomaly/2.0)
    elif eccentricityAsFloat > 1.0 :
        sinHa = sy.sin(trueAnomaly)*sy.sqrt((eccentricity**2) - 1)/(1+eccentricity*sy.cos(trueAnomaly))
        cosHa = (eccentricity+sy.cos(trueAnomaly))/(1+eccentricity*sy.cos(trueAnomaly))
        ea = sy.atan2(sinHa, cosHa)
    if hasattr(ea, "evalf"):
        ea = ea.evalf()
    if ea.is_Float:
        ea = float(ea)%(2*math.pi)
    return ea

def MeanAnomalyFromTrueAnomaly(trueAnomaly, eccentricity):
    ea = EccentricAnomalyFromTrueAnomaly(trueAnomaly, eccentricity)
    ma = MeanAnomalyFromEccentricAnomaly(ea, eccentricity)
    return ma

def TrueAnomalyFromMeanAnomaly(meanAnomaly, eccentricity) :
    ea = EccentricAnomalyFromMeanAnomaly(meanAnomaly, eccentricity)
    ta = TrueAnomalyFromEccentricAnomaly(ea, eccentricity)
    return ta

class KeplerianElements() :
    """Represents a set of Keplerian Elements with true anomaly as the fast variable. Note that 
    Keplerian Elements do not represent parabolic or circular orbits well.
    """
    def __init__(self, sma :SymbolOrNumber, ecc:SymbolOrNumber, inc:SymbolOrNumber, aop:SymbolOrNumber, raan:SymbolOrNumber, ta:SymbolOrNumber, mu:SymbolOrNumber) :
        """Initializes a new instance.  The values passed in are often numbers or symbols.

        Args:
            sma : The semi-major axis
            ecc : The eccentricity
            inc : The inclination
            aop : The argument of periapsis
            raan : The right ascension of the ascending node
            ta : The true anomaly
            mu : The gravitational parameter
        """
        self.SemiMajorAxis=sma
        self.Eccentricity=ecc
        self.Inclination=inc
        self.RightAscensionOfAscendingNode=raan
        self.ArgumentOfPeriapsis=aop
        self.TrueAnomaly = ta
        self.GravitationalParameter=mu

    @staticmethod
    def FromCartesian(x:SymbolOrNumber, y:SymbolOrNumber, z:SymbolOrNumber, vx:SymbolOrNumber, vy:SymbolOrNumber, vz:SymbolOrNumber, mu:SymbolOrNumber) -> KeplerianElements:
        """Creates a set of Keplerian Elements from the provided position and velocity elements. 
        The passed in values are often numbers or symbols.

        Args:
            x : The x component of position
            y : The y component of position
            z : The z component of position
            vx : The x component of velocity
            vy : The y component of velocity
            vz : The z component of velocity
            mu : The gravitational parameter

        Raises:
            Exception: If the orbit is parabolic this will throw.

        Returns:
            KeplerianElements: The Keplerian Elements that represent the passed in values.
        """
        r = Cartesian(x, y, z)
        rMag = r.norm()
        v = Cartesian(vx, vy, vz)
        vMag = v.norm()
        h = r.cross(v)
        hMag = h.norm()
        
        k = Cartesian(0, 0, 1)
        n = k.cross(h)
        eVec = (r*(vMag*vMag-mu/rMag)-(v*(r.dot(v))))/mu
        e = eVec.norm()

        energy = vMag*vMag/2-mu/rMag
        if(e != 1) :
            a = -mu/(2*energy)
        else :
            raise Exception("Eccentricity is parabolic, Keplerian elements do not work")
        i = sy.acos(h.Z/hMag)
        raan = sy.acos(n.X/n.norm())
        if(isinstance(raan, sy.Float) and n.Y < 0) :
            raan = 2*math.pi - raan
        
        aop = sy.acos(n.dot(eVec)/(n.norm()*e))
        if(isinstance(aop, sy.Float) and eVec.Z < 0) :
            aop = 2*math.pi - aop
        
        ta = sy.acos(eVec.dot(r)/(e*rMag))
        if(isinstance(ta, sy.Float) and r.dot(v) < 0) :
            ta = 2*math.pi - ta

        return KeplerianElements(a, e, i, aop, raan, ta, mu)
    
    @staticmethod
    def FromMotionCartesian(motion : MotionCartesian, mu:SymbolOrNumber) -> KeplerianElements:
        """Creates a set of Keplerian Elements from the provided motion.
        The values in the motion and mu are often numbers or symbols.

        Args:
            motion (MotionCartesian): The motion to convert.
            mu : The gravitational parameter.

        Raises:
            Exception: If the orbit is parabolic this will throw.

        Returns:
            KeplerianElements: The Keplerian Elements that represent the passed in values.
        """
        return KeplerianElements.FromCartesian(motion.Position[0], motion.Position[1], motion.Position[2], motion.Velocity[0], motion.Velocity[1], motion.Velocity[2], mu)

    def ToArrayOfElements(self) ->List[SymbolOrNumber]:
        return [self.SemiMajorAxis, self.Eccentricity, self.Inclination, self.ArgumentOfPeriapsis, self.RightAscensionOfAscendingNode, self.TrueAnomaly]
    
    def ToSympyMatrix(self) -> sy.Matrix :
        return sy.Matrix([[self.SemiMajorAxis, self.Eccentricity, self.Inclination, self.ArgumentOfPeriapsis, self.RightAscensionOfAscendingNode, self.TrueAnomaly]]).transpose()

    @property
    def SemiMinorAxis(self) ->SymbolOrNumber:        
        """Gets the semi-minor axis

        Returns:
            float or sy.Ex: The semi-minor axis.
        """
        return self.SemiMajorAxis * sy.sqrt(1.0-self.Eccentricity**2)
    
    @property
    def MeanMotion(self)->SymbolOrNumber :
        """Gets the mean motion.

        Returns:
            float or sy.Ex: The mean motion
        """        
        return sy.sqrt(self.GravitationalParameter/(sy.Abs(self.SemiMajorAxis)**3))
    
    @property
    def Parameter(self)->SymbolOrNumber :
        """Gets the parameter (semilatus rectum)

        Returns:
            float or sy.Ex: The parameter
        """
        return self.SemiMajorAxis*(1.0-self.Eccentricity**2)

    @property
    def ArgumentOfLatitude(self) ->SymbolOrNumber:
        """Gets the argument of latitude.

        Returns:
            float or sy.Ex: The argument of latitude
        """
        return self.ArgumentOfPeriapsis+self.RightAscensionOfAscendingNode

    @property
    def BattinH(self) -> SymbolOrNumber:
        """Gets an expression of the magnitude of the angular momentum in terms of the mean motion, semi-major axis 
        and semi-minor axis.  This is from Battin's textbook (in the 400's).

        Returns:
            float or sy.Ex: The magnitude of the angular momentum.
        """
        return self.MeanMotion*self.SemiMinorAxis*self.SemiMajorAxis

    @property
    def Radius(self) ->SymbolOrNumber:
        """Gets an expression for the radius of the orbit.

        Returns:
            float or sy.Ex: The radius of the orbit.
        """
        return self.Parameter/(1+self.Eccentricity*sy.cos(self.TrueAnomaly))

    def PerifocalToInertialRotationMatrix(self) -> sy.Matrix :
        """Gets the rotation matrix from perifocal to inertial coordinates.

        Returns:
            sy.Matrix: The rotation matrix from perifocal to inertial coordinates.
        """
        return RotAboutZValladoConvention(-1.0*self.RightAscensionOfAscendingNode) * RotAboutXValladoConvention(-1.0*self.Inclination) * RotAboutZValladoConvention(-1*self.ArgumentOfPeriapsis)

    def ToPerifocalCartesian(self, theParameter : Optional[SymbolOrNumber] = None) -> MotionCartesian :
        """Converts these elements to cartesian elements in perifocal Cartesian values.

        Args:
            theParameter (obj, optional): The semilatus rectum.  If None then it will be evaluated. Defaults to None.

        Returns:
            MotionCartesian: The perifocal coordinates.
        """
        
        if(theParameter == None) :
            theParameter = self.Parameter
        p=theParameter 
        ta = self.TrueAnomaly
        rDenom = 1+self.Eccentricity*sy.cos(ta)
        
        mu = self.GravitationalParameter
        r = Cartesian(p*sy.cos(ta)/rDenom, p*sy.sin(ta)/rDenom, 0)
        e = self.Eccentricity
        firstPart = sy.sqrt(mu/p)#type:ignore
        v = Cartesian(-1*firstPart *sy.sin(ta), firstPart*(e+sy.cos(ta)),0)
        return MotionCartesian(r, v)

    def ToInertialMotionCartesian(self) -> MotionCartesian:
        """Converts these elements to inertial Cartesian values.

        Returns:
            MotionCartesian: The inertial Cartesian elements.
        """
        motion = self.ToPerifocalCartesian()
        rotMatrix = self.PerifocalToInertialRotationMatrix()
        return MotionCartesian(rotMatrix*motion.Position, rotMatrix*motion.Velocity)

    def ToInertialMotionCartesianOverridingTrueAnomaly(self, trueAnomaly) -> MotionCartesian:
        tempElements = KeplerianElements(self.SemiMajorAxis, self.Eccentricity, self.Inclination, self.ArgumentOfPeriapsis, self.RightAscensionOfAscendingNode, trueAnomaly, self.GravitationalParameter)
        return tempElements.ToInertialMotionCartesian()

    def ToArray(self, deg : Optional[bool] = False) -> List[SymbolOrNumber] :
        radToDeg = 180.0/math.pi
        return [self.SemiMajorAxis, self.Eccentricity, self.Inclination*radToDeg, self.ArgumentOfPeriapsis*radToDeg, self.RightAscensionOfAscendingNode*radToDeg, self.TrueAnomaly*radToDeg]

    def NamesToArray(self) -> List[str]:
        return ["SemiMajorAxis", "Eccentricity", "Inclination", "Argument Of Periapsis", "Right Ascension Of Ascending Node", "TrueAnomaly"]
        
class GaussianEquationsOfMotionVallado :
    def __init__(self, elements : KeplerianElements, accelerationVector : Cartesian) :
        self.Elements = elements
        # Battin page 488
        # note that Battin has f = true anomaly, theta = argument of latitude
        mu = elements.GravitationalParameter
        a = elements.SemiMajorAxis
        b = elements.SemiMinorAxis
        e = elements.Eccentricity
        i = elements.Inclination
        #raan = elements.RightAscensionOfAscendingNode
        #aop = elements.ArgumentOfPeriapsis
        ta = elements.TrueAnomaly
        u = elements.ArgumentOfLatitude

        r = elements.Radius
        b = elements.SemiMinorAxis        
        n = elements.MeanMotion
        h = n*a*b
        p = b**2/a
        cTa = sy.cos(ta)
        sTa = sy.sin(ta)
        cU = sy.cos(u)
        sU = sy.sin(u)

        ar = accelerationVector.X
        ah = accelerationVector.Y # cross track
        at = accelerationVector.Z 
        sqrt1me2 = sy.sqrt(1-e*e)
        self.SemiMajorAxisDot = 2/(n*sqrt1me2)*(e*sTa*ar+(p/r)*at)
        self.EccentricityDot = (sqrt1me2/(n*a))*(sTa*ar+(cTa+(e+cTa)/(1+e*cTa))*at)
        self.InclinationDot = (r*cU/h)*ah
        self.RightAscensionOfAscendingNodeDot = (r*sU/h*sy.sin(i))*ah        
        self.ArgumentOfPeriapsisDot = (sqrt1me2/(n*a*e))*(-cTa*ar+sTa*(1+r/p)*at)-r*sy.cot(i)*sy.sin(u)*ah/h

        # take your pick
        #self.TrueAnomalyDot = h/(r**2)-(1/e*h)*(p*cTa*ar-(p+r)*sTa*aTh)
        self.TrueAnomalyDot = h/(r**2)+(1/(e*h))*(p*cTa*ar-(p+r)*sTa*at)
        self.MeanAnomalyDot = n + b/(a*h*e)*((p*cTa-2*r*e)*ar - (p+r)*sTa*at)
        self.EccentricAnomalyDot = n*a/r + (1/(n*a*e))*((cTa-e)*ar-(1+r/a)*sTa*at)

    def ToSympyMatrixEquation(self) -> sy.Eq:
        lhs = sy.Matrix([[self.Elements.SemiMajorAxis, self.Elements.Eccentricity, self.Elements.Inclination, self.Elements.ArgumentOfPeriapsis, self.Elements.RightAscensionOfAscendingNode, self.Elements.TrueAnomaly]]).transpose()
        rhs = sy.Matrix([[self.SemiMajorAxisDot, self.EccentricityDot, self.InclinationDot, self.ArgumentOfPeriapsisDot, self.RightAscensionOfAscendingNodeDot, self.TrueAnomalyDot]]).transpose()
        return sy.Eq(lhs, rhs)

class GaussianEquationsOfMotion :
    def __init__(self, elements : KeplerianElements, accelerationVector : Cartesian) :
        self.Elements = elements
        # Battin page 488
        # note that Battin has f = true anomaly, theta = argument of latitude
        mu = elements.GravitationalParameter
        a = elements.SemiMajorAxis
        b = elements.SemiMinorAxis
        e = elements.Eccentricity
        i = elements.Inclination
        #raan = elements.RightAscensionOfAscendingNode
        #aop = elements.ArgumentOfPeriapsis
        ta = elements.TrueAnomaly
        u = elements.ArgumentOfLatitude

        r = elements.Radius
        b = elements.SemiMinorAxis        
        n = elements.MeanMotion
        h = n*a*b
        p = b**2/a
        cTa = sy.cos(ta)
        sTa = sy.sin(ta)
        cU = sy.cos(u)
        sU = sy.sin(u)

        ar = accelerationVector.X
        ah = accelerationVector.Y # cross track
        aTh = accelerationVector.Z 

        self.SemiMajorAxisDot = (2.0*(a**2.0)/h) * (e*sTa*ar + (p/r)*aTh)
        self.EccentricityDot = (1/h)*(p*sTa*ar+((p+r)*cTa+r*e)*aTh)
        self.InclinationDot = (r*cU/h)*ah
        self.RightAscensionOfAscendingNodeDot = (r*sU/h*sy.sin(i))*ah        
        self.ArgumentOfPeriapsisDot = (1/(h*e))*(-1*p*cTa*ar+(p+r)*sTa*aTh)-r*sTa*sy.cos(i)*ah/(h*sy.sin(i))                

        # take your pick
        #self.TrueAnomalyDot = h/(r**2)-(1/e*h)*(p*cTa*ar-(p+r)*sTa*aTh)
        self.TrueAnomalyDot = h/(r**2)+(1/(e*h))*(p*cTa*ar-(p+r)*sTa*aTh)
        self.MeanAnomalyDot = n + b/(a*h*e)*((p*cTa-2*r*e)*ar - (p+r)*sTa*aTh)
        self.EccentricAnomalyDot = n*a/r + (1/(n*a*e))*((cTa-e)*ar-(1+r/a)*sTa*aTh)

    def ToSympyMatrixEquation(self) -> sy.Eq:
        lhs = sy.Matrix([[self.Elements.SemiMajorAxis, self.Elements.Eccentricity, self.Elements.Inclination, self.Elements.ArgumentOfPeriapsis, self.Elements.RightAscensionOfAscendingNode, self.Elements.TrueAnomaly]]).transpose()
        rhs = sy.Matrix([[self.SemiMajorAxisDot, self.EccentricityDot, self.InclinationDot, self.ArgumentOfPeriapsisDot, self.RightAscensionOfAscendingNodeDot, self.TrueAnomalyDot]]).transpose()
        return sy.Eq(lhs, rhs)


def CreateSymbolicElements(elementsFunctionOf : Optional[Union[sy.Expr, List[sy.Expr]]] = None, mu : Optional[SymbolOrNumber] = None) -> KeplerianElements :
    """Creates a KeplerianElements structure made of symbols.

    Args:
        elementsFunctionOf (sy.Symbol, optional): If these elements should be a function of something (like time) pass it here. Defaults to None.

    Returns:
        KeplerianElements: Symbolic KeplerianElements.
    """
    if(elementsFunctionOf == None) :
        a = sy.Symbol("a", real=True)
        ecc = sy.Symbol('e', real=True, nonnegative=True)
        inc = sy.Symbol('i', real=True)
        raan = sy.Symbol('\Omega', real=True)
        aop = sy.Symbol('\omega', real=True)
        ta = sy.Symbol(r'\nu', real=True)
    else:
        if isinstance(elementsFunctionOf, sy.Expr) :
            elementsFunctionOf = [elementsFunctionOf]

        a = sy.Function("a", real=True)(*elementsFunctionOf)
        ecc = sy.Function('e', nonnegative=True)(*elementsFunctionOf)
        inc = sy.Function('i', real=True)(*elementsFunctionOf)
        raan = sy.Function('\Omega', real=True)(*elementsFunctionOf)
        aop = sy.Function('\omega', real=True)(*elementsFunctionOf)
        ta = sy.Function(r'\nu', real=True)(*elementsFunctionOf)
    if mu is None :
        mu = sy.Symbol('\mu', positive=True, real=True)
    return KeplerianElements(a, ecc, inc, aop, raan, ta, mu)
