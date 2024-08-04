import pyeq2orb.Coordinates.KeplerianModule as Keplerian #type: ignore
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian #type: ignore
import pytest
import pyeq2orb.Coordinates.RotationMatrix as RotationMatrix #type: ignore
import sympy as sy
import math as math
from pyeq2orb.Utilities.utilitiesForTest import assertAlmostEquals #type: ignore

def testCreatingBasicElements() :
    elements = Keplerian.CreateSymbolicElements()
    assert "a"== str(elements.SemiMajorAxis), "sma"
    assert "e"== str(elements.Eccentricity), "ecc"
    assert "i"== str(elements.Inclination), "inc"
    assert "\omega"== str(elements.ArgumentOfPeriapsis), "aop"
    assert '\Omega'== str(elements.RightAscensionOfAscendingNode), "raan"
    assert r'\nu'== str(elements.TrueAnomaly), "ta"
    assert '\mu'== str(elements.GravitationalParameter), "mu"

    # and again of T
    elements = Keplerian.CreateSymbolicElements(sy.Symbol('t'))
    assert "a(t)"== str(elements.SemiMajorAxis), "sma of t"
    assert "e(t)"== str(elements.Eccentricity), "ecc of t"
    assert "i(t)"== str(elements.Inclination), "inc of t"
    assert "\omega(t)"== str(elements.ArgumentOfPeriapsis), "aop of t"
    assert '\Omega(t)'== str(elements.RightAscensionOfAscendingNode), "raan of t"
    assert r'\nu(t)'== str(elements.TrueAnomaly), "ta of t"
    assert '\mu'== str(elements.GravitationalParameter), "mu of t"      

def testFromCartesian() :
    # page 114 - 116 of Vallado 4th edition
    # units are km and seconds
    theMotion = MotionCartesian(Cartesian(6524.834, 6862.875, 6448.296), Cartesian(4.901327, 5.533756, -1.976341))
    mu = 398600.4418 

    keplerianElements = Keplerian.KeplerianElements.FromMotionCartesian(theMotion, mu)
    assertAlmostEquals(36127.343, keplerianElements.SemiMajorAxis, 1, "a")
    assertAlmostEquals(0.832853, keplerianElements.Eccentricity, 6, "e")
    assertAlmostEquals(87.870, keplerianElements.Inclination * 180.0/math.pi, 2, "i")
    assertAlmostEquals(227.898, keplerianElements.RightAscensionOfAscendingNode * 180.0/math.pi, 3, "raan")
    assertAlmostEquals(53.38, keplerianElements.ArgumentOfPeriapsis * 180.0/math.pi, 2, "aop")
    assertAlmostEquals(92.335, keplerianElements.TrueAnomaly * 180.0/math.pi, 3, "ta")
    assertAlmostEquals(mu, keplerianElements.GravitationalParameter, 9, "mu")

    # this next call is just making sure nothing throws when using symbols.  We know the algorithm is good from the above checks
    theMotion = MotionCartesian(Cartesian(sy.Symbol('x'), sy.Symbol('y'), sy.Symbol('z')), Cartesian(sy.Symbol('v_x'), sy.Symbol('v_y'), sy.Symbol('v_z')))
    keplerianElements = Keplerian.KeplerianElements.FromMotionCartesian(theMotion, mu)

def testToCartesian() :
    # page 114 - 116 of Vallado 4th edition, going backwards though
    # units are km and seconds
    degToRad = math.pi/180.0
    mu = 398600.4418 

    theElements = Keplerian.KeplerianElements(36127.343, 0.832853, 87.870*degToRad, 53.38*degToRad, 227.898*degToRad, 92.335*degToRad, mu)
    theMotion = MotionCartesian(Cartesian(6524.834, 6862.875, 6448.296), Cartesian(4.901327, 5.533756, -1.976341))
    convertedMotion = theElements.ToInertialMotionCartesian()
    assert theMotion.EqualsWithinTolerance(convertedMotion, 1.0, 0.01)

    # this next call is just making sure nothing throws when using symbols.  We know the algorithm is good from the above checks
    theElements = Keplerian.CreateSymbolicElements()
    convertedMotion = theElements.ToInertialMotionCartesian()

def testAnomalyCalculations() :
    ma = 235.4*math.pi/180.0
    ecc = 0.4
    ea = 220.512074767522*math.pi/180.0
    ta = (360 - 152.836008230786)*math.pi/180.0
    assertAlmostEquals(ea, Keplerian.EccentricAnomalyFromMeanAnomaly(ma, ecc), 9)
    assertAlmostEquals(ma, Keplerian.MeanAnomalyFromEccentricAnomaly(ea, ecc), 9)
    assertAlmostEquals(ta, Keplerian.TrueAnomalyFromEccentricAnomaly(ea, ecc), 9)
    assertAlmostEquals(ea, Keplerian.EccentricAnomalyFromTrueAnomaly(ta, ecc), 9)
    assertAlmostEquals(ta, Keplerian.TrueAnomalyFromMeanAnomaly(ma, ecc), 9)
    assertAlmostEquals(ma, Keplerian.MeanAnomalyFromTrueAnomaly(ta, ecc), 9)        


 