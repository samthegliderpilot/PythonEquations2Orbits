import contextlib
import pyeq2orb.Coordinates.KeplerianModule as Keplerian
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
import unittest
import pyeq2orb.Coordinates.RotationMatrix as RotationMatrix
import sympy as sy
import math as math

class testKeplerianElements(unittest.TestCase) :
    def testCreatingBasicElements(self) :
        elements = Keplerian.CreateSymbolicElements()
        self.assertEquals("a", str(elements.SemiMajorAxis), msg="sma")
        self.assertEquals("e", str(elements.Eccentricity), msg="ecc")
        self.assertEquals("i", str(elements.Inclination), msg="inc")
        self.assertEquals("\omega", str(elements.ArgumentOfPeriapsis), msg="aop")
        self.assertEquals('\Omega', str(elements.RightAscensionOfAscendingNode), msg="raan")
        self.assertEquals(r'\nu', str(elements.TrueAnomaly), msg="ta")
        self.assertEquals('\mu', str(elements.GravitationalParameter), msg="mu")

        # and again of T
        elements = Keplerian.CreateSymbolicElements(sy.Symbol('t'))
        self.assertEquals("a(t)", str(elements.SemiMajorAxis), msg="sma of t")
        self.assertEquals("e(t)", str(elements.Eccentricity), msg="ecc of t")
        self.assertEquals("i(t)", str(elements.Inclination), msg="inc of t")
        self.assertEquals("\omega(t)", str(elements.ArgumentOfPeriapsis), msg="aop of t")
        self.assertEquals('\Omega(t)', str(elements.RightAscensionOfAscendingNode), msg="raan of t")
        self.assertEquals(r'\nu(t)', str(elements.TrueAnomaly), msg="ta of t")
        self.assertEquals('\mu', str(elements.GravitationalParameter), msg="mu of t")        

    def testFromCartesian(self) :
        # page 114 - 116 of Vallado 4th edition
        # units are km and seconds
        theMotion = MotionCartesian(Cartesian(6524.834, 6862.875, 6448.296), Cartesian(4.901327, 5.533756, -1.976341))
        mu = 398600.4418 

        keplerianElements = Keplerian.KeplerianElements.FromMotionCartesian(theMotion, mu)
        self.assertAlmostEquals(36127.343, keplerianElements.SemiMajorAxis, 1, msg="a")
        self.assertAlmostEquals(0.832853, keplerianElements.Eccentricity, 6, msg="e")
        self.assertAlmostEquals(87.870, keplerianElements.Inclination * 180.0/math.pi, 2, msg="i")
        self.assertAlmostEquals(227.898, keplerianElements.RightAscensionOfAscendingNode * 180.0/math.pi, 3, msg="raan")
        self.assertAlmostEquals(53.38, keplerianElements.ArgumentOfPeriapsis * 180.0/math.pi, 2, msg="aop")
        self.assertAlmostEquals(92.335, keplerianElements.TrueAnomaly * 180.0/math.pi, 3, msg="ta")
        self.assertAlmostEquals(mu, keplerianElements.GravitationalParameter, msg="mu")

        # this next call is just making sure nothing throws when using symbols.  We know the algorithm is good from the above checks
        theMotion = MotionCartesian(Cartesian(sy.Symbol('x'), sy.Symbol('y'), sy.Symbol('z')), Cartesian(sy.Symbol('v_x'), sy.Symbol('v_y'), sy.Symbol('v_z')))
        keplerianElements = Keplerian.KeplerianElements.FromMotionCartesian(theMotion, mu)

    def testToCartesian(self) :
        # page 114 - 116 of Vallado 4th edition, going backwards though
        # units are km and seconds
        degToRad = math.pi/180.0
        mu = 398600.4418 

        theElements = Keplerian.KeplerianElements(36127.343, 0.832853, 87.870*degToRad, 53.38*degToRad, 227.898*degToRad, 92.335*degToRad, mu)
        theMotion = MotionCartesian(Cartesian(6524.834, 6862.875, 6448.296), Cartesian(4.901327, 5.533756, -1.976341))
        convertedMotion = theElements.ToInertialMotionCartesian()
        self.assertTrue(theMotion.EqualsWithinTolerance(convertedMotion, 0.1, 0.001))


    
