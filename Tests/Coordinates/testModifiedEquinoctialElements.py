import pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule as mee
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
import unittest
import pyeq2orb.Coordinates.RotationMatrix as RotationMatrix
import sympy as sy
import math as math

class testModifiedEquinoctialElements(unittest.TestCase) :

    def testToCartesian(self):
        # https://ai-solutions.com/_help_Files/orbit_element_types.htm
        equiElements = mee.ModifiedEquinoctialElements(7070766.0, 0.00180, -0.00170, 0.610, -0.980, 136.64*math.pi/180.0, 3.986004418e14) 
        expectedCart = MotionCartesian(Cartesian(-3410.673, 5950.957, -1788.627)*1000, Cartesian(1.893, -1.071, -7.176)*1000)
        actualCart = equiElements.ToMotionCartesian()
        self.assertAlmostEqual(expectedCart.Position.X, actualCart.Position.X, 1, msg="X")
        self.assertAlmostEqual(expectedCart.Position.Y, actualCart.Position.Y, 1, msg="Y")
        self.assertAlmostEqual(expectedCart.Position.Z, actualCart.Position.Z, 1, msg="Z")

        self.assertAlmostEqual(expectedCart.Velocity.X, actualCart.Velocity.X, 5, msg="v_X")
        self.assertAlmostEqual(expectedCart.Velocity.Y, actualCart.Velocity.Y, 5, msg="v_Y")
        self.assertAlmostEqual(expectedCart.Velocity.Z, actualCart.Velocity.Z, 5, msg="v_Z")       

    def testFromCartesian(self) :
        # https://ai-solutions.com/_help_Files/orbit_element_types.htm
        expectedEquiElements = mee.ModifiedEquinoctialElements(7070766.0, 0.00180, -0.00170, 0.610, -0.980, 136.64*math.pi/180.0, 3.986004418e14) 
        cartMotion = MotionCartesian(Cartesian(-3410.673, 5950.957, -1788.627)*1000, Cartesian(1.893, -1.071, -7.176)*1000)
        actualEqui = mee.ModifiedEquinoctialElements.FromMotionCartesian(cartMotion, 3.986004418e14)
        self.assertAlmostEqual(expectedEquiElements.SemiParameter, actualEqui.SemiParameter, 1, msg="p")
        self.assertAlmostEqual(expectedEquiElements.EccentricityCosTermF, actualEqui.EccentricityCosTermF, 5, msg="f")
        self.assertAlmostEqual(expectedEquiElements.EccentricitySinTermG, actualEqui.EccentricitySinTermG, 5, msg="g")
        self.assertAlmostEqual(expectedEquiElements.InclinationCosTermH, actualEqui.InclinationCosTermH, 5, msg="h")
        self.assertAlmostEqual(expectedEquiElements.InclinationSinTermK, actualEqui.InclinationSinTermK, 5, msg="k")
        self.assertAlmostEqual(expectedEquiElements.TrueLongitude, actualEqui.TrueLongitude, 5, msg="L")


    def testRoundTripFromCartesian(self) :
        # https://ai-solutions.com/_help_Files/orbit_element_types.htm
        expectedEquiElements = mee.ModifiedEquinoctialElements(7070766.0, 0.00180, -0.00170, 0.610, -0.980, 136.64*math.pi/180.0, 3.986004418e14) 
        toCart = expectedEquiElements.ToMotionCartesian()
        actualEqui = mee.ModifiedEquinoctialElements.FromMotionCartesian(toCart, expectedEquiElements.GravitationalParameter)
        self.assertAlmostEqual(expectedEquiElements.SemiParameter, actualEqui.SemiParameter, 4, msg="p")
        self.assertAlmostEqual(expectedEquiElements.EccentricityCosTermF, actualEqui.EccentricityCosTermF, 9, msg="f")
        self.assertAlmostEqual(expectedEquiElements.EccentricitySinTermG, actualEqui.EccentricitySinTermG, 9, msg="g")
        self.assertAlmostEqual(expectedEquiElements.InclinationCosTermH, actualEqui.InclinationCosTermH, 9, msg="h")
        self.assertAlmostEqual(expectedEquiElements.InclinationSinTermK, actualEqui.InclinationSinTermK, 9, msg="k")
        self.assertAlmostEqual(expectedEquiElements.TrueLongitude, actualEqui.TrueLongitude% (math.pi*2), 9, msg="L")
