import pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule as mee #type: ignore
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian #type: ignore
import pytest 
import pyeq2orb.Coordinates.RotationMatrix as RotationMatrix #type: ignore
import sympy as sy
import math as math
from pyeq2orb.Utilities.utilitiesForTest import assertAlmostEquals #type: ignore

def testToCartesian():
    # https://ai-solutions.com/_help_Files/orbit_element_types.htm
    equiElements = mee.ModifiedEquinoctialElements(7070766.0, 0.00180, -0.00170, 0.610, -0.980, 136.64*math.pi/180.0, 3.986004418e14) 
    expectedCart = MotionCartesian(Cartesian(-3404.2252, 5955.58667, -1785.0657)*1000, Cartesian(1.88356358310406, -1.07216265087468, -7.16916147012584)*1000)
    actualCart = equiElements.ToMotionCartesian()
    assertAlmostEquals(expectedCart.Position.X, actualCart.Position.X, 1, msg="X")
    assertAlmostEquals(expectedCart.Position.Y, actualCart.Position.Y, 1, msg="Y")
    assertAlmostEquals(expectedCart.Position.Z, actualCart.Position.Z, 1, msg="Z")

    assertAlmostEquals(expectedCart.Velocity.X, actualCart.Velocity.X, 5, msg="v_X")
    assertAlmostEquals(expectedCart.Velocity.Y, actualCart.Velocity.Y, 5, msg="v_Y")
    assertAlmostEquals(expectedCart.Velocity.Z, actualCart.Velocity.Z, 5, msg="v_Z")       

def testFromCartesian() :
    # https://ai-solutions.com/_help_Files/orbit_element_types.htm
    expectedEquiElements = mee.ModifiedEquinoctialElements(7070766.0, 0.00180, -0.00170, 0.610, -0.980, 136.64*math.pi/180.0, 3.986004418e14) 
    cartMotion = MotionCartesian(Cartesian(-3404.2252, 5955.58667, -1785.0657)*1000, Cartesian(1.88356358310406, -1.07216265087468, -7.16916147012584)*1000)
    actualEqui = mee.ModifiedEquinoctialElements.FromMotionCartesian(cartMotion, 3.986004418e14)
    assertAlmostEquals(expectedEquiElements.SemiParameter, actualEqui.SemiParameter, 1, msg="p")
    assertAlmostEquals(expectedEquiElements.EccentricityCosTermF, actualEqui.EccentricityCosTermF, 5, msg="f")
    assertAlmostEquals(expectedEquiElements.EccentricitySinTermG, actualEqui.EccentricitySinTermG, 5, msg="g")
    assertAlmostEquals(expectedEquiElements.InclinationCosTermH, actualEqui.InclinationCosTermH, 5, msg="h")
    assertAlmostEquals(expectedEquiElements.InclinationSinTermK, actualEqui.InclinationSinTermK, 5, msg="k")
    assertAlmostEquals(expectedEquiElements.TrueLongitude, actualEqui.TrueLongitude% (2*math.pi), 5, msg="L")


def testRoundTripFromCartesian() :
    # https://ai-solutions.com/_help_Files/orbit_element_types.htm
    expectedEquiElements = mee.ModifiedEquinoctialElements(7070766.0, 0.00180, -0.00170, 0.610, -0.980, 136.64*math.pi/180.0, 3.986004418e14) 
    toCart = expectedEquiElements.ToMotionCartesian()
    actualEqui = mee.ModifiedEquinoctialElements.FromMotionCartesian(toCart, expectedEquiElements.GravitationalParameter)
    assertAlmostEquals(expectedEquiElements.SemiParameter, actualEqui.SemiParameter, 4, msg="p")
    assertAlmostEquals(expectedEquiElements.EccentricityCosTermF, actualEqui.EccentricityCosTermF, 9, msg="f")
    assertAlmostEquals(expectedEquiElements.EccentricitySinTermG, actualEqui.EccentricitySinTermG, 9, msg="g")
    assertAlmostEquals(expectedEquiElements.InclinationCosTermH, actualEqui.InclinationCosTermH, 9, msg="h")
    assertAlmostEquals(expectedEquiElements.InclinationSinTermK, actualEqui.InclinationSinTermK, 9, msg="k")
    assertAlmostEquals(expectedEquiElements.TrueLongitude, actualEqui.TrueLongitude% (math.pi*2), 9, msg="L")
