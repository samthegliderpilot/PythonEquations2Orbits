import pyeq2orb.Coordinates.KeplerianModule as Keplerian #type: ignore
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian #type: ignore
import pytest
import pyeq2orb.Coordinates.RotationMatrix as RotationMatrix #type: ignore
import pyeq2orb.Coordinates.OrbitFunctions as orb #type: ignore
import sympy as sy
import math as math
from pyeq2orb.Utilities.utilitiesForTest import assertAlmostEquals #type: ignore

def testRicAxes() :
    rad = Cartesian(-1.0, 1.0, 0.0).Normalize()
    vel = Cartesian(-1.0, -1.0, 0.0).Normalize()
    motion = MotionCartesian(rad, vel)

    ricToInert = orb.CreateComplicatedRicToInertialMatrix(motion)
    expected = sy.Matrix([[rad.X, rad.Y, rad.Z], [vel.X, vel.Y, vel.Z], [0.0, 0.0, 1.0]]).transpose()
    assertAlmostEquals(expected[0,0], ricToInert[0,0], places=10, msg="0,0")
    assertAlmostEquals(expected[0,1], ricToInert[0,1], places=10, msg="0,1")
    assertAlmostEquals(expected[0,2], ricToInert[0,2], places=10, msg="0,2")
    assertAlmostEquals(expected[1,0], ricToInert[1,0], places=10, msg="1,0")
    assertAlmostEquals(expected[1,1], ricToInert[1,1], places=10, msg="1,1")
    assertAlmostEquals(expected[1,2], ricToInert[1,2], places=10, msg="1,2")
    assertAlmostEquals(expected[2,0], ricToInert[2,0], places=10, msg="2,0")
    assertAlmostEquals(expected[2,1], ricToInert[2,1], places=10, msg="2,1")
    assertAlmostEquals(expected[2,2], ricToInert[2,2], places=10, msg="2,2")
    ricDir = Cartesian(1.0, 1.0, 0.0).Normalize()
    expectedInertDirection = Cartesian(-1.0, 0.0, 0.0)
    
    calculatedInertial =  ricToInert*ricDir
    
    assertAlmostEquals(expectedInertDirection[0], calculatedInertial[0], places=10, msg="converted 0,0 " + str(expectedInertDirection) + " but was " + str(calculatedInertial))
    assertAlmostEquals(expectedInertDirection[1], calculatedInertial[1], places=10, msg="converted 0,1 " + str(expectedInertDirection) + " but was " + str(calculatedInertial))
    assertAlmostEquals(expectedInertDirection[2], calculatedInertial[2], places=10, msg="converted 0,2 " + str(expectedInertDirection) + " but was " + str(calculatedInertial))





