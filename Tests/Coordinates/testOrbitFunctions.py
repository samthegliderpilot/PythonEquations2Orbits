import pyeq2orb.Coordinates.KeplerianModule as Keplerian
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
import unittest
import pyeq2orb.Coordinates.RotationMatrix as RotationMatrix
import pyeq2orb.Coordinates.OrbitFunctions as orb
import sympy as sy
import math as math


class testOrbitFunctions(unittest.TestCase) :
    def testRicAxes(self) :
        rad = Cartesian(-1.0, 1.0, 0.0).Normalize()
        vel = Cartesian(-1.0, -1.0, 0.0).Normalize()
        motion = MotionCartesian(rad, vel)

        ricToInert = orb.CreateComplicatedRicToInertialMatrix(motion)
        expected = sy.Matrix([[rad.X, rad.Y, rad.Z], [vel.X, vel.Y, vel.Z], [0.0, 0.0, 1.0]]).transpose()
        self.assertAlmostEqual(expected[0,0], ricToInert[0,0], places=10, msg="0,0")
        self.assertAlmostEqual(expected[0,1], ricToInert[0,1], places=10, msg="0,1")
        self.assertAlmostEqual(expected[0,2], ricToInert[0,2], places=10, msg="0,2")
        self.assertAlmostEqual(expected[1,0], ricToInert[1,0], places=10, msg="1,0")
        self.assertAlmostEqual(expected[1,1], ricToInert[1,1], places=10, msg="1,1")
        self.assertAlmostEqual(expected[1,2], ricToInert[1,2], places=10, msg="1,2")
        self.assertAlmostEqual(expected[2,0], ricToInert[2,0], places=10, msg="2,0")
        self.assertAlmostEqual(expected[2,1], ricToInert[2,1], places=10, msg="2,1")
        self.assertAlmostEqual(expected[2,2], ricToInert[2,2], places=10, msg="2,2")
        ricDir = Cartesian(1.0, 1.0, 0.0).Normalize()
        expectedInertDirection = Cartesian(-1.0, 0.0, 0.0)
        print(ricToInert)
        calculatedInertial =  ricToInert*ricDir
        
        self.assertAlmostEqual(expectedInertDirection[0], calculatedInertial[0], places=10, msg="converted 0,0 " + str(expectedInertDirection) + " but was " + str(calculatedInertial))
        self.assertAlmostEqual(expectedInertDirection[1], calculatedInertial[1], places=10, msg="converted 0,1 " + str(expectedInertDirection) + " but was " + str(calculatedInertial))
        self.assertAlmostEqual(expectedInertDirection[2], calculatedInertial[2], places=10, msg="converted 0,2 " + str(expectedInertDirection) + " but was " + str(calculatedInertial))





