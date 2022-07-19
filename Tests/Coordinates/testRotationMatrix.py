import unittest
import pyeq2orb.Coordinates.RotationMatrix as RotationMatrix
import sympy as sy
import math as math
class testRotationMatrix(unittest.TestCase) :
    def testAboutX(self) :
        angle = 30.0*math.pi/180.0
        sinAngle = math.sin(angle)
        cosAngle = math.cos(angle)
        expected = sy.Matrix([[1, 0, 0], [0, cosAngle, -1*sinAngle], [0, sinAngle, cosAngle]])
        self.assertEqual(expected, RotationMatrix.RotAboutX(angle))

    def testAboutY(self) :
        angle = 30.0*math.pi/180.0
        sinAngle = math.sin(angle)
        cosAngle = math.cos(angle)
        expected = sy.Matrix([[cosAngle, 0, sinAngle], [0, 1, 0], [-1*sinAngle, 0, cosAngle]])
        self.assertEqual(expected, RotationMatrix.RotAboutY(angle))

    def testAboutZ(self) :
        angle = 30.0*math.pi/180.0
        sinAngle = math.sin(angle)
        cosAngle = math.cos(angle)
        expected = sy.Matrix([[cosAngle, -1*sinAngle, 0], [sinAngle, cosAngle, 0], [0, 0, 1]])
        self.assertEqual(expected, RotationMatrix.RotAboutZ(angle))               