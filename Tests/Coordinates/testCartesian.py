import unittest
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
import math
import sympy as sy

class testCartesian(unittest.TestCase) :
    def testMagnitude(self):
        testingCart = Cartesian(3.0, 4.0, 0.0)
        self.assertEquals(5,  testingCart.Magnitude())
    
    def testBasicOperations(self) :
        aCart = Cartesian(1.0,2.0,3.0)
        anotherCart = Cartesian(4,5,6)

        expectedAdd = Cartesian(5,7,9)
        expectedSubtract = Cartesian(-3,-3,-3)
        expectedMagnitutde = math.sqrt(1+4+9)
        expectedNorm = Cartesian(1.0/expectedMagnitutde, 2.0/expectedMagnitutde, 3.0/expectedMagnitutde)
        expectedDot = 4+10+18
        expectedCross = Cartesian(12-15, 12-6, 5-8)
        self.assertEqual(aCart+anotherCart, expectedAdd, msg="add")
        self.assertEqual(aCart-anotherCart, expectedSubtract, msg="subtract")
        self.assertEqual(aCart.Magnitude(), expectedMagnitutde, msg="mag")
        self.assertEqual(aCart.Normalize(), expectedNorm, msg="norm")
        self.assertEqual(aCart.dot(anotherCart), expectedDot, msg="dot")
        self.assertEqual(aCart.cross(anotherCart), expectedCross, msg="cross")

        self.assertEqual(aCart.X, 1, msg="x element")
        self.assertEqual(aCart.Y, 2, msg="y element")
        self.assertEqual(aCart.Z, 3, msg="z element")

        self.assertEqual(aCart[0], 1, msg="first element")
        self.assertEqual(aCart[1], 2, msg="second element")
        self.assertEqual(aCart[2], 3, msg="third element")

    def testEquality(self) :
        aCart = Cartesian(1,2,3)
        same = Cartesian(1,2,3)
        notACart = "Cartesian(if, I try, hard)"
        diffX = Cartesian(2,2,3)
        diffY = Cartesian(1,3,3)
        diffZ = Cartesian(1,2,2)
        floats = Cartesian(1.0, 2.0, 3.0)
        trans = aCart.transpose()
        sameMat = sy.Matrix([[1],[2],[3]])
        diffMat = sy.Matrix([[1],[2],[3], [4]])

        self.assertTrue(aCart == aCart, msg="identical")
        self.assertTrue(aCart == same, msg="same")
        self.assertFalse(aCart == notACart, msg="not a cart")
        self.assertFalse(aCart == diffX, msg="diffX")
        self.assertFalse(aCart == diffY, msg="diffY")
        self.assertFalse(aCart == diffZ, msg="diffZ")
        self.assertTrue(aCart == floats, msg="floats")
        self.assertFalse(aCart == trans, msg="transpose")
        self.assertTrue(aCart == sameMat, msg="same Matrix")
        self.assertFalse(aCart == diffMat, msg="diffMat")

    def testEqualsWithinTolerance(self) :
        aCart = Cartesian(1,2,3)
        within1X = Cartesian(2,2,3)
        within1Y = Cartesian(1,3,3)
        within1Z = Cartesian(1,2,4)

        self.assertTrue(aCart.EqualsWithinTolerance(aCart, 0), msg="self")

        self.assertTrue(aCart.EqualsWithinTolerance(within1X, 2), msg="X similar, large tolerance")
        self.assertFalse(aCart.EqualsWithinTolerance(within1X, 0.5), msg="X similar, small tolerance")
        self.assertTrue(aCart.EqualsWithinTolerance(within1Y, 2), msg="Y similar, large tolerance")
        self.assertFalse(aCart.EqualsWithinTolerance(within1Y, 0.5), msg="Y similar, small tolerance")
        self.assertTrue(aCart.EqualsWithinTolerance(within1Z, 2), msg="Z similar, large tolerance")
        self.assertFalse(aCart.EqualsWithinTolerance(within1Z, 0.5), msg="Z similar, small tolerance")


class testMotionCartesian(unittest.TestCase) :
    def testBasics(self) :
        motion = MotionCartesian(Cartesian(1,2,3), Cartesian(4,5,6))
        self.assertEqual(Cartesian(1,2,3), motion.Position, msg="pos")
        self.assertEqual(Cartesian(4,5,6), motion.Velocity, msg="vel")
        self.assertEqual(1, motion.Order, msg="order with vel")
        self.assertEqual(motion[0], motion.Position, msg="position indexing")
        self.assertEqual(motion[1], motion.Velocity, msg="velocity indexing")

    def testNoVelocity(self) :
        motion = MotionCartesian(Cartesian(1,2,3), None)
        self.assertEqual(Cartesian(1,2,3), motion.Position, msg="pos only")
        self.assertEqual(None, motion.Velocity, msg="vel null")
        self.assertEqual(0, motion.Order, msg="order without vel")
        self.assertEqual(motion[0], motion.Position, msg="position indexing")
        self.assertEqual(motion[1], None, msg="velocity indexing")        




