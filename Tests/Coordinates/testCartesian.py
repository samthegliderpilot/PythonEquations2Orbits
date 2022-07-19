import unittest
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
import math
class testCartesian(unittest.TestCase) :
    def testMagnitude(self):
        testingCart = Cartesian(3, 4, 0)
        self.assertEquals(5,  testingCart.Magnitude())
    
    def testBasicOperations(self) :
        aCart = Cartesian(1,2,3)
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
        self.assertEqual(aCart.Dot(anotherCart), expectedDot, msg="dot")
        self.assertEqual(aCart.Cross(anotherCart), expectedCross, msg="cross")

        self.assertEqual(aCart.X, 1, msg="x element")
        self.assertEqual(aCart.Y, 2, msg="y element")
        self.assertEqual(aCart.Z, 3, msg="z element")

        self.assertEqual(aCart[0], 1, msg="first element")
        self.assertEqual(aCart[1], 2, msg="second element")
        self.assertEqual(aCart[2], 3, msg="third element")

        self.assertEqual(aCart[0,0], 1, msg="first column element")
        self.assertEqual(aCart[0,1], 2, msg="second column element")
        self.assertEqual(aCart[0,2], 3, msg="third column element")

    def testEquality(self) :
        aCart = Cartesian(1,2,3)
        same = Cartesian(1,2,3)
        notACart = "Cartesian(if, I try, hard)"
        diffX = Cartesian(2,2,3)
        diffY = Cartesian(1,3,3)
        diffZ = Cartesian(1,2,2)
        floats = Cartesian(1.0, 2.0, 3.0)

        self.assertTrue(aCart == aCart, msg="identical")
        self.assertTrue(aCart == same, msg="same")
        self.assertFalse(aCart == notACart, msg="not a cart")
        self.assertFalse(aCart == diffX, msg="diffX")
        self.assertFalse(aCart == diffY, msg="diffY")
        self.assertFalse(aCart == diffZ, msg="diffZ")
        self.assertTrue(aCart == floats, msg="floats")

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




