
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian #type: ignore
import math
import sympy as sy

def testMagnitude():
    testingCart = Cartesian(3.0, 4.0, 0.0)
    assert 5 == testingCart.Magnitude()

def testBasicOperations() :
    aCart = Cartesian(1.0,2.0,3.0)
    anotherCart = Cartesian(4,5,6)

    expectedAdd = Cartesian(5,7,9)
    expectedSubtract = Cartesian(-3,-3,-3)
    expectedMagnitude = math.sqrt(1+4+9)
    expectedNorm = Cartesian(1.0/expectedMagnitude, 2.0/expectedMagnitude, 3.0/expectedMagnitude)
    expectedDot = 4+10+18
    expectedCross = Cartesian(12-15, 12-6, 5-8)
    assert aCart+anotherCart== expectedAdd, "add"
    assert aCart-anotherCart== expectedSubtract, "subtract"
    assert aCart.Magnitude()== expectedMagnitude, "mag"
    assert aCart.Normalize()== expectedNorm, "norm"
    assert aCart.dot(anotherCart)== expectedDot, "dot"
    assert aCart.cross(anotherCart)== expectedCross, "cross"

    assert aCart.X== 1, "x element"
    assert aCart.Y== 2, "y element"
    assert aCart.Z== 3, "z element"

    assert aCart[0]== 1, "first element"
    assert aCart[1]== 2, "second element"
    assert aCart[2]== 3, "third element"

def testEquality() :
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

    assert aCart == aCart, "identical"
    assert aCart == same, "same"
    assert not aCart == notACart, "not a cart"
    assert not aCart == diffX, "diffX"
    assert not aCart == diffY, "diffY"
    assert not aCart == diffZ, "diffZ"
    assert aCart == floats, "floats"
    assert not aCart == trans, "transpose"
    assert aCart == sameMat, "same Matrix"
    assert not aCart == diffMat, "diffMat"

def testEqualsWithinTolerance() :
    aCart = Cartesian(1,2,3)
    within1X = Cartesian(2,2,3)
    within1Y = Cartesian(1,3,3)
    within1Z = Cartesian(1,2,4)

    assert aCart.EqualsWithinTolerance(aCart, 0), ""

    assert aCart.EqualsWithinTolerance(within1X, 2), "X similar, large tolerance"
    assert not aCart.EqualsWithinTolerance(within1X, 0.5), "X similar, small tolerance"
    assert aCart.EqualsWithinTolerance(within1Y, 2), "Y similar, large tolerance"
    assert not aCart.EqualsWithinTolerance(within1Y, 0.5), "Y similar, small tolerance"
    assert aCart.EqualsWithinTolerance(within1Z, 2), "Z similar, large tolerance"
    assert not aCart.EqualsWithinTolerance(within1Z, 0.5), "Z similar, small tolerance"


#class testMotionCartesian(unittest.TestCase) :
def testBasics() :
    motion = MotionCartesian(Cartesian(1,2,3), Cartesian(4,5,6))
    assert Cartesian(1,2,3)== motion.Position, "pos"
    assert Cartesian(4,5,6)== motion.Velocity, "vel"
    assert 1== motion.Order, "order with vel"
    assert motion[0]== motion.Position, "position indexing"
    assert motion[1]== motion.Velocity, "velocity indexing"

def testNoVelocity() :
    motion = MotionCartesian(Cartesian(1,2,3), None)
    assert Cartesian(1,2,3)== motion.Position, "pos only"
    assert motion.Velocity is None, "vel null"
    assert 0== motion.Order, "order without vel"
    assert motion[0]== motion.Position, "position indexing"
    assert motion[1]== None, "velocity indexing"  

def testMotionEqualsWithinTolerance() :
    motion = MotionCartesian(Cartesian(1,2,3), Cartesian(4,5,6))
    asFloat = MotionCartesian(Cartesian(1.0,2.0,3.0), Cartesian(4.0,5.0,6.0))
    almostSame =  MotionCartesian(Cartesian(1.1,2.1,3.1), Cartesian(4.01,5.01,6.01))
    assert motion.EqualsWithinTolerance(asFloat, 0.000001, 0.000001), "float and int match"
    assert asFloat.EqualsWithinTolerance(almostSame, 0.2, 0.2), "within tolerance"
    assert not asFloat.EqualsWithinTolerance(almostSame, 0.0, 0.2), "pos not in tolerance"
    assert not asFloat.EqualsWithinTolerance(almostSame, 0.2, 0.0002), "vel not in tolerance"
