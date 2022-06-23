import sympy as sy
from pyeq2orb.Symbolics.Vectors import Vector
import unittest

class testVectors(unittest.TestCase) :
    def testFromArray(self) :
        vec = Vector.fromArray([1,2,3])
        assert vec[0] == 1
        assert vec[1] == 2
        assert vec[2] == 3

        assert len(vec) == 3

    def testFromValues(self) :
        vec = Vector.zeros(3)
        assert vec[0] == 0
        assert vec[1] == 0
        assert vec[2] == 0
        assert len(vec) == 3

    def testSetting(self) :
        vec = Vector.zeros(3)
        vec[0] = 1
        vec[1] = 2
        vec[2] = 3
        assert vec[0] == 1
        assert vec[1] == 2
        assert vec[2] == 3
        assert len(vec) == 3

    def testToArray(self) :
        original = Vector.fromArray([1,2,3])
        vec = original.toArray()
        assert vec[0] == 1
        assert vec[1] == 2
        assert vec[2] == 3

    def testMagnitude(self) :
        vec = Vector.fromValues(3.0, 4.0)
        self.assertEqual(5, vec.Magnitude())

    def testZeros(self) :
        vec = Vector.zeros(5)
        assert len(vec) == 5
        for i in range(0, 5) :
            assert 0 == vec[i]
        