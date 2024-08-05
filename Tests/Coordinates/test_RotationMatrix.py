import pytest
import pyeq2orb.Coordinates.RotationMatrix as RotationMatrix #type: ignore
import sympy as sy
import math as math

def test_AboutX() :
    angle = 30.0*math.pi/180.0
    sinAngle = math.sin(angle)
    cosAngle = math.cos(angle)
    expected = sy.Matrix([[1, 0, 0], [0, cosAngle, -1*sinAngle], [0, sinAngle, cosAngle]])
    assert expected == RotationMatrix.RotAboutX(angle)

def test_AboutY() :
    angle = 30.0*math.pi/180.0
    sinAngle = math.sin(angle)
    cosAngle = math.cos(angle)
    expected = sy.Matrix([[cosAngle, 0, sinAngle], [0, 1, 0], [-1*sinAngle, 0, cosAngle]])
    assert expected == RotationMatrix.RotAboutY(angle)

def test_AboutZ() :
    angle = 30.0*math.pi/180.0
    sinAngle = math.sin(angle)
    cosAngle = math.cos(angle)
    expected = sy.Matrix([[cosAngle, -1*sinAngle, 0], [sinAngle, cosAngle, 0], [0, 0, 1]])
    assert expected == RotationMatrix.RotAboutZ(angle)            