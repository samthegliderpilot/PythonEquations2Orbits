import sympy as sy
from typing import Union
from numbers import Real
from pyeq2orb.Utilities.Typing import SymbolOrNumber
def RotAboutZValladoConvention(rotationAngle : SymbolOrNumber) ->sy.Matrix :
    """Creates a 3x3 sympy Matrix representing the rotation about Z of the provided value.

    Args:
        rotationAngle : The rotation angle, often a number or a sympy Symbol

    Returns:
        sy.Matrix: The rotation matrix about Z of the provided angle
    """
    cv = sy.cos(rotationAngle)
    sv = sy.sin(rotationAngle)
    return sy.Matrix([[cv, sv, 0], [-1*sv, cv, 0], [0,0,1]])

def RotAboutYValladoConvention(rotationAngle : SymbolOrNumber) ->sy.Matrix :
    """Creates a 3x3 sympy Matrix representing the rotation about Y of the provided value.

    Args:
        rotationAngle : The rotation angle, often a number or a sympy Symbol

    Returns:
        sy.Matrix: The rotation matrix about Y of the provided angle
    """
    cv = sy.cos(rotationAngle)
    sv = sy.sin(rotationAngle)
    return sy.Matrix([[cv, 0, -1*sv], [0, 1, 0], [sv, 0, cv]])

def RotAboutXValladoConvention(rotationAngle : SymbolOrNumber) ->sy.Matrix :
    """Creates a 3x3 sympy Matrix representing the rotation about X of the provided value.

    Args:
        rotationAngle : The rotation angle, often a number or a sympy Symbol

    Returns:
        sy.Matrix: The rotation matrix about X of the provided angle
    """
    cv = sy.cos(rotationAngle)
    sv = sy.sin(rotationAngle)
    return sy.Matrix([[1, 0, 0],[0, cv, sv], [0, -1*sv, cv]])


def RotAboutZ(rotationAngle : SymbolOrNumber) ->sy.Matrix :
    """Creates a 3x3 sympy Matrix representing the rotation about Z of the provided value.

    Args:
        rotationAngle : The rotation angle, often a number or a sympy Symbol

    Returns:
        sy.Matrix: The rotation matrix about Z of the provided angle
    """
    cv = sy.cos(rotationAngle)
    sv = sy.sin(rotationAngle)
    return sy.Matrix([[cv, -1*sv, 0], [sv, cv, 0], [0,0,1]])

def RotAboutY(rotationAngle : SymbolOrNumber) ->sy.Matrix :
    """Creates a 3x3 sympy Matrix representing the rotation about Y of the provided value.

    Args:
        rotationAngle : The rotation angle, often a number or a sympy Symbol

    Returns:
        sy.Matrix: The rotation matrix about Y of the provided angle
    """
    cv = sy.cos(rotationAngle)
    sv = sy.sin(rotationAngle)
    return sy.Matrix([[cv, 0, sv], [0, 1, 0], [-1*sv, 0, cv]])

def RotAboutX(rotationAngle : SymbolOrNumber) ->sy.Matrix :
    """Creates a 3x3 sympy Matrix representing the rotation about X of the provided value.

    Args:
        rotationAngle : The rotation angle, often a number or a sympy Symbol

    Returns:
        sy.Matrix: The rotation matrix about X of the provided angle
    """
    cv = sy.cos(rotationAngle)
    sv = sy.sin(rotationAngle)
    return sy.Matrix([[1, 0, 0],[0, cv, -1*sv], [0, sv, cv]])    