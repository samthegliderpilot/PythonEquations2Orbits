import pytest
import math
import sympy as sy
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper, OdeLambdifyHelper #type: ignore
from pyeq2orb.Spice.rotationMatrixWrapper import rotationMatrixFunction
import spiceypy as spice

def test_something():
        initialPosVel = [7100000.0, 0.0, 1300000.0,   0.0, 7350.0, 1000.0] # m and m/sec, LEO
        etEpoch = 0.0 # TAI at 0.0

        rotHelper = rotationMatrixFunction("J2000", "ITRF93")

        helperDict = {}
        rotHelper.populateRedirectionDictWithCallbacks(helperDict)
        i2fSymbol = sy.Matrix([[sy.Function('Rxx', real=True)(t), sy.Function('Rxy', real=True)(t), sy.Function('Rxz', real=True)(t)],
                               [sy.Function('Ryx', real=True)(t), sy.Function('Ryy', real=True)(t), sy.Function('Ryz', real=True)(t)],
                               [sy.Function('Rzx', real=True)(t), sy.Function('Rzy', real=True)(t), sy.Function('Rzz', real=True)(t)]])
