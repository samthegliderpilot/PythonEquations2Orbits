#%%
import __init__ # type: ignore
import sympy as sy
import math
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)
sy.init_printing()
import scipyPaperPrinter as jh
import numpy as np
from sympy import ImmutableDenseMatrix
from collections import OrderedDict

def lunarPosition(t)-> np.array:
    return np.array([1, 2, 3])

def lunarPositionX(t)-> np.array:
    return 1.0
def lunarPositionY(t)-> np.array:
    return 2.0
def lunarPositionZ(t)-> np.array:
    return 3.0

def evaluateRotationMatrix(t)-> np.matrix:
    return np.matrix([[math.cos(t), math.sin(t), 0.0],[math.sin(t), -1*math.cos(t), 0.0], [0.0,0.0,1.0]])

class lunarPositionHelper:
    def __init__(self, cacheSize):
        self._lastSeveralResults = OrderedDict()
        self._cacheSize = cacheSize

    @staticmethod
    def simpleLunarPosition(t):
        return [1, 2, 3]

    def cachingAndIndexing(self, t, i):
        if t in self._lastSeveralResults.keys():
            return self._lastSeveralResults[t][i]
        if len(self._lastSeveralResults.keys()) >self._cacheSize:
            self._lastSeveralResults.popitem(False)
        newAnswer = lunarPositionHelper.simpleLunarPosition(t)
        self._lastSeveralResults[t] = newAnswer
        return newAnswer[i]

t = sy.Symbol('t')
x = sy.Function('x')(t)
y = sy.Function('y')(t)
z = sy.Function('z')(t)

xDot = x.diff(t)
yDot = y.diff(t)
zDot = z.diff(t)
satVec = sy.Matrix([x,y,z])

muEarth = sy.Symbol(r'\mu_e')
muMoon = sy.Symbol(r'\mu_l')

rEarth2 = x*x+y*y+z*z

moonX = sy.Function('x_l')(t)#moonLoc[0]
moonY = sy.Function('y_l')(t)#moonLoc[1]
moonZ = sy.Function('z_l')(t)#moonLoc[2]
moonVec = sy.Matrix([moonX, moonY, moonZ]) # will be column
relToMoon = moonVec - satVec

rSatToMoon2 = xRelativeToMoon**2+yRelativeToMoon**2+zRelativeToMoon**2



eom = [xDot, yDot, zDot, muEarth*x/rEarth2 + muMoon*moonX/rSatToMoon2, muEarth*y/rEarth2*moonY/rSatToMoon2, muEarth*z/rEarth2*moonZ/rSatToMoon2]
#jh.showEquation("Z", eom)
initialState = [7000.0, 6000.0, 0.0, 0.0, 7.0, 5.0]
muEarthValue = 3.344*10**5 #TOOD: Look up
muMoonValue = 1.123*10**5 #TODO: Look up
lambdifyState = [t, *[x, y, z, xDot, yDot, zDot],muEarth, muMoon]

lunarWrapper = lunarPositionHelper(20)

cb = sy.lambdify(lambdifyState, eom, modules={"x_l":lambda tf : lunarWrapper.cachingAndIndexing(tf, 0), "y_l":lambda tf:lunarWrapper.cachingAndIndexing(tf, 0), "z_l":lambda tf: lunarWrapper.cachingAndIndexing(tf, 0)})

print(cb(0.0, *initialState, muEarthValue, muMoonValue))


# %%
