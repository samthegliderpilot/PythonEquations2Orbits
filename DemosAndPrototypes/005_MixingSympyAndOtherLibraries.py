#%%
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
from typing import Optional, List, Dict
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from IPython.display import display
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

    def cartesianTwoBodyAcceleration(mu : sy.Symbol, x : sy.Symbol, y : sy.Symbol, z : sy.Symbol, subsDict= Optional[Dict[sy.Expr, SymbolOrNumber]]):
        rSatMag = sy.sqrt(x**2 + y**2+z**2)
        rSatVec = sy.Matrix([[x], [y], [z]])
        
        if subsDict is not None:
            deepRSatMag = rSatMag
            rSatMag = sy.Symbol(r'r_{sat}', real=True, positive=True)
            
            deepRSat = rSatVec
            rSatVec = sy.MatrixSymbol("\hat{r}_{sat}", 3, 1)
            
            subsDict[rSatMag] = deepRSatMag
            subsDict[rSatVec] = deepRSat
        
        accel = mu*rSatVec/(rSatMag**3)
        return -1*accel

    @staticmethod    
    def simpleThreeBodyAcceleration(muPrime : sy.Symbol, muThirdBody : sy.Symbol, xSat : sy.Symbol, ySat : sy.Symbol, zSat : sy.Symbol, xThirdBody : sy.Symbol, yThirdBody : sy.Symbol, zThirdBody : sy.Symbol, subsDict = Optional[Dict[sy.Expr, SymbolOrNumber]]) -> sy.Matrix:
        rSatMag = sy.sqrt(xSat**2 + ySat**2+zSat**2)
        rSatVec = sy.Matrix([[xSat], [ySat], [zSat]])
        
        rThirdBodyMag = sy.sqrt(xThirdBody**2 + yThirdBody**2+zThirdBody**2)
        rThirdBodyVec = sy.Matrix([[xThirdBody], [yThirdBody], [zThirdBody]])
        if subsDict is not None:
            deepRSatMag = rSatMag
            rSatMag = sy.Symbol(r'r_{sat}', real=True, positive=True)
            
            deepRSat = rSatVec
            rSatVec = sy.MatrixSymbol("\hat{r}_{sat}", 3, 1)
            
            deepRThirdBodyMag = rThirdBodyMag
            rThirdBodyMag = sy.Symbol(r'r_{3}', real=True, positive=True)

            deepRThirdBody = rThirdBodyVec
            rThirdBodyVec = sy.MatrixSymbol("\hat{r}_{3}", 3, 1)
            
            subsDict[rSatMag] = deepRSatMag
            subsDict[rSatVec] = deepRSat
            subsDict[rThirdBodyMag] = deepRThirdBodyMag
            subsDict[rThirdBodyVec] = deepRThirdBody
            

        term1 = -1*muPrime*rSatVec/(rSatMag**3)
        term2 = muThirdBody*(rSatVec/(rSatMag**3)- (rThirdBodyVec/rThirdBodyMag**3)) # Vallado's equation is confusing here...
        return term1+term2

t = sy.Symbol('t', real=True)
x = sy.Function('x', real=True)(t)
y = sy.Function('y', real=True)(t)
z = sy.Function('z', real=True)(t)

xDot = x.diff(t)
yDot = y.diff(t)
zDot = z.diff(t)
satVec = sy.Matrix([x,y,z])

muEarth = sy.Symbol(r'\mu_e', real=True, positive=True)
muMoon = sy.Symbol(r'\mu_l', real=True, positive=True)

rEarth2 = x*x+y*y+z*z

moonX = sy.Function('x_l', real=True)(t)#moonLoc[0]
moonY = sy.Function('y_l', real=True)(t)#moonLoc[1]
moonZ = sy.Function('z_l', real=True)(t)#moonLoc[2]
moonVec = sy.Matrix([moonX, moonY, moonZ]) # will be column
relToMoon = moonVec - satVec


subsDict = {}
matrixThirdBodyAcceleration = lunarPositionHelper.simpleThreeBodyAcceleration(muEarth, muMoon, x, y, z, moonX, moonY, moonZ, subsDict)
matrixTwoBodyAcceleration = lunarPositionHelper.cartesianTwoBodyAcceleration(muEarth, x, y, z, subsDict)

eom = [xDot, yDot, zDot, matrixTwoBodyAcceleration[0]+matrixThirdBodyAcceleration[0], matrixTwoBodyAcceleration[1]+matrixThirdBodyAcceleration[1], matrixTwoBodyAcceleration[2]+matrixThirdBodyAcceleration[2]]
jh.showEquation(sy.MatrixSymbol(r'\ddot{r_{E}}', 3, 1), matrixTwoBodyAcceleration)
jh.showEquation(sy.MatrixSymbol(r'\ddot{r_{3}}', 3, 1), matrixThirdBodyAcceleration)
display(matrixTwoBodyAcceleration)
display(matrixTwoBodyAcceleration[0])
#display(matrixThirdBodyAcceleration)
#jh.showEquation("Z", eom)
initialState = [7000.0, 6000.0, 0.0, 0.0, 7.0, 5.0]
muEarthValue = 3.344*10**5 #TOOD: Look up
muMoonValue = 1.123*10**5 #TODO: Look up
lambdifyState = [t, *[x, y, z, xDot, yDot, zDot],muEarth, muMoon]

lunarWrapper = lunarPositionHelper(20)

cb = sy.lambdify(lambdifyState, eom, modules={"x_l":lambda tf : lunarWrapper.cachingAndIndexing(tf, 0), "y_l":lambda tf:lunarWrapper.cachingAndIndexing(tf, 0), "z_l":lambda tf: lunarWrapper.cachingAndIndexing(tf, 0)})

print(cb(0.0, *initialState, muEarthValue, muMoonValue))


# %%
