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
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelper
from IPython.display import display
from scipy.integrate import solve_ivp #type: ignore
from pyeq2orb.Numerical import ScipyCallbackCreators #type: ignore
from pyeq2orb.Graphics.Primitives import EphemerisArrays #type: ignore
import pyeq2orb.Graphics.Primitives as prim #type: ignore
from pyeq2orb.Graphics.PlotlyUtilities import PlotAndAnimatePlanetsWithPlotly
from pyeq2orb.Coordinates.RotationMatrix import RotAboutZ #type: ignore
import math

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
    def simpleThreeBodyAcceleration(muPrime : sy.Symbol, muThirdBody : sy.Symbol, xSat : sy.Symbol, ySat : sy.Symbol, zSat : sy.Symbol, xThirdBody : sy.Symbol, yThirdBody : sy.Symbol, zThirdBody : sy.Symbol, subsDict : Optional[Dict[sy.Expr, SymbolOrNumber]] = None) -> sy.Matrix:
        rSatMag = sy.sqrt(xSat**2 + ySat**2+zSat**2)
        rSatVec = sy.Matrix([[xSat], [ySat], [zSat]])
        
        rThirdBodyMag = sy.sqrt(xThirdBody**2 + yThirdBody**2+zThirdBody**2)
        rThirdBodyVec = sy.Matrix([[xThirdBody], [yThirdBody], [zThirdBody]])
        
        rSatVecFromThirdBody = rThirdBodyVec- rSatVec
        rSatFromThirdBodyMag = sy.sqrt(rSatVecFromThirdBody[0]**2 + rSatVecFromThirdBody[1]**2+rSatVecFromThirdBody[2]**2)

        if subsDict is not None:
            deepRSatMag = rSatMag
            rSatMag = sy.Symbol(r'r_{sat}', real=True, positive=True)
            
            deepRSat = rSatVec
            rSatVec = sy.MatrixSymbol("\hat{r}_{sat}", 3, 1)
            
            deepRThirdBodyMag = rThirdBodyMag
            rThirdBodyMag = sy.Symbol(r'r_{3}', real=True, positive=True)

            deepRThirdBody = rThirdBodyVec
            rThirdBodyVec = sy.MatrixSymbol("\hat{r}_{3}", 3, 1)

            deepRSatVecFromThirdBody = rSatVecFromThirdBody
            rSatVecFromThirdBody = sy.MatrixSymbol("\hat{r}_{s-3}", 3, 1)

            deepRSatMagFromThirdBody = rSatFromThirdBodyMag
            rSatFromThirdBodyMag= sy.Symbol(r'r_{s-3}', real=True, positive=True)

            subsDict[rSatMag] = deepRSatMag
            subsDict[rSatVec] = deepRSat
            subsDict[rThirdBodyMag] = deepRThirdBodyMag
            subsDict[rThirdBodyVec] = deepRThirdBody
            subsDict[rSatVecFromThirdBody] = deepRSatVecFromThirdBody
            subsDict[rSatFromThirdBodyMag] = deepRSatMagFromThirdBody

        term1 = -1*muPrime*rSatVec/(rSatMag**3)
        term2 = muThirdBody*(rSatVecFromThirdBody/(rSatFromThirdBodyMag**3)) # Vallado's equation is confusing here...
        return term1+term2


class gravationalBody:
    def __init__(self, x, y, z, mu):
        self.x = x
        self.y = y
        self.z = z
        self.mu = mu

def nBodyDifferentialEquation(x, y, z, listOfNBodies : List[gravationalBody]) -> sy.Matrix:
    sum = sy.Matrix([[0.0], [0.0], [0.0]])
    for i in range(0, len(listOfNBodies)):
        cb = listOfNBodies[i]
        mu = cb.mu
        xcb = cb.x
        ycb = cb.y
        zcb = cb.z
        rVector = sy.Matrix([[x-xcb], [y-ycb], [z-zcb]])
        rMag = sy.sqrt(rVector[0]**2 + rVector[1]**2+rVector[2]**2)
        term = -1*mu*rVector/(rMag**3)
        sum = sum + term
    return sum

t = sy.Symbol('t', real=True)
x = sy.Function('x', real=True)(t)
y = sy.Function('y', real=True)(t)
z = sy.Function('z', real=True)(t)

vx = x.diff(t)
vy = y.diff(t)
vz = z.diff(t)
satVec = sy.Matrix([x,y,z])

muEarth = sy.Symbol(r'\mu_e', real=True, positive=True)
muMoon = sy.Symbol(r'\mu_l', real=True, positive=True)

rEarth2 = x*x+y*y+z*z

moonX = sy.Function('x_l', real=True)(t)#moonLoc[0]
moonY = sy.Function('y_l', real=True)(t)#moonLoc[1]
moonZ = sy.Function('z_l', real=True)(t)#moonLoc[2]

earthX = sy.Function('x_e', real=True)(t)#moonLoc[0]
earthY = sy.Function('y_e', real=True)(t)#moonLoc[1]
earthZ = sy.Function('z_e', real=True)(t)#moonLoc[2]

moonVec = sy.Matrix([moonX, moonY, moonZ]) # will be column
relToMoon = moonVec - satVec

muVal = 0.01215
subsDict = {muMoon: muVal, muEarth:1.0-muVal}
def moonXVal(t):
    return math.cos(t)*(1-muVal)

def moonYVal(t):
    return math.sin(t)*(1-muVal)

def moonZVal(t):
    return 0.0

def earthXVal(t):
    return -1*math.cos(t)*(muVal)

def earthYVal(t):
    return -1*math.sin(t)*(muVal)

def earthZVal(t):
    return 0.0
    

#matrixThirdBodyAcceleration = lunarPositionHelper.simpleThreeBodyAcceleration(muEarth, muMoon, x, y, z, moonX, moonY, moonZ, subsDict)
matrixThirdBodyAcceleration = nBodyDifferentialEquation(x, y, z, [gravationalBody(moonX, moonY, moonZ, muVal), gravationalBody(earthX, earthY, earthZ, (1-muVal)) ])
eom = [vx, vy, vz, matrixThirdBodyAcceleration[0], matrixThirdBodyAcceleration[1], matrixThirdBodyAcceleration[2]]

helper = OdeLambdifyHelper(t, [x,y,z,vx,vy,vz], eom, [], subsDict)
helper.FunctionRedirectionDictionary["x_l"] = moonXVal
helper.FunctionRedirectionDictionary["y_l"] = moonYVal
helper.FunctionRedirectionDictionary["z_l"] = moonZVal

helper.FunctionRedirectionDictionary["x_e"] = earthXVal
helper.FunctionRedirectionDictionary["y_e"] = earthYVal
helper.FunctionRedirectionDictionary["z_e"] = earthZVal

integratorCallback = helper.CreateSimpleCallbackForSolveIvp()
tArray = np.linspace(0.0, 10.0, 1000)
# values were found on degenerate conic blog, but are originally from are from https://figshare.com/articles/thesis/Trajectory_Design_and_Targeting_For_Applications_to_the_Exploration_Program_in_Cislunar_Space/14445717/1
nhrlState = [1.02134, 0, -0.18162, 0, -0.10176+1.02134, 9.76561e-07] #[ 	1.0277926091, 0.0, -0.1858044184, 0.0, -0.1154896637+1.0277926091, 0.0]
ipvResults = solve_ivp(integratorCallback, [tArray[0], tArray[-1]], nhrlState, t_eval=tArray, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, ipvResults)
satEphemeris = EphemerisArrays()
satEphemeris.ExtendValues(ipvResults.t, solutionDictionary[x], solutionDictionary[y], solutionDictionary[z]) #type: ignore
satPath = prim.PathPrimitive(satEphemeris, "#ff00ff")


inertialEphemeris = EphemerisArrays()
moonInertialEphemeris = EphemerisArrays()
moonPos = sy.Matrix([[1.0],[0.0],[0.0]])
satEphemerisRotatingFrame = EphemerisArrays()
for i in range(0, len(tArray)):
    tNow = tArray[i]
    rotMat = RotAboutZ(tNow).evalf()
    inverseRotMat = RotAboutZ(tNow).inv().evalf()
    newXyz = inverseRotMat*sy.Matrix([[satEphemeris.X[i]],[satEphemeris.Y[i]],[satEphemeris.Z[i]]])
    satEphemerisRotatingFrame.AppendValues(tNow, float(newXyz[0]), float(newXyz[1]), float(newXyz[2]))
    newMoonXyz = rotMat*moonPos
    moonInertialEphemeris.AppendValues(tNow, float(newMoonXyz[0]), float(newMoonXyz[1]), float(newMoonXyz[2]))
fig = PlotAndAnimatePlanetsWithPlotly("NHRL In Inertial Frame", [prim.PathPrimitive(satEphemeris, "#ff00ff", 3), prim.PathPrimitive(moonInertialEphemeris, "#000000", 3)], tArray, None)
fig.update_layout(
     margin=dict(l=20, r=20, t=20, b=20))
fig.show()  

x_2 = np.linspace(1.0, 1.0, 1000)
y_2 = np.linspace(0.0, 0.0, 1000)
moonRotatingEphemeris = EphemerisArrays()
moonRotatingEphemeris.ExtendValues(tArray, x_2, y_2, y_2) #type: ignore

fig = PlotAndAnimatePlanetsWithPlotly("NHRL In Rotating Frame", [prim.PathPrimitive(satEphemerisRotatingFrame, "#ff00ff", 3), prim.PathPrimitive(moonRotatingEphemeris, "#000000", 3)], tArray, None)
fig.update_layout(
     margin=dict(l=20, r=20, t=20, b=20))
fig.show()  

jh.showEquation(sy.MatrixSymbol(r'\ddot{r_{3}}', 3, 1), matrixThirdBodyAcceleration)
display(matrixThirdBodyAcceleration)
display(matrixThirdBodyAcceleration[0])
#%%
#display(matrixThirdBodyAcceleration)
#jh.showEquation("Z", eom)
initialState = [7000.0, 6000.0, 0.0, 0.0, 7.0, 5.0]
muEarthValue = 3.344*10**5 #TOOD: Look up
muMoonValue = 1.123*10**5 #TODO: Look up
lambdifyState = [t, *[x, y, z, vx, vy, vz],muEarth, muMoon]

sy.lambdify()

#lunarWrapper = lunarPositionHelper(20)

#cb = sy.lambdify(lambdifyState, eom, modules={"x_l":lambda tf : lunarWrapper.cachingAndIndexing(tf, 0), "y_l":lambda tf:lunarWrapper.cachingAndIndexing(tf, 0), "z_l":lambda tf: lunarWrapper.cachingAndIndexing(tf, 0)})

print(cb(0.0, *initialState, muEarthValue, muMoonValue))
#%%


# %%
import spiceypy as spice
import json
from typing import List
settings = json.loads(open("demoSettings.json", "r").read())


kernelPath = settings["kernelsDirectory"]

def getCriticalKernelsRelative()-> List[str]:
    criticalKernels = []
    criticalKernels.append("lsk/naif0012.tls")
    criticalKernels.append("pck/gm_de440.tpc")
    criticalKernels.append("pck/earth_latest_high_prec.cmt")
    criticalKernels.append("pck/earth_latest_high_prec.bpc")
    criticalKernels.append("pck/pck00010.tpc")

    return criticalKernels


def currentLunarFixedFrameKernelsRelative() ->List[str]:
    return ['pck/moon_pa_de440_200625.cmt', 'pck/moon_pa_de440_200625.bpc']

class spiceScope:
    #_standardGregorianFormat = "DD Mon YYYY-HH:MM:SC.######"

    def __init__(self, kernelPaths : List[str], baseDirectory : Optional[str]):
        self._kernelPaths :List[str] = []
        self._baseDirectory = baseDirectory
        if self._baseDirectory is not None:
            for partialPath in self._kernelPaths:
                self._kernelPaths.append(os.path.join(self._baseDirectory, partialPath))
        else:
            self._kernelPaths = kernelPaths

    def __enter__(self):
        for kernel in self._kernelPaths:
            spice.furnsh(kernel)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for kernel in self._kernelPaths:
            spice.unload(kernel)

    @staticmethod
    def etToUtc(et: float)->str:
        return spice.et2utc(et, 'C', 6)

    @staticmethod
    def utcToEt(utc : str)->float:
        return spice.utc2et(utc)

allMyKernels = getCriticalKernels(kernelPath)
allMyKernels.append("spk/planets/de440s.bsp") # big one here...
with spiceScope(allMyKernels, kernelPath) as scope:
    moonPos = spice.spkpos("Moon", 0.0, "J2000", 'NONE', 'EARTH BARYCENTER')
    print(moonPos)
    print(spiceScope.etToUtc(5.0))
    print(str(spiceScope.utcToEt("1 Jul 2025 12:14:16.123456")))

