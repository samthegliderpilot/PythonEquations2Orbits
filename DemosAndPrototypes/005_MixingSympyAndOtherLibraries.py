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
from typing import Optional, List, Dict, Tuple
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
from pyeq2orb.ForceModels.NBodyEquations import nBodyDifferentialEquation, gravationalBody, cr3bpPlanetCallbacks
from pyeq2orb.Spice.spiceScope import spiceScope

t = sy.Symbol('t', real=True)
x = sy.Function('x', real=True)(t)
y = sy.Function('y', real=True)(t)
z = sy.Function('z', real=True)(t)
vx = x.diff(t)
vy = y.diff(t)
vz = z.diff(t)
satVec = sy.Matrix([x,y,z])

mu = sy.Symbol(r'\mu', real=True, positive=True)
earthX = sy.Function('x_e', real=True)(t)
earthY = sy.Function('y_e', real=True)(t)
earthZ = sy.Function('z_e', real=True)(t)

moonX = sy.Function('x_l', real=True)(t)
moonY = sy.Function('y_l', real=True)(t)
moonZ = sy.Function('z_l', real=True)(t)
matrixThirdBodyAcceleration = nBodyDifferentialEquation(x, y, z, [gravationalBody(moonX, moonY, moonZ, mu), gravationalBody(earthX, earthY, earthZ, 1-mu) ])
eom = [vx, vy, vz, matrixThirdBodyAcceleration[0], matrixThirdBodyAcceleration[1], matrixThirdBodyAcceleration[2]]

muVal = 0.01215
subsDict = {mu: muVal}

helper = OdeLambdifyHelper(t, [x,y,z,vx,vy,vz], eom, [], subsDict)
helper.FunctionRedirectionDictionary["x_l"] = cr3bpPlanetCallbacks.secondaryXCallbackOfTime(t, muVal)
helper.FunctionRedirectionDictionary["y_l"] = cr3bpPlanetCallbacks.secondaryYCallbackOfTime(t, muVal)
helper.FunctionRedirectionDictionary["z_l"] = cr3bpPlanetCallbacks.secondaryZCallbackOfTime(t, muVal)

helper.FunctionRedirectionDictionary["x_e"] = cr3bpPlanetCallbacks.primaryXCallbackOfTime(t, muVal)
helper.FunctionRedirectionDictionary["y_e"] = cr3bpPlanetCallbacks.primaryYCallbackOfTime(t, muVal)
helper.FunctionRedirectionDictionary["z_e"] = cr3bpPlanetCallbacks.primaryZCallbackOfTime(t, muVal)

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
import functools

# convert that to spice-backed orbit
class spiceMoonCalculator:
    def __init__(self):
        pass

    @functools.lru_cache
    def lunarPosition(self, et:float)->Tuple[float, float, float]:
        moonPos = spice.spkpos("Moon", et, "J2000", 'NONE', 'EARTH BARYCENTER')
        return (moonPos[0][0], moonPos[0][1], moonPos[0][2])

    def lunarX(self, et)->float:
        return self.lunarPosition(et)[0]

    def lunarY(self, et)->float:
        return self.lunarPosition(et)[1]

    def lunarZ(self, et)->float:
        return self.lunarPosition(et)[2]

    def lunarOrbitalRadius(self, et)->float:
        pos = self.lunarPosition(et)
        return math.sqrt(pos[0]**2+pos[1]**2+pos[2]**2)

    @functools.lru_cache
    def earthPosition(self, et:float)->Tuple[float, float, float]:
        earthPos = spice.spkpos("Earth", et, "J2000", 'NONE', 'EARTH BARYCENTER')
        return (earthPos[0][0], earthPos[0][1], earthPos[0][2])

    def earthX(self, et)->float:
        return self.earthPosition(et)[0]

    def earthY(self, et)->float:
        return self.earthPosition(et)[1]

    def earthZ(self, et)->float:
        return self.earthPosition(et)[2]

    @functools.cached_property
    def earthMu(self) -> float:
        return spice.bodvrd("EARTH", "GM", 1)[1][0]

    @functools.cached_property
    def moonMu(self) -> float:
        return spice.bodvrd("MOON", "GM", 1)[1][0]

    def earthOrbitalRadius(self, et)->float:
        pos = self.earthPosition(et)
        return math.sqrt(pos[0]**2+pos[1]**2+pos[2]**2)

    @staticmethod
    def convertRotationalCr3bpStateToInertial(t, x, y, z, vx, vy, vz) ->Tuple[float, float, float, float, float, float]:
        ct = sy.cos(t)
        st = sy.sin(t)
        xi = ct*x+st*y
        yi = -1*st*x + ct*y
        zi = z

        vxi = (vx-y)*ct+st*(vy+x)
        vyi = -1*(vx-y)*st+ct*(vy+x)
        vzi = vz

        return (xi, yi, zi, vxi, vyi, vzi)

    @staticmethod
    def convertInertialToRotationalCr3bpState(t, x, y, z, vx, vy, vz) ->Tuple[float, float, float, float, float, float]:
        ct = sy.cos(-1*t)
        st = sy.sin(-1*t)
        xi = ct*x+st*y
        yi = -1*st*x + ct*y
        zi = z

        vxi = (vx+y)*ct+st*(vy-x)
        vyi = -1*(vx+y)*st+ct*(vy-x)
        vzi = vz

        return (xi, yi, zi, vxi, vyi, vzi)

import spiceypy as spice
import json
from typing import List
settings = json.loads(open("demoSettings.json", "r").read())
print(settings)

kernelPath = settings["kernelsDirectory"]

def currentLunarFixedFrameKernelsRelative() ->List[str]:
    return ['pck/moon_pa_de440_200625.cmt', 'pck/moon_pa_de440_200625.bpc']



def getCriticalKernelsRelativePaths()-> List[str]:
    criticalKernels = []
    criticalKernels.append("lsk/naif0012.tls")
    criticalKernels.append("pck/earth_latest_high_prec.cmt")
    criticalKernels.append("pck/earth_latest_high_prec.bpc")
    criticalKernels.append("pck/pck00010.tpc")
    criticalKernels.append("pck/gm_de440.tpc")

    return criticalKernels



allMyKernels = getCriticalKernelsRelativePaths()
allMyKernels.append("spk/planets/de440s.bsp") # big one here...
with spiceScope(allMyKernels, kernelPath) as scope:
    moonPos = spice.spkpos("Moon", 0.0, "J2000", 'NONE', 'EARTH BARYCENTER')
    print(moonPos)
    print(spiceScope.etToUtc(5.0))
    print(str(spiceScope.utcToEt("1 Jul 2025 12:34:36.123456")))

    calc = spiceMoonCalculator()
    print(str(calc.earthMu))

