#%%
import sympy as sy
import math
import os
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

muVal = 0.01215
subsDict = {}

t = sy.Symbol('t', real=True)
x = sy.Function('x', real=True)(t)
y = sy.Function('y', real=True)(t)
z = sy.Function('z', real=True)(t)
vx = x.diff(t)
vy = y.diff(t)
vz = z.diff(t)
satVec = sy.Matrix([x,y,z])

mu = sy.Symbol(r'\mu', real=True, positive=True)
subsDict[mu]= muVal

earthX = sy.Function('x_e', real=True)(t)
earthY = sy.Function('y_e', real=True)(t)
earthZ = sy.Function('z_e', real=True)(t)

moonX = sy.Function('x_l', real=True)(t)
moonY = sy.Function('y_l', real=True)(t)
moonZ = sy.Function('z_l', real=True)(t)
matrixThirdBodyAcceleration = nBodyDifferentialEquation(x, y, z, [gravationalBody(moonX, moonY, moonZ, mu), gravationalBody(earthX, earthY, earthZ, 1-mu) ])


#low thrust engine
gVal = 9.8065 
gSy = sy.Symbol('g', real=True, positive=True) #9.8065
m0Val = 2000.0
ispVal = 3000.0
thrustVal = 0.1997*1.2#0.25 # odd number pulled from just under Fig14

subsDict[gSy] = gVal

azi = sy.Function(r'\theta', real=True)(t)
elv = sy.Function(r'\phi', real=True)(t)
thrust = sy.Symbol('T', real=True, positive=True)
throttle = sy.Function('\delta', real=True, positive=True)(t)
m = sy.Function('m', real=True, positive=True)(t)
isp = sy.Symbol("I_{sp}", real=True, positive=True)

alp = sy.Matrix([[sy.cos(azi)*sy.cos(elv)], [sy.sin(azi)*sy.cos(elv)], [sy.sin(elv)]])
B = symbolicElements.CreatePerturbationMatrix(subsDict)
overallThrust = thrust*B*alp*(throttle)/(m) 
c = isp * gSy
mDot = -1*thrust*throttle/(isp*gSy)




eom = [vx, vy, vz, matrixThirdBodyAcceleration[0], matrixThirdBodyAcceleration[1], matrixThirdBodyAcceleration[2]]



helper = OdeLambdifyHelper(t, [x,y,z,vx,vy,vz], eom, [], subsDict)
helper.FunctionRedirectionDictionary["x_l"] = cr3bpPlanetCallbacks.secondaryXCallbackOfTime(t, muVal)
helper.FunctionRedirectionDictionary["y_l"] = cr3bpPlanetCallbacks.secondaryYCallbackOfTime(t, muVal)
helper.FunctionRedirectionDictionary["z_l"] = cr3bpPlanetCallbacks.secondaryZCallbackOfTime(t, muVal)

helper.FunctionRedirectionDictionary["x_e"] = cr3bpPlanetCallbacks.primaryXCallbackOfTime(t, muVal)
helper.FunctionRedirectionDictionary["y_e"] = cr3bpPlanetCallbacks.primaryYCallbackOfTime(t, muVal)
helper.FunctionRedirectionDictionary["z_e"] = cr3bpPlanetCallbacks.primaryZCallbackOfTime(t, muVal)

integratorCallback = helper.CreateSimpleCallbackForSolveIvp()
tArray = np.linspace(0.0, 10.0, 1000)