#%%
import sympy as sy
from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import ModifiedEquinoctialElements, CreateSymbolicElements
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb import SafeSubs
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper, OdeLambdifyHelper, OdeLambdifyHelperWithBoundaryConditions
import scipyPaperPrinter as jh#type: ignore
import numpy as np
import math as math
from scipy.integrate import solve_ivp
from typing import Union, Dict, List, Callable
import pyeq2orb.Graphics.Primitives as prim
from pyeq2orb.Graphics.Plotly2DModule import plot2DLines
from pyeq2orb.Graphics.PlotlyUtilities import PlotAndAnimatePlanetsWithPlotly
from pyeq2orb.Numerical.ScalingHelpers import scaledEquationOfMotionHolder
from IPython.display import display

subsDict : Dict[Union[sy.Symbol, sy.Expr], SymbolOrNumber]= {}

t = sy.Symbol('t', real=True)
t0 = sy.Symbol('t_0', real=True)
tf = sy.Symbol('t_f', real=True)
mu = sy.Symbol(r'\mu', real=True, positive=True)

symbolicElements = CreateSymbolicElements(t, mu)
twoBodyOdeMatrix = CreateTwoBodyMotionMatrix(symbolicElements, subsDict)
twoBodyEvaluationHelper = OdeLambdifyHelper(t, symbolicElements.ToArray(), twoBodyOdeMatrix, [mu], subsDict)
twoBodyOdeCallback = twoBodyEvaluationHelper.CreateSimpleCallbackForSolveIvp()


#%%
tfVal = 793*86400.0
n = 303
tSpace = np.linspace(0.0, tfVal, n)

muVal = 1.32712440042e20
r0 = Cartesian(58252488010.7, 135673782531.3, 2845058.1)
v0 = Cartesian(-27844.5, 11659.9, 0000.3)
initialElements = ModifiedEquinoctialElements.FromMotionCartesian(MotionCartesian(r0, v0), muVal)

rf = Cartesian(36216277800.4, -211692395522.5, -5325189049.9)
vf = Cartesian(24798.8, 6168.2, -480.0)
finalElements = ModifiedEquinoctialElements.FromMotionCartesian(MotionCartesian(rf, vf), muVal)

earthSolution = solve_ivp(twoBodyOdeCallback, [0.0, tfVal], initialElements.ToArray(), args=[muVal], t_eval=np.linspace(0.0, tfVal,n), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
(tArray, earthArrays) = twoBodyEvaluationHelper.SolveIvpResultsReshaped(earthSolution)
earthMmes = [ModifiedEquinoctialElements(*x, muVal) for x in earthArrays]
motions = ModifiedEquinoctialElements.CreateEphemeris(earthMmes)
earthPath = prim.PlanetPrimitive.fromMotionEphemeris(tArray, motions, "#00ff00")


marsSolution = solve_ivp(twoBodyOdeCallback, [tfVal, 0.0], finalElements.ToArray(), args=[muVal], t_eval=np.linspace(tfVal, 0.0, n*2), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
(tArray, marsStateArrays) = twoBodyEvaluationHelper.SolveIvpResultsReshaped(marsSolution)
marsMees = [ModifiedEquinoctialElements(*x, muVal) for x in marsStateArrays]
marsMotion = ModifiedEquinoctialElements.CreateEphemeris(marsMees)
marsPath = prim.PlanetPrimitive.fromMotionEphemeris(tArray, marsMotion, "#ff0000")


#%%
# build up a perturbation matrix
Au = 149597870700.0
AuSy = sy.Symbol('A_u')
gVal = 9.8065 
gSy = sy.Symbol('g', real=True, positive=True) #9.8065
m0Val = 2000.0
ispVal = 3000.0
nRev = 2.0
thrustVal = 0.203 # odd number pulled from just under Fig14

azi = sy.Function(r'\theta', real=True)(t)
elv = sy.Function(r'\phi', real=True)(t)
thrust = sy.Symbol('T', real=True, positive=True)
throttle = sy.Function('\delta', real=True, positive=True)(t)
m = sy.Function('m', real=True, positive=True)(t)
isp = sy.Symbol("I_{sp}", real=True, positive=True)

alp = sy.Matrix([[sy.cos(azi)*sy.cos(elv)], [sy.sin(azi)*sy.cos(elv)], [sy.sin(elv)]])
B = symbolicElements.CreatePerturbationMatrix(subsDict)
overallThrust = thrust*B*alp*(throttle)/(m) 
stateDynamics = twoBodyOdeMatrix + overallThrust

c = isp * gSy
pathCost = throttle* thrust/c#TODO: VERY VERY IMPORTANT but currently unused!!!
mDot = -1*thrust*throttle/(isp*gSy)

stateDynamics=stateDynamics.row_insert(6, sy.Matrix([mDot]))

stateVariables = [*symbolicElements.ToArray(), m]
#for i in range(0, 7):
#    jh.showEquation(stateVariables[i].diff(t), stateDynamics[i])
newSvs = scaledEquationOfMotionHolder.CreateVariablesWithBar(stateVariables, t)
tau = sy.Symbol(r'\tau', positive=True, real=True)
scalingFactors =  [Au, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
scaledEoms = scaledEquationOfMotionHolder.ScaleStateVariablesAndTimeInFirstOrderOdes(stateVariables, stateDynamics, newSvs,scalingFactors, tau, tf, [azi, elv, throttle])

for seom in scaledEoms.scaledFirstOrderDynamics:
    display("a", seom)
subsDict[gSy] = gVal
simpleThrustCallbackHelper = OdeLambdifyHelper(t, stateVariables, stateDynamics, [mu, azi, elv, thrust, throttle, isp], subsDict)
simpleThrustCallback = simpleThrustCallbackHelper.CreateSimpleCallbackForSolveIvp()

satSolution = solve_ivp(simpleThrustCallback, [0.0, tfVal], [*initialElements.ToArray(), m0Val], args=[muVal, 1.5, 0.0, thrustVal, 1.0, ispVal], t_eval=np.linspace(0.0, tfVal,n), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
(tArray, satArrays) = simpleThrustCallbackHelper.SolveIvpResultsReshaped(satSolution)
(satMees) = [ModifiedEquinoctialElements(*x[:6], muVal) for x in satArrays]
motions = ModifiedEquinoctialElements.CreateEphemeris(satMees)
satPath = prim.PlanetPrimitive.fromMotionEphemeris(tArray, motions, "#00ffff")
#%%
simpleThrustCallbackHelperScaled = OdeLambdifyHelper(tau, scaledEoms.newStateVariables, scaledEoms.scaledFirstOrderDynamics, [mu, *scaledEoms.otherSymbols, isp], subsDict)
simpleThrustCallbackScaledScaled = simpleThrustCallbackHelperScaled.CreateSimpleCallbackForSolveIvp()

initialElementsScaled = [*initialElements]
initialElementsScaled[0] = initialElementsScaled[0]/Au
satSolutionScaled = solve_ivp(simpleThrustCallbackScaledScaled, [0.0, 1.0], [*initialElementsScaled, m0Val], args=[muVal, 1.5, 0.0, thrustVal, 1.0, ispVal], t_eval=np.linspace(0.0, tfVal,n), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
(tArrayScaled, satArraysScaled) = simpleThrustCallbackHelper.SolveIvpResultsReshaped(initialElementsScaled)

satArrays2 = scaledEoms.descaleStates(tArrayScaled, satArraysScaled, scalingFactors, tfVal)
(satMees2) = [ModifiedEquinoctialElements(*x[:6], muVal) for x in satArrays2]
motions2 = ModifiedEquinoctialElements.CreateEphemeris(satMees2)
satPath2 = prim.PlanetPrimitive.fromMotionEphemeris(tArray, motions2, "#ff00ff")


fig = PlotAndAnimatePlanetsWithPlotly("Earth and Mars", [earthPath, marsPath, satPath, satPath2], tArray, None)
fig.update_layout()
fig.show()  


diff = np.array(satArrays) - np.array(satArrays2)
import matplotlib.pyplot as plt
import numpy as np


plt.plot(diff)
plt.show()