import __init__ #type: ignore
from pyeq2orb.Graphics import CzmlUtilities
from datetime import datetime
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import EquinoctialElementsHalfI, ModifiedEquinoctialElements
import numpy as np
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper, OdeLambdifyHelper
from scipy.integrate import solve_ivp #type: ignore
import pyeq2orb.Graphics.Primitives as prim
import sympy as sy
from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import ModifiedEquinoctialElements, CreateSymbolicElements
from typing import cast,List

tfVal = 3*86400.0
m0Val = 2000.0
isp = 3000.0
nRev = 2.0
thrustVal =  0.1997*1.2
g = 9.8065 
n = 201
tSpace = np.linspace(0.0, tfVal, n)
t = sy.Symbol('t')
Au = 149597870700.0
AuSy = sy.Symbol('A_u')
muVal = 3.986004418e14  
r0 = Cartesian(8000.0e3, 8000.0e3, 0.0)
v0 = Cartesian(0, 5.000e3, 4.500e3)
initialElements = ModifiedEquinoctialElements.FromMotionCartesian(MotionCartesian(r0, v0), muVal)

gSy = sy.Symbol('g', real=True, positive=True)
symbolicElements = CreateSymbolicElements(t)
def GetEquiElementsOutOfIvpResults(ivpResults) :
    t = []
    equi = []
    yFromIntegrator = ivpResults.y 
    for i in range(0, len(yFromIntegrator[0])):
        temp = ModifiedEquinoctialElements(yFromIntegrator[0][i], yFromIntegrator[1][i], yFromIntegrator[2][i], yFromIntegrator[3][i], yFromIntegrator[4][i], yFromIntegrator[5][i], muVal)
        equi.append(temp)
        t.append(ivpResults.t[i])

    if t[0] > t[1] :
        t.reverse()
        equi.reverse()
    return (t, equi)

twoBodyMatrix = CreateTwoBodyListForModifiedEquinoctialElements(symbolicElements)
simpleTwoBodyLambdifyCreator = OdeLambdifyHelper(t, twoBodyMatrix, [], {symbolicElements.GravitationalParameter: muVal})
odeCallback =simpleTwoBodyLambdifyCreator.CreateSimpleCallbackForSolveIvp()
earthSolution = solve_ivp(odeCallback, [0.0, tfVal], initialElements.ToArray(), args=tuple(), t_eval=np.linspace(0.0, tfVal,n), dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)

(tArray, equiElements) = GetEquiElementsOutOfIvpResults(earthSolution)
motions = ModifiedEquinoctialElements.CreateEphemeris(equiElements)
earthEphemeris = prim.EphemerisArrays()
earthEphemeris.InitFromMotions(tArray, motions)
earthPath = prim.PathPrimitive(earthEphemeris)
earthPath.color = "#0000ff"



czmlDoc = CzmlUtilities.createCzmlFromPoints(datetime(year=2020, month=1, day=1), "Planets", [earthPath])
f = open(r'C:\src\CesiumElectronStarter\someCzml.czml', "w")
f.write(str(czmlDoc))
f.close()