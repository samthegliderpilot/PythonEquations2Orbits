#%%
import sympy as sy
import os
import sys
import math
from collections import OrderedDict
sy.init_printing()
from typing import Union, List, Optional, Sequence, cast, Dict
from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
import pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule as mee
from IPython.display import display
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions
import scipyPaperPrinter as jh #type: ignore
from scipy.integrate import solve_ivp #type: ignore
import numpy as np
from pyeq2orb.Graphics.Plotly2DModule import plot2DLines
import pyeq2orb.Graphics.Primitives as prim
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb.Graphics.PlotlyUtilities import PlotAndAnimatePlanetsWithPlotly
from pyeq2orb import SafeSubs, MakeMatrixOfSymbols
import pyeq2orb
#jh.printMarkdown("# SEPSPOT Recreation")
#jh.printMarkdown("In working my way up through low-thrust modeling for satellite maneuvers, it is inevitable to run into Dr. Edelbaum's work.  Newer work such as Jean Albert Kechichian's practically requires understanding SEPSPOT as a prerequesit.  This writeup will go through the basics of SEPSPOT's algorithsm as described in the references below.")



#jh.printMarkdown("In other work in this python library, I have already created many helper types such as Equinoctial elements, their equations of motion, rotation matrices, and more. To start, we will define out set of equinoctial elements.  Unlike the orignial paper, I will be using the modified elements.  This replaces the semi-major axis with the parameter and reorders/renames some of the other elements.")
t=sy.Symbol('t', real=True)
mu = sy.Symbol(r'\mu', real=True, positive=True)
muVal = 3.986004418e5  
#kepElements = KepModule.CreateSymbolicElements(t, mu)
fullSubsDictionary = OrderedDict() #type: dict[sy.Expr, SymbolOrNumber]

simpleBoringEquiElements = mee.EquinoctialElementsHalfI.CreateSymbolicElements(t, mu)
a = cast(sy.Expr, simpleBoringEquiElements.SemiMajorAxis)
h = cast(sy.Expr, simpleBoringEquiElements.EccentricitySinTermH)
k = cast(sy.Expr, simpleBoringEquiElements.EccentricityCosTermK)
p = cast(sy.Expr, simpleBoringEquiElements.InclinationSinTermP)
q = cast(sy.Expr, simpleBoringEquiElements.InclinationCosTermQ)
F = cast(sy.Expr, simpleBoringEquiElements.Longitude)
n = sy.sqrt(mu/(a**3))
x = sy.Matrix([[simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermK, simpleBoringEquiElements.InclinationSinTermP, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.Longitude]]).transpose()
z = [simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermK, simpleBoringEquiElements.InclinationSinTermP, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.Longitude]
beta = simpleBoringEquiElements.BetaSy
betaExp = simpleBoringEquiElements.Beta
eccentricLongitude = simpleBoringEquiElements.Longitude
f = eccentricLongitude

rotMatrix = mee.EquinoctialElementsHalfI.CreateFgwToInertialAxesStatic(p, q)
fHatSy = MakeMatrixOfSymbols(r'\hat{f}', 3, 1, [p, q])
gHatSy = MakeMatrixOfSymbols(r'\hat{g}', 3, 1, [p, q])
wHatSy = MakeMatrixOfSymbols(r'\hat{w}', 3, 1, [p, q])
display(fHatSy)
for i in range(0, 3):
    fullSubsDictionary[fHatSy[i]] = rotMatrix.col(0)[i]
    fullSubsDictionary[gHatSy[i]] = rotMatrix.col(1)[i]
    fullSubsDictionary[wHatSy[i]] = rotMatrix.col(2)[i]

jh.showEquation(fHatSy, rotMatrix.col(0))
jh.showEquation(gHatSy, rotMatrix.col(1))
jh.showEquation(wHatSy, rotMatrix.col(2))
M = simpleBoringEquiElements.CreatePerturbationMatrixWithMeanLongitude(f, fullSubsDictionary)
jh.showEquation("M", M)
display(M.shape)

#mFullSymbol = makeMatrixOfSymbols("M", 5, 3, [a,h,k,p,q,F])

rOverA = simpleBoringEquiElements.ROverA
taDifeq = n*sy.sqrt(1-h**2-k**2)/(rOverA**2)
aSy = sy.Function('A', commutative=True)(x, t)
u1 = sy.Symbol("u_1", real=True)
u2 = sy.Symbol("u_2", real=True)
u3 = sy.Symbol("u_3", real=True)
uSy = sy.Matrix([[u1, u2, u3]]).transpose()
accelSy = sy.Symbol('a', real=True, positive=True)
zDot = M*uSy*accelSy + sy.Matrix([[0,0,0,0,0,taDifeq]]).transpose()

#%%

def recurseArgs(someFunction, argsICareAbout, existingArgs) : 
    recursed = False
    if someFunction in argsICareAbout and not someFunction in existingArgs:
        existingArgs.append(someFunction)
        return existingArgs
    if( hasattr(someFunction, "args")) :
        for arg in someFunction.args:
            if hasattr(arg, "args"):
                recurseArgs(arg, argsICareAbout, existingArgs)
            elif not arg in existingArgs:
                existingArgs.append(arg)
    elif not someFunction in existingArgs:
        existingArgs.append(arg)
    return existingArgs

def createMatrixOfFunctionsFromDenseMatrix(someMatrix, argsICareAbout,stringName):
    mat = sy.Matrix.zeros(*someMatrix.shape)
    for r in range(0, someMatrix.rows) :
        for c in range(0, someMatrix.cols):
            thisEleemntName = stringName + "_{" + str(r) + "," + str(c)+"}"            
            mat[r,c] =sy.Function(thisEleemntName)(*recurseArgs(someMatrix[r,c], argsICareAbout, []))
    return mat

mFullSymbol = createMatrixOfFunctionsFromDenseMatrix(M, x, "M")

#%%
n = len(x)
jh.printMarkdown("Staring with our x:")
xSy = sy.MatrixSymbol('x', n, 1)
jh.showEquation(xSy, x)
jh.printMarkdown(r'We write our $\underline{\dot{x}}$ with the assumed optimal control vector $\underline{\hat{u}}$ as:')
g1Sy = MakeMatrixOfSymbols(r'g_{1}', n, 1, [t])

display(mFullSymbol)
#xDotSy = SymbolicProblem.CreateCoVector(x, r'\dot{x}', t)
#xDot = g1Sy+ aSy*mFullSymbol*uSy
#jh.printMarkdown("Filling in our Hamiltonian, we get the following expression for our optimal thrust direction:")
#lambdas = sy.Matrix([[r'\lambda_{1}',r'\lambda_{2}',r'\lambda_{3}',r'\lambda_{4}',r'\lambda_{5}']]).transpose()
lambdas = SymbolicProblem.CreateCoVector(x, r'\lambda', t)
#lambdasSymbol = sy.Symbol(r'\lambda^T', commutative=False)
hamiltonin = lambdas.transpose()*zDot
# print(hamiltonin)
# display(hamiltonin)
# jh.showEquation("H", hamiltonin)
# stationaryCondition = sy.diff(hamiltonin, uSy)
# print(stationaryCondition)
# optU = sy.solve(stationaryCondition, uSy)
# jh.showEquation(uSy, optU)
#jh.printMarkdown(r'Sympy is having some trouble doing the derivative with MatrixSymbol\'s, so I\'ll explain instead.  The stationary condition will give us an expression for the optimal control, $\underline}{\hat{u}}$ by taking the partial derivative of H with respect to the control and setting it equal to zero. Then, solve for the control.  If we do that, noting that the control only appears with the $G_2$ term, and remembering that we want the normalized direction of the control vector, we get:')
jh.printMarkdown(r'Although normally we would take the partial derivative of the Hamiltonian with respect to the control, since the Hamiltonian is linear in the control, we need to take a more intuitive approch.  We want to maximize the $G_2$ term.  It ends up being $\lambda^T a G_2 u$.  Remembering that a is a magnitude scalar and u will be a normalized direction, we can drop it.  U is a 3 by 1 matrix, and $\lambda^T G_2$ will be a 1 by 3 matrix.  Clearly to maximize this term, the optimal u needs to be in the same direction as $\lambda^T G_2$, giving us our optimal u of')
uOpt = lambdas.transpose()*M / ((lambdas.transpose()*M).norm())
display(uOpt)

# jh.printMarkdown("Putting this back into our Hamiltonian, we get")
# hStar = (lambdas.transpose() * g1Sy)[0,0] + aSy*(uOpt.norm())
# jh.showEquation("H^{*}", hStar)
# jh.printMarkdown("Although not as cleanly defined as in the paper, we will soon be substituting expressions into this to create our equations of motion and boundary conditions.")




#accel = sy.Symbol('a', real=True, nonnegative=True)


#lambdas = sy.Matrix([[sy.Symbol(r'\lambda_a')],[sy.Symbol(r'\lambda_h')],[sy.Symbol(r'\lambda_k')],[sy.Symbol(r'\lambda_p')],[sy.Symbol(r'\lambda_q')]])
#acceleration= sy.Symbol('a')#sy.Matrix([[sy.Symbol('a_x'),sy.Symbol('a_y'),sy.Symbol('a_z')]])
# MtimesLambdas = mFullSymbol.transpose()*lambdas
# mTimesLambdasMagnitude = MtimesLambdas.norm()
# MNormalized =(MtimesLambdas)/(mTimesLambdasMagnitude)
# zDot = acceleration * (mFullSymbol*MNormalized)
# display(zDot.shape)
#%%
# dHdu = SymbolicProblem.CreateHamiltonianControlExpressionsStatic(hamiltonin, uSy)
# jh.showEquation("dh", dHdu)
#display(controlSolved)
optUOrg = lambdas.transpose()*M
optU = optUOrg/optUOrg.norm()
# jh.showEquation("u", optU.transpose())
#%%
#controlSolved = sy.solve(sy.Eq(0, dHdu), uSy)
fullSubsDictionary[uSy[0]]= optU[0]
fullSubsDictionary[uSy[1]]= optU[1]
fullSubsDictionary[uSy[2]]= optU[2]

#%%

# s = (1-k*sy.cos(F)-h*sy.sin(F))/(2*sy.pi)
# delSDelZ = sy.Matrix([[0, -sy.sin(F), -sy.cos(F), 0, 0, 0]]) *2*sy.pi
# zDotOverAnOrbit = -acceleration*sy.Integral((M*MNormalized)*s  , (F, -sy.pi, sy.pi))
# delZDotDelZ = acceleration*M.applyfunc(lambda s: sy.diff(s, F))*MNormalized
# display(delZDotDelZ.shape)



# jh.printMarkdown("In implimenting equation 40, note that the delZDot del z is made on a per-row basis.")
#z =[a, h, k, p, q, F]
# part2 = lambdas.transpose()*zDot*delSDelZ
# lmdDotArray = []
# print("starting 0")
# for i in range(0, n) :
#     delZDotDelZ = acceleration*M.applyfunc(lambda s: sy.diff(s, z[i]))*MNormalized 
#     part1 = (((lambdas.transpose()*delZDotDelZ) * s)[0])
#     fullIntegralOfThisEom = -sy.Integral(part1 + part2[0,i], (F, -sy.pi, sy.pi))
#     lmdDotArray.append(fullIntegralOfThisEom)
#     print("finished " + str(i))

lmdDotArray = []
print("starting 0")
for i in range(0, n) :
    fullIntegralOfThisEom = hamiltonin.diff(x[i])[0]
    lmdDotArray.append(fullIntegralOfThisEom)
    print("finished " + str(i))

i=0
for expr in lmdDotArray:
    jh.showEquation(lambdas[i].diff(t), expr)
    i=i+1
    break

# now we try to integrate
#%%
accelVal = 9.8e-6
fullSubsDictionary[accelSy] = accelVal
fullSubsDictionary[mu]=muVal
#fullSubsDictionary[eccentricLongitude] = 0
# fullSubsDictionary[uSy[0]] = -accelVal
# fullSubsDictionary[uSy[1]] = 0
# fullSubsDictionary[uSy[2]] = accelVal/10
eoms = []

# for thisLmd in lambdas:
#     fullSubsDictionary[thisLmd] = 0



for i in range(0, len(x)):
    theEq = sy.Eq(x[i].diff(t), zDot[i])
    eoms.append(theEq)
    #jh.showEquation(theEq)
for i in range(0, len(lambdas)):
    eoms.append(sy.Eq(lambdas[i].diff(t), lmdDotArray[i]))
#%%
eom1 = eoms[0]
jh.showEquation(eom1)

actualSubsDic = {}
for k,v in fullSubsDictionary.items() :
    actualSubsDic[k] = SafeSubs(v, fullSubsDictionary)
fullSubsDictionary = actualSubsDic
#jh.showEquation(eom1.lhs, eom1.rhs.subs(fullSubsDictionary))
#jh.showEquation(eom1.lhs, eom1.rhs.expand().subs(fullSubsDictionary))
#jh.showEquation(eom1.lhs, eom1.rhs.subs(actualSubsDic))
# jh.showEquation("u1", optUOrg[0]/optU.norm())
# jh.showEquation("u1", optUOrg[0]/optU.norm().subs(fullSubsDictionary))
# jh.showEquation("u1", optUOrg[0]/optU.norm().subs(fullSubsDictionary).doit())


#%%
lmdHelper = OdeLambdifyHelperWithBoundaryConditions(t, sy.Symbol('t_0', real=True), sy.Symbol('t_f', real=True), eoms, [], [], fullSubsDictionary)

z0 = SafeSubs(z, {t: lmdHelper.t0})
zF = SafeSubs(z, {t: lmdHelper.tf})

a0V = 7000
h0V = 0.0
k0V = 0
p0V = 0
q0V = 28.5*math.pi/180.0
lon0= -2.274742851
t0V = 2444239.0 * 86400

afV = 42000
hfV = 0
kfV = 0.001
pfV = 1*math.pi/180.0
qfV = 0
tfV = 0

lmdHelper.BoundaryConditionExpressions.append(zF[0]-afV)
lmdHelper.BoundaryConditionExpressions.append(zF[1])
lmdHelper.BoundaryConditionExpressions.append(zF[2])
lmdHelper.BoundaryConditionExpressions.append(zF[3])
lmdHelper.BoundaryConditionExpressions.append(zF[4])

# #%%
# zDotFinal =zDot[0]
# for k,v in fullSubsDictionary.items():
#     display(k)
#     zDotFinal = zDotFinal.subs(k, fullSubsDictionary[k])    
# jh.showEquation("z", zDotFinal)


lmdGuess = [4.675229762, 5.413413947e-2, -9.202702084e-3, 1.778011878e1, -2.258455855e-4, -2.274742851]

fullInitialState = [a0V, h0V, k0V, p0V, q0V]
fullInitialState.extend(lmdGuess)
print("read to lambidfy")

initialState = [a0V, h0V,k0V, p0V, q0V, lon0 ]
initialState.extend(lmdGuess)
integratorCallback = lmdHelper.CreateSimpleCallbackForSolveIvp()
dxAtStart = integratorCallback(0, initialState)
display(dxAtStart)


tArray = np.linspace(0.0, 48089.90058, 6000)
solution = solve_ivp(integratorCallback, [tArray[0], tArray[-1]], initialState, t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
print(solution)





#azimuthPlotDataSim = prim.XAndYPlottableLineData(time, np.array(simOtherValues[stateSymbols[7]])*180.0/math.pi, "azimuth_sim", '#ff00ff', 2, 0)
#elevationPlotDataSim = prim.XAndYPlottableLineData(time, np.array(simOtherValues[stateSymbols[8]])*180.0/math.pi, "elevation_sim", '#ffff00', 2, 0)

plot2DLines([prim.XAndYPlottableLineData(tArray/86400, solution.y[0], "sma", '#0000ff', 2, 0)], "sma (km)")
plot2DLines([prim.XAndYPlottableLineData(tArray/86400, solution.y[1], "h", '#0000ff', 2, 0)], "h")
plot2DLines([prim.XAndYPlottableLineData(tArray/86400, solution.y[2], "k", '#0000ff', 2, 0)], "k")
plot2DLines([prim.XAndYPlottableLineData(tArray/86400, solution.y[3], "p", '#0000ff', 2, 0)], "p")
plot2DLines([prim.XAndYPlottableLineData(tArray/86400, solution.y[4], "q", '#0000ff', 2, 0)], "q")
plot2DLines([prim.XAndYPlottableLineData(tArray/86400, solution.y[5], "lon", '#0000ff', 2, 0)], "Lon (rad)")
equiElements = []
for i in range(0, len(tArray)):    
    temp = mee.EquinoctialElementsHalfI(solution.y[0][i], solution.y[1][i], solution.y[2][i],solution.y[3][i],solution.y[4][i],solution.y[5][i],  muVal)
    #realEqui = scaleEquinoctialElements(temp, 1.0, 1.0)
    equiElements.append(temp)
motions = mee.EquinoctialElementsHalfI.CreateEphemeris(equiElements)
satEphemeris = prim.EphemerisArrays()
satEphemeris.InitFromMotions(tArray, motions)
satPath = prim.PathPrimitive(satEphemeris)
satPath.color = "#ff00ff"



import plotly.graph_objects as go #type: ignore
def ms(x, y, z, radius, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

earth = ms(0,0,0, 6378.137, 30)
sphere = go.Surface()
fig = PlotAndAnimatePlanetsWithPlotly("Orbiting Earth", [satPath], tArray, None)
fig.add_surface(x=earth[0], y=earth[1], z=earth[2], opacity=1.0, autocolorscale=False, showlegend=False)
fig.update_xaxes(visible=False, showticklabels=False)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
fig['layout']['sliders'][0]['pad']=dict(r= 0, t= 0, b=0, l=0)
fig['layout']['updatemenus'][0]['pad']=dict(r= 0, t= 0, b=0, l=0)
fig.show()  

