#%%
import sympy as sy
import os
import sys
import math
from collections import OrderedDict
sy.init_printing()
from typing import Union, List, Optional, Sequence
from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
import pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule as mee
from IPython.display import display
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions
import scipyPaperPrinter as jh #type: ignore
#jh.printMarkdown("# SEPSPOT Recreation")
#jh.printMarkdown("In working my way up through low-thrust modeling for satellite maneuvers, it is inevitable to run into Dr. Edelbaum's work.  Newer work such as Jean Albert Kechichian's practically requires understanding SEPSPOT as a prerequesit.  This writeup will go through the basics of SEPSPOT's algorithsm as described in the references below.")

#jh.printMarkdown("In other work in this python library, I have already created many helper types such as Equinoctial elements, their equations of motion, rotation matrices, and more. To start, we will define out set of equinoctial elements.  Unlike the orignial paper, I will be using the modified elements.  This replaces the semi-major axis with the parameter and reorders/renames some of the other elements.")
t=sy.Symbol('t', real=True)
mu = sy.Symbol(r'\mu', real=True, positive=True)
muVal = 3.986004418e5  
#kepElements = KepModule.CreateSymbolicElements(t, mu)
fullSubsDictionary = OrderedDict() #type: ignore

simpleBoringEquiElements = mee.EquinoctialElementsHalfI.CreateSymbolicElements(t, mu)
a = simpleBoringEquiElements.SemiMajorAxis
h = simpleBoringEquiElements.EccentricitySinTermH
k = simpleBoringEquiElements.EccentricityCosTermK
p = simpleBoringEquiElements.InclinationSinTermP
q = simpleBoringEquiElements.InclinationCosTermQ
n = sy.sqrt(mu/(a**3))
x = sy.Matrix([[simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermK, simpleBoringEquiElements.InclinationSinTermP, simpleBoringEquiElements.InclinationCosTermQ]]).transpose()

beta = simpleBoringEquiElements.BetaSy
eccentricLongitude = simpleBoringEquiElements.Longitude
simpleBoringEquiElements.F = eccentricLongitude  #type: ignore

r = simpleBoringEquiElements.CreateFgwToInertialAxes()

x1Sy = sy.Function('X_1')(a, h, k, eccentricLongitude)
x2Sy = sy.Function('X_2')(a, h, k, eccentricLongitude)
x1DotSy = sy.Function(r'\dot{X_1}')(a, h, k, eccentricLongitude)
x2DotSy = sy.Function(r'\dot{X_2}')(a, h, k, eccentricLongitude)

[x1SimpleEqui, x2SimpleEqui] = simpleBoringEquiElements.RadiusInFgw(eccentricLongitude, fullSubsDictionary)
[x1DotSimpleEqui, x2DotSimpleEqui] = simpleBoringEquiElements.VelocityInFgw(eccentricLongitude, fullSubsDictionary)

fullSubsDictionary[x1Sy] = x1SimpleEqui
fullSubsDictionary[x2Sy] = x2SimpleEqui
fullSubsDictionary[x1DotSy] = x1DotSimpleEqui
fullSubsDictionary[x2DotSy] = x2DotSimpleEqui

jh.showEquation(x1Sy, x1SimpleEqui)
jh.showEquation(x2Sy, x1SimpleEqui)
jh.showEquation(x1DotSy, x1SimpleEqui)
jh.showEquation(x2DotSy, x1SimpleEqui)


def makeMatrixOfSymbols(baseString : str, rows: int, cols : int,funcArgs : Optional[List[sy.Expr]]= None) :
    endString = ''
    if baseString.endswith('}') :
        baseString = baseString[:-1]
        endString = '}'
    mat = sy.Matrix.zeros(rows, cols)
    for r in range(0, rows) :
        for c in range(0, cols):
            if funcArgs== None :
                mat[r,c] = sy.Symbol(baseString + "_{" + str(r) + "," + str(c)+"}" + endString)
            else:
                mat[r,c] = sy.Function(baseString + "_{" + str(r) + "," + str(c)+"}"+ endString)(*funcArgs)
    return mat

rotMatrix = mee.EquinoctialElementsHalfI.CreateFgwToInertialAxesStatic(p, q)
fHatSy = makeMatrixOfSymbols(r'\hat{f}', 3, 1, [p, q])
gHatSy = makeMatrixOfSymbols(r'\hat{g}', 3, 1, [p, q])
wHatSy = makeMatrixOfSymbols(r'\hat{w}', 3, 1, [p, q])
display(fHatSy)
for i in range(0, 3):
    fullSubsDictionary[fHatSy[i]] = rotMatrix.col(0)[i]
    fullSubsDictionary[gHatSy[i]] = rotMatrix.col(1)[i]
    fullSubsDictionary[wHatSy[i]] = rotMatrix.col(2)[i]

jh.showEquation(fHatSy, rotMatrix.col(0))
jh.showEquation(gHatSy, rotMatrix.col(1))
jh.showEquation(wHatSy, rotMatrix.col(2))


x1 = x1Sy
y1 = x2Sy
xDot = x1DotSy
yDot = x2DotSy
F = eccentricLongitude
dX1dh = a*(-h*beta*sy.cos(F)- (beta+(h**2)*beta**3)*(h*sy.cos(F)-k*sy.sin(F))/(1-beta))
dY1dh = a*( k*beta*sy.cos(F)-1      +h*k* (beta**3)*(h*sy.cos(F)-k*sy.sin(F))/(1-beta))
dX1dk = a*( h*beta*sy.sin(F)-1      -h*k* (beta**3)*(h*sy.cos(F)-k*sy.sin(F))/(1-beta))
dY1dk = a*(-k*beta*sy.sin(F)+beta+((k**2)*(beta**3)*(h*sy.cos(F)-k*sy.sin(F))/(1-beta)))
m11 = 2*xDot/(n*n*a)
m12 = 2*yDot/(n*n*a)
m13 = 0
m21 = (sy.sqrt(1-h**2-k**2)/(n*a**2))*(dX1dk+(xDot/n)*(sy.sin(F)-h*beta))
m22 = (sy.sqrt(1-h**2-k**2)/(n*a**2))*(dY1dk+(yDot/n)*(sy.sin(F)-h*beta))
m23 = k*(q*y1-p*x1)/(n*(a**2)*sy.sqrt(1-h**2-k**2))
m31 = -1*(sy.sqrt(1-h**2-k**2)/(n*a**2))*(dX1dh-(xDot/n)*(sy.cos(F)-k*beta))
m32 = -1*(sy.sqrt(1-h**2-k**2)/(n*a**2))*(dY1dh-(yDot/n)*(sy.cos(F)-k*beta))
m33 = -1*h*(q*y1-p*x1)/(n*(a**2)*sy.sqrt(1-h**2-k**2))
m41 = 0
m42 = 0
m43 = (1+p**2+q**2)*y1/(2*n*a**2*sy.sqrt(1-h**2-k**2))
m51 = 0
m52 = 0
m53 = (1+p**2+q**2)*x1/(2*n*a**2*sy.sqrt(1-h**2-k**2))


M = sy.Matrix([[m11, m12, m13], [m21, m22, m23],[m31, m32, m33],[m41, m42, m43],[m51, m52, m53]])
jh.showEquation("M", M)
display(M.shape)

mFullSymbol = makeMatrixOfSymbols("M", 5, 3, [a,h,k,p,q])


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
n = 5
jh.printMarkdown("Staring with our x:")
xSy = sy.MatrixSymbol('x', n, 1)
jh.showEquation(xSy, x)
jh.printMarkdown(r'We write our $\underline{\dot{x}}$ with the assumed optimal control vector $\underline{\hat{u}}$ as:')
g1Sy = makeMatrixOfSymbols(r'g_{1}', n, 1, [t])
aSy = sy.Function('A', commutative=True)(x, t)
uSy = sy.Matrix([["u1", "u2", "u3"]]).transpose()
display(mFullSymbol)
xDotSy = SymbolicProblem.CreateCoVector(x, r'\dot{x}', t)
xDot = g1Sy+ aSy*mFullSymbol*uSy
jh.printMarkdown("Filling in our Hamiltonian, we get the following expression for our optimal thrust direction:")
#lambdas = sy.Matrix([[r'\lambda_{1}',r'\lambda_{2}',r'\lambda_{3}',r'\lambda_{4}',r'\lambda_{5}']]).transpose()
lambdas = SymbolicProblem.CreateCoVector(x, r'\lambda', t)
#lambdasSymbol = sy.Symbol(r'\lambda^T', commutative=False)
hamiltonin = lambdas.transpose()*xDot
# print(hamiltonin)
# display(hamiltonin)
# jh.showEquation("H", hamiltonin)
# stationaryCondition = sy.diff(hamiltonin, uSy)
# print(stationaryCondition)
# optU = sy.solve(stationaryCondition, uSy)
# jh.showEquation(uSy, optU)
#jh.printMarkdown(r'Sympy is having some trouble doing the derivative with MatrixSymbol\'s, so I\'ll explain instead.  The stationary condition will give us an expression for the optimal control, $\underline}{\hat{u}}$ by taking the partial derivative of H with respect to the control and setting it equal to zero. Then, solve for the control.  If we do that, noting that the control only appears with the $G_2$ term, and remembering that we want the normalized direction of the control vector, we get:')
jh.printMarkdown(r'Although normally we would take the partial derivative of the Hamiltonian with respect to the control, since the Hamiltonian is linear in the control, we need to take a more intuitive approch.  We want to maximize the $G_2$ term.  It ends up being $\lambda^T a G_2 u$.  Remembering that a is a magnitude scalar and u will be a normalized direction, we can drop it.  U is a 3 by 1 matrix, and $\lambda^T G_2$ will be a 1 by 3 matrix.  Clearly to maximize this term, the optimal u needs to be in the same direction as $\lambda^T G_2$, giving us our optimal u of')
uOpt = lambdas.transpose()*mFullSymbol / ((lambdas.transpose()*mFullSymbol).norm())
display(uOpt)

jh.printMarkdown("Putting this back into our Hamiltonian, we get")
hStar = (lambdas.transpose() * g1Sy)[0,0] + aSy*(uOpt.norm())
jh.showEquation("H^{*}", hStar)
jh.printMarkdown("Although not as cleanly defined as in the paper, we will soon be substituting expressions into this to create our equations of motion and boundary conditions.")




#accel = sy.Symbol('a', real=True, nonnegative=True)


#lambdas = sy.Matrix([[sy.Symbol(r'\lambda_a')],[sy.Symbol(r'\lambda_h')],[sy.Symbol(r'\lambda_k')],[sy.Symbol(r'\lambda_p')],[sy.Symbol(r'\lambda_q')]])
acceleration= sy.Symbol('a')#sy.Matrix([[sy.Symbol('a_x'),sy.Symbol('a_y'),sy.Symbol('a_z')]])
MtimesLambdas = mFullSymbol.transpose()*lambdas
mTimesLambdasMagnitude = MtimesLambdas.norm()
MNormalized =(MtimesLambdas)/(mTimesLambdasMagnitude)
zDot = acceleration * (mFullSymbol*MNormalized)
display(zDot.shape)



s = (1-k*sy.cos(F)-h*sy.sin(F))/(2*sy.pi)
delSDelZ = sy.Matrix([[0, -sy.sin(F), -sy.cos(F), 0, 0]]) *2*sy.pi
zDotOverAnOrbit = -acceleration*sy.Integral((M*MNormalized)*s  , (F, -sy.pi, sy.pi))
delZDotDelZ = acceleration*M.applyfunc(lambda s: sy.diff(s, F))*MNormalized
display(delZDotDelZ.shape)



jh.printMarkdown("In implimenting equation 40, note that the delZDot del z is made on a per-row basis.")
z = [a, h, k, p, q]
part2 = lambdas.transpose()*zDot*delSDelZ
lmdDotArray = []
print("starting 0")
for i in range(0, 5) :
    delZDotDelZ = acceleration*M.applyfunc(lambda s: sy.diff(s, z[i]))*MNormalized 
    part1 = (((lambdas.transpose()*delZDotDelZ) * s)[0])
    fullIntegralOfThisEom = -sy.Integral(part1 + part2[0,i], (F, -sy.pi, sy.pi))
    lmdDotArray.append(fullIntegralOfThisEom)
    print("finished " + str(i))

i=0
for expr in lmdDotArray:
    jh.showEquation(lambdas[i].diff(t), expr)
    i=i+1
    break

# now we try to integrate
#%%
accelVal = 9.798e-4  #units are km, sec
fullSubsDictionary[acceleration] = accelVal
fullSubsDictionary[mu]=muVal
fullSubsDictionary[eccentricLongitude] = 0
fullSubsDictionary[uSy[0]] = 0.0001
fullSubsDictionary[uSy[1]] = 0
fullSubsDictionary[uSy[2]] = 0
eoms = []

for thisLmd in lambdas:
    fullSubsDictionary[thisLmd] = 0

justStateEom = M*uSy
for i in range(0, len(x)):
    theEq = sy.Eq(x[i].diff(t), justStateEom[i])
    eoms.append(theEq)
    jh.showEquation(theEq)
# for i in range(0, len(lambdas)):
#     eoms.append(sy.Eq(lambdas[i].diff(t), lmdDotArray[i]))
lmdHelper = OdeLambdifyHelperWithBoundaryConditions(t, sy.Symbol('t_0', real=True), sy.Symbol('t_f', real=True), eoms, [], [], fullSubsDictionary)

z0 = SymbolicProblem.SafeSubs(z, {t: lmdHelper.t0})
zF = SymbolicProblem.SafeSubs(z, {t: lmdHelper.tf})

a0V = 10509.0
h0V = 0.325
k0V = 0
p0V = 28.5*math.pi/180.0
q0V = 0
t0V = 2444239.0 * 86400

afV = 42241.19
hfV = 0
kfV = 0
pfV = 0
qfV = 0
tfV = 0

lmdHelper.BoundaryConditionExpressions.append(zF[0]-afV)
lmdHelper.BoundaryConditionExpressions.append(zF[1])
lmdHelper.BoundaryConditionExpressions.append(zF[2])
lmdHelper.BoundaryConditionExpressions.append(zF[3])
lmdHelper.BoundaryConditionExpressions.append(zF[4])


lmdGuess = [20.0,0.01,0.18,8.0,85.0]

fullInitialState = [a0V, h0V, k0V, p0V, q0V]
fullInitialState.extend(lmdGuess)
print("read to lambidfy")

integratorCallback = lmdHelper.CreateSimpleCallbackForSolveIvp()

from scipy.integrate import solve_ivp #type: ignore
import numpy as np
initialState = [a0V, h0V,k0V, p0V, q0V ]
tArray = np.linspace(0.0, 24*86400, 1200)
solution = solve_ivp(integratorCallback, [0, 24*86400], initialState, t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
print(solution)
dxAtStart = integratorCallback(0, initialState)
display(dxAtStart)

