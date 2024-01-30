#%%
import sympy as sy
import os
import sys
sys.path.append('../')
sys.path.append('../../')
import math
from collections import OrderedDict
sy.init_printing()
from typing import Union, List, Optional, Sequence, cast, Dict, Iterator
import pyeq2orb
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
from scipy.optimize import fsolve  #type: ignore
jh.printMarkdown("# SEPSPOT Recreation")
#jh.printMarkdown("In working my way up through low-thrust modeling for satellite maneuvers, it is inevitable to run into Dr. Edelbaum's work.  Newer work such as Jean Albert Kechichian's practically requires understanding SEPSPOT as a prerequesit.  This writeup will go through the basics of SEPSPOT's algorithsm as described in the references below.")



#jh.printMarkdown("In other work in this python library, I have already created many helper types such as Equinoctial elements, their equations of motion, rotation matrices, and more. To start, we will define out set of equinoctial elements.  Unlike the orignial paper, I will be using the modified elements.  This replaces the semi-major axis with the parameter and reorders/renames some of the other elements.")
t=sy.Symbol('t', real=True)
mu = sy.Symbol(r'\mu', real=True, positive=True)
muVal = 3.986004418e5  
#kepElements = KepModule.CreateSymbolicElements(t, mu)
fullSubsDictionary = OrderedDict() #type: dict[sy.Expr, SymbolOrNumber]

simpleBoringEquiElements = mee.EquinoctialElementsHalfITrueLongitude.CreateSymbolicElements(t, mu)
a = cast(sy.Expr, simpleBoringEquiElements.SemiMajorAxis)
h = cast(sy.Expr, simpleBoringEquiElements.EccentricitySinTermH)
k = cast(sy.Expr, simpleBoringEquiElements.EccentricityCosTermK)
p = cast(sy.Expr, simpleBoringEquiElements.InclinationSinTermP)
q = cast(sy.Expr, simpleBoringEquiElements.InclinationCosTermQ)
F = cast(sy.Expr, simpleBoringEquiElements.TrueLongitude)
n = sy.sqrt(mu/(a**3))
x = sy.Matrix([[simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermK, simpleBoringEquiElements.InclinationSinTermP, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.TrueLongitude]]).transpose()
z = [simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermK, simpleBoringEquiElements.InclinationSinTermP, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.TrueLongitude]
beta = simpleBoringEquiElements.BetaSy
betaExp = simpleBoringEquiElements.Beta

rotMatrix = mee.EquinoctialElementsHalfITrueLongitude.CreateFgwToInertialAxesStatic(p, q)
fHatSy = MakeMatrixOfSymbols(r'\hat{f}', 3, 1, [p, q])
gHatSy = MakeMatrixOfSymbols(r'\hat{g}', 3, 1, [p, q])
wHatSy = MakeMatrixOfSymbols(r'\hat{w}', 3, 1, [p, q])
display(fHatSy)
for i in range(0, 3):
    fullSubsDictionary[fHatSy[i]] = rotMatrix.col(0)[i]
    fullSubsDictionary[gHatSy[i]] = rotMatrix.col(1)[i]
    fullSubsDictionary[wHatSy[i]] = rotMatrix.col(2)[i]
#%%
F = sy.Symbol('F', real=True)
G = sy.Symbol('G')
L = simpleBoringEquiElements.TrueLongitude
r = a*G**2/(1+h*sy.sin(L)+k*sy.cos(L))

B = 1/(1+G)

cosL = (a/r)*((1-B*h**2)*sy.cos(F)+h*k*B*sy.sin(F)-k)
sinL = (a/r)*(h*k*B*sy.cos(F)+(1-B*k**2)*sy.sin(F)-h)
sf = h+(r/a)*((1-B*h**2)*sy.sin(L)-h*k*B*sy.cos(L))/(G)
cf = k+(r/a)*((1-B*k**2)*sy.cos(L)-h*k*B*sy.sin(L))/(G)
drdk_F = -a*cf
drdk_L = -r*(2*a*k+r*sy.cos(L))/(a*G**2)



jh.showEquation(r'\frac{dr}{dk}_F', (drdk_F/k).simplify())
jh.showEquation(r'\frac{dr}{dk}_L', (drdk_L/k).simplify())
jh.showEquation("s", (drdk_F - drdk_L).simplify().trigsimp(deep=True))

r_f = a*(1-k*cf-h*sf)
jh.showEquation("s",(r-r_f).simplify())

#%%

jh.showEquation(fHatSy, rotMatrix.col(0))
jh.showEquation(gHatSy, rotMatrix.col(1))
jh.showEquation(wHatSy, rotMatrix.col(2))
#M = simpleBoringEquiElements.CreatePerturbationMatrixWithMeanLongitude(f, fullSubsDictionary)
rOverA = simpleBoringEquiElements.ROverA
#taDifeq = n*sy.sqrt(1-h**2-k**2)/(rOverA**2)
aSy = sy.Function('A', commutative=True)(x, t)
u1 = sy.Symbol("u_1", real=True)
u2 = sy.Symbol("u_2", real=True)
u3 = sy.Symbol("u_3", real=True)
uSy = sy.Matrix([[u1, u2, u3]]).transpose()
accelSy = sy.Symbol('a', real=True, positive=True)



def CreatePerturbationMatrixWithTrueLongitude(eelm, subsDict : Dict[sy.Expr, SymbolOrNumber]) ->sy.Matrix:
    p = eelm.InclinationSinTermP
    q = eelm.InclinationCosTermQ
    h = eelm.EccentricitySinTermH
    k = eelm.EccentricityCosTermK
    mu = eelm.GravitationalParameter
    a = eelm.SemiMajorAxis
    l = eelm.TrueLongitude

    G = sy.sqrt(1-h**2-k**2)
    K = (1+p**2+q**2)
    n = eelm.NSy
    subsDict[n] = eelm.N
    GExp = G
    G = sy.Function("G")(h, k)
    subsDict[G] = GExp

    KExp = K
    K = sy.Function("K")(p, q)
    subsDict[K] = KExp

    r = sy.Function('r')(a)
    subsDict[r] = eelm.ROverA*a

    sl = sy.sin(eelm.TrueLongitude)
    cl = sy.cos(eelm.TrueLongitude)
    #u is radial, intrack, out of plane, AKA r, theta, h

    onephspkc = 1+h*sl+k*cl
    aDotMult = (2/(n*G))
    b11 = aDotMult*(k*sl-h*cl) #aDot in r direction
    b12 = aDotMult*(onephspkc) #aDot in theta direction
    b13 = 0  # a dot in h direction, you get the pattern...

    hDotMult = G/(n*a*onephspkc)
    b21 = hDotMult*(-(onephspkc)*cl)
    b22 = hDotMult*((h+(2+h*sl+k*cl)*sl))
    b23 = -hDotMult*(k*(p*cl-q*sl))

    kDotMult = G/(n*a*onephspkc)
    b31 = kDotMult*((onephspkc)*sl)
    b32 = kDotMult*((k+(2+h*sl+k*cl)*cl))
    b33 = kDotMult*(h*(p*cl-q*sl)) 

    pDotMult = G/(2*n*a*onephspkc)
    b41 = 0
    b42 = 0
    b43 = pDotMult*K*sl
    
    qDotMult = G/(2*n*a*onephspkc)
    b51 = 0
    b52 = 0
    b53 = qDotMult*K*cl
    
    b61 = 0
    b62 = 0
    b63 = (G*(q*sl-p*cl))/(n*a*onephspkc)
    #b63 = r*(q*sl-p*cl)/(n*G*a**2)

    #M = sy.Matrix([[m11, m12, m13], [m21, m22, m23],[m31, m32, m33],[m41, m42, m43],[m51, m52, m53]])
    B = sy.Matrix([[b11, b12, b13], [b21, b22, b23],[b31, b32, b33],[b41, b42, b43],[b51, b52, b53],[b61, b62, b63]])   
    return B     

def CreatePerturbationMatrixWithTrueLongitudeDirectlyFromBook(eelm, subsDict : Dict[sy.Expr, SymbolOrNumber]) ->sy.Matrix:
    p = eelm.InclinationSinTermP
    q = eelm.InclinationCosTermQ
    h = eelm.EccentricitySinTermH
    k = eelm.EccentricityCosTermK
    mu = eelm.GravitationalParameter
    a = eelm.SemiMajorAxis
    l = eelm.TrueLongitude

    G = sy.sqrt(1-h**2-k**2)
    K = (1+p**2+q**2)
    n = eelm.NSy
    subsDict[n] = eelm.N
    GExp = G
    G = sy.Function("G")(h, k)
    subsDict[G] = GExp

    KExp = K
    K = sy.Function("K")(p, q)
    subsDict[K] = KExp

    r = sy.Function('r')(a)
    subsDict[r] = eelm.ROverA*a

    sl = sy.sin(eelm.TrueLongitude)
    cl = sy.cos(eelm.TrueLongitude)
    #u is radial, intrack, out of plane, AKA r, theta, h

    onephspkc = 1+h*sl+k*cl
    aDotMult = (2/(n*G))
    b11 = aDotMult*(k*sl-h*cl) #aDot in r direction
    b12 = 2*a*G/(n*r) #aDot in theta direction
    b13 = 0  # a dot in h direction, you get the pattern...

    hDotMult = G/(n*a*onephspkc)
    b21 = hDotMult*(-(1+onephspkc)*cl)
    b22 = r*(h+sl)/(n*G*a**2)+G*sl/(n*a)
    b23 = r*k*(p*cl-q*sl)/(n*G*a**2)

    kDotMult = G/(n*a*onephspkc)
    b31 = G*sl/(n*a)
    b32 = r*(k+cl)/(n*G*a**2)+G*cl/(n*a)
    b33 = r*h*(p*cl-q*sl/(n*G*a**2))

    pDotMult = G/(2*n*a*onephspkc)
    b41 = 0
    b42 = 0
    b43 = r*K*sl/(2*n*G*a**2)
    
    qDotMult = G/(2*n*a*onephspkc)
    b51 = 0
    b52 = 0
    b53 = r*K*cl/(2*n*G*a**2)
    
    b61 = 0
    b62 = 0
    b63 = r*(q*sl-p*cl)/(n*G*a**2)

    #M = sy.Matrix([[m11, m12, m13], [m21, m22, m23],[m31, m32, m33],[m41, m42, m43],[m51, m52, m53]])
    B = sy.Matrix([[b11, b12, b13], [b21, b22, b23],[b31, b32, b33],[b41, b42, b43],[b51, b52, b53],[b61, b62, b63]])   
    return B     

def UnperturbedTrueLongitudeTimeDerivative(eelm, subsDict : Optional[Dict[sy.Expr, SymbolOrNumber]]=None) ->sy.Expr :
    p = eelm.InclinationSinTermP
    q = eelm.InclinationCosTermQ
    h = eelm.EccentricitySinTermH
    k = eelm.EccentricityCosTermK
    mu = eelm.GravitationalParameter
    a = eelm.SemiMajorAxis
    l = eelm.TrueLongitude 
    n = eelm.N
    sl = sy.sin(l)
    cl = sy.cos(l)
    return (n*(1+h*sl+k*cl)**2)/(1-h**2-k**2)**(3/2)

def UnperturbedTrueLongitudeTimeDerivativeWithWeirdRadius(eelm, subsDict : Dict[sy.Expr, SymbolOrNumber]) ->sy.Expr :
    p = eelm.InclinationSinTermP
    q = eelm.InclinationCosTermQ
    h = eelm.EccentricitySinTermH
    k = eelm.EccentricityCosTermK
    mu = eelm.GravitationalParameter
    a = eelm.SemiMajorAxis
    f = eelm.EccentricLongitude
    l = eelm.TrueLongitude 
    n = eelm.N
    sl = sy.sin(eelm.TrueLongitude)
    cl = sy.cos(eelm.TrueLongitude)

    r = sy.Function('r')(a, f, k)
    subsDict[r] = eelm.ROverA*a

    return n*(a**2)*sy.sqrt(1-h**2-k**2)/(r**2)

#B = CreatePerturbationMatrixWithTrueLongitudeDirectlyFromBook(simpleBoringEquiElements, fullSubsDictionary)
#lonDot = sy.Matrix([[0],[0],[0],[0],[0],[1]])*UnperturbedTrueLongitudeTimeDerivativeWithWeirdRadius(simpleBoringEquiElements, fullSubsDictionary)

B = CreatePerturbationMatrixWithTrueLongitude(simpleBoringEquiElements, fullSubsDictionary)
lonDot = sy.Matrix([[0],[0],[0],[0],[0],[1]])*UnperturbedTrueLongitudeTimeDerivative(simpleBoringEquiElements, fullSubsDictionary)



#%%

r = simpleBoringEquiElements.ROverA * simpleBoringEquiElements.SemiMajorAxis
jh.showEquation("r", r)

#%%
zDot = B*uSy*accelSy + lonDot

for i in range(0, 6):
    for j in range(0, 3):
        jh.showEquation("B_{" + str(i+1) +"," +str(j+1) + "}", B[i,j])
# zDot = M*uSy*accelSy + sy.Matrix([[0,0,0,0,0,taDifeq]]).transpose()
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
        existingArgs.append(someFunction) #TODO: Not sure about this
    return existingArgs

def createMatrixOfFunctionsFromDenseMatrix(someMatrix, argsICareAbout,stringName):
    mat = sy.Matrix.zeros(*someMatrix.shape)
    for r in range(0, someMatrix.rows) :
        for c in range(0, someMatrix.cols):
            thisElementName = stringName + "_{" + str(r) + "," + str(c)+"}"            
            mat[r,c] =sy.Function(thisElementName)(*recurseArgs(someMatrix[r,c], argsICareAbout, []))
    return mat

mFullSymbol = createMatrixOfFunctionsFromDenseMatrix(B, x, "B")


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
hamiltonian = lambdas.transpose()*zDot
# print(hamiltonian)
# display(hamiltonian)
# jh.showEquation("H", hamiltonian)
# stationaryCondition = sy.diff(hamiltonian, uSy)
# print(stationaryCondition)
# optU = sy.solve(stationaryCondition, uSy)
# jh.showEquation(uSy, optU)
#jh.printMarkdown(r'Sympy is having some trouble doing the derivative with MatrixSymbol\'s, so I\'ll explain instead.  The stationary condition will give us an expression for the optimal control, $\underline}{\hat{u}}$ by taking the partial derivative of H with respect to the control and setting it equal to zero. Then, solve for the control.  If we do that, noting that the control only appears with the $G_2$ term, and remembering that we want the normalized direction of the control vector, we get:')
jh.printMarkdown(r'Although normally we would take the partial derivative of the Hamiltonian with respect to the control, since the Hamiltonian is linear in the control, we need to take a more intuitive approch.  We want to maximize the $G_2$ term.  It ends up being $\lambda^T a G_2 u$.  Remembering that a is a magnitude scalar and u will be a normalized direction, we can drop it.  U is a 3 by 1 matrix, and $\lambda^T G_2$ will be a 1 by 3 matrix.  Clearly to maximize this term, the optimal u needs to be in the same direction as $\lambda^T G_2$, giving us our optimal u of')

optUOrg = lambdas.transpose()*B
optU =optUOrg/optUOrg.norm()

#controlSolved = sy.solve(sy.Eq(0, dHdu), uSy)
fullSubsDictionary[uSy[0]]= optU[0]
fullSubsDictionary[uSy[1]]= optU[1]
fullSubsDictionary[uSy[2]]= optU[2]




lmdDotArray = []
print("starting 0")
for i in range(0, n) :
    fullIntegralOfThisEom = -1*hamiltonian.diff(x[i])[0]
    lmdDotArray.append(fullIntegralOfThisEom)
    print("finished " + str(i))

i=0
for expr in lmdDotArray:
    jh.showEquation(lambdas[i].diff(t), expr)
    i=i+1
    break

# now we try to integrate
#%%
accelVal = 9.8e-5
fullSubsDictionary[accelSy] = accelVal
fullSubsDictionary[mu]=muVal

eoms = []

for i in range(0, len(x)):
    theEq = sy.Eq(x[i].diff(t), zDot[i])
    eoms.append(theEq)
    #jh.showEquation(theEq)
for i in range(0, len(lambdas)):
    eoms.append(sy.Eq(lambdas[i].diff(t), lmdDotArray[i]))

eom1 = eoms[0]
#jh.showEquation(eom1)

actualSubsDic = {}
for k,v in fullSubsDictionary.items() :
    actualSubsDic[k] = SafeSubs(v, fullSubsDictionary)
fullSubsDictionary = actualSubsDic




initialKepElements = KeplerianElements(7000, 0.0, 28.5*math.pi/180.0, 0, 0, -2.274742851, muVal)
initialModifiedEquiElements = mee.ConvertKeplerianToEquinoctial(initialKepElements)
initialEquiElements = mee.EquinoctialElementsHalfITrueLongitude.FromModifiedEquinoctialElements(initialModifiedEquiElements)
a0V = float(initialEquiElements.SemiMajorAxis)
h0V = float(initialEquiElements.EccentricitySinTermH)
k0V = float(initialEquiElements.EccentricityCosTermK)
p0V = float(initialEquiElements.InclinationSinTermP)
q0V = float(initialEquiElements.InclinationCosTermQ)
lon0= float(initialEquiElements.TrueLongitude)
#t0V = 2444239.0 * 86400

afV = 42000
hfV = 0.0
kfV = 0.001
pfV = math.tan((1*math.pi/180.0)/2)
qfV = 0.0
tfV = 0.0



#%%


problem = SymbolicProblem()
problem.TimeSymbol = t
problem.TimeInitialSymbol = sy.Symbol('t_0', real=True)
problem.TimeFinalSymbol = sy.Symbol('t_f', real=True)
z0 = SafeSubs(z, {t: problem.TimeInitialSymbol})
zF = SafeSubs(z, {t: problem.TimeFinalSymbol})
problem.StateVariables.extend(x)
problem.StateVariables.extend(lambdas)
problem.StateVariableDynamics.extend(zDot)
problem.StateVariableDynamics.extend(lmdDotArray)

problem.BoundaryConditions.append(zF[0]-afV)
problem.BoundaryConditions.append(zF[1]-hfV)
problem.BoundaryConditions.append(zF[2]-kfV)
problem.BoundaryConditions.append(zF[3]-pfV)
problem.BoundaryConditions.append(zF[4]-qfV)
problem.BoundaryConditions.append(zF[5]*0)
for (k,v) in fullSubsDictionary.items():
    problem.SubstitutionDictionary[k] =v

scaleDict = {} #type: Dict[sy.Symbol, SymbolOrNumber]
for sv in problem.StateVariables :
    scaleDict[sv] = 1.0
originalProblem = problem
problem = problem.ScaleProblem(problem.StateVariables, scaleDict, problem.TimeFinalSymbol)

lmdHelper = OdeLambdifyHelperWithBoundaryConditions.CreateFromProblem(problem)

#lmdHelper.OtherArguments.append(originalProblem.TimeFinalSymbol)
lmdHelper.SubstitutionDictionary[originalProblem.TimeFinalSymbol] = originalProblem.TimeFinalSymbol
#lmdHelper = OdeLambdifyHelperWithBoundaryConditions(t, sy.Symbol('t_0', real=True), sy.Symbol('t_f', real=True), list(x), list(zDot), [], [], fullSubsDictionary)



# lmdHelper.BoundaryConditionExpressions.append(zF[0]-afV)
# lmdHelper.BoundaryConditionExpressions.append(zF[1]-hfV)
# lmdHelper.BoundaryConditionExpressions.append(zF[2]-kfV)
# lmdHelper.BoundaryConditionExpressions.append(zF[3]-pfV)
# lmdHelper.BoundaryConditionExpressions.append(zF[4]-qfV)
# lmdHelper.BoundaryConditionExpressions.append(zF[5]*0)
lmdHelper.SymbolsToSolveForWithBoundaryConditions.append(z0[5].subs(originalProblem.TimeInitialSymbol, lmdHelper.t0))
lmdHelper.SymbolsToSolveForWithBoundaryConditions.append(lambdas[0].subs(t, lmdHelper.t0))
lmdHelper.SymbolsToSolveForWithBoundaryConditions.append(lambdas[1].subs(t, lmdHelper.t0))
lmdHelper.SymbolsToSolveForWithBoundaryConditions.append(lambdas[2].subs(t, lmdHelper.t0))
lmdHelper.SymbolsToSolveForWithBoundaryConditions.append(lambdas[3].subs(t, lmdHelper.t0))
lmdHelper.SymbolsToSolveForWithBoundaryConditions.append(lambdas[4].subs(t, lmdHelper.t0))
#lmdHelper.SymbolsToSolveForWithBoundaryConditions.append(lambdas[5].subs(t, lmdHelper.t0))
#lmdHelper.OtherArguments.append(lambdas[5].subs(t, lmdHelper.t0))
# #%%
# zDotFinal =zDot[0]
# for k,v in fullSubsDictionary.items():
#     display(k)
#     zDotFinal = zDotFinal.subs(k, fullSubsDictionary[k])    
# jh.showEquation("z", zDotFinal)
# for i in range(0, len(lambdas)):
#     lmdHelper.AddStateVariable(lambdas[i], lmdDotArray[i])

lmdGuess = [4.675229762, 5.413413947e2, -9.202702084e3, 1.778011878e1, -2.268455855e4, -2.274742851]#-2.2747428]
#lmdGuess = [4.675229762, 8.413413947e2, -9.202702084e3, 1.778011878e1, -2.260455855e4, -2.2747428]
fullInitialState = [a0V, h0V, k0V, p0V, q0V]
fullInitialState.extend(lmdGuess)
print("read to lambdify")
#%%
tArray = np.linspace(0.0, 1.0, 800)
initialState = [a0V, h0V,k0V, p0V, q0V, lon0 ]
initialState.extend(lmdGuess)
tfV = 58089.9005
initialState.append(tfV)
from pyeq2orb.Numerical import ScipyCallbackCreators #type: ignore

ipvCallback = lmdHelper.CreateSimpleCallbackForSolveIvp()
def realIpvCallback(initialStateInCb) :
    #print(initialStateInCb)
    solution = solve_ivp(ipvCallback, [tArray[0], tArray[-1]], initialStateInCb, t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
    solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(lmdHelper.NonTimeLambdifyArguments, solution)
    return solution

solverCb = lmdHelper.createCallbackToSolveForBoundaryConditions(realIpvCallback, tArray, initialState)

#display(solverCb([1.0, 0.001, 0.001, 0.0]))
#display(lmdHelper.GetExpressionToLambdifyInMatrixForm())
#print(ipvCallback(0, [r0, u0, v0, lon0, 1.0, 0.001, 0.001, 0.0]))
#solution = solve_ivp(ipvCallback, [tArray[0], tArray[-1]], initialState, t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)

fSolveSol = fsolve(solverCb, initialState[5:12], epsfcn=0.0001, full_output=True, factor=0.01)

print(fSolveSol)










#print(fSolveSol)
#integratorCallback = lmdHelper.CreateSimpleCallbackForSolveIvp()
#fsolveCallback = lmdHelper.createCallbackToSolveForBoundaryConditions(integratorCallback, tArray, initialState)
#dxAtStart = integratorCallback(0, initialState)
#display(dxAtStart)

#def overallSolveIvpCallback(initialStateArray):
#    return solve_ivp(integratorCallback, [tArray[0], tArray[-1]], initialState, t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)

#fSolveInitialState = [58089.9005]
#fSolveInitialState.extend(initialState)
#fSolveSol = fsolve(fsolveCallback, fSolveInitialState, epsfcn=0.00001, full_output=True) # just to speed things up and see how the initial one works
#print(fSolveSol)
finalInitialState = [a0V, h0V,k0V, p0V, q0V]#, lon0 ]
finalInitialState.extend(fSolveSol[0])
finalInitialState.append(lmdGuess[5])
solution = solve_ivp(ipvCallback, [tArray[0], tArray[-1]], finalInitialState, t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
print(solution)



#azimuthPlotDataSim = prim.XAndYPlottableLineData(time, np.array(simOtherValues[stateSymbols[7]])*180.0/math.pi, "azimuth_sim", '#ff00ff', 2, 0)
#elevationPlotDataSim = prim.XAndYPlottableLineData(time, np.array(simOtherValues[stateSymbols[8]])*180.0/math.pi, "elevation_sim", '#ffff00', 2, 0)
graphTArray = cast(Iterator[float], tArray/3600)
plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[0], "sma", '#0000ff', 2, 0)], "sma (km)")
plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[1], "h", '#0000ff', 2, 0)], "h")
plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[2], "k", '#0000ff', 2, 0)], "k")
plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[3], "p", '#0000ff', 2, 0)], "p")
plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[4], "q", '#0000ff', 2, 0)], "q")
plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[5], "lon", '#0000ff', 2, 0)], "Lon (rad)")


plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[6], r'\lambda_{sma}', '#0000ff', 2, 0)], r'\lambda_{sma}')
plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[7], r'\lambda_{h}', '#0000ff', 2, 0)], "lambda h")
plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[8], r'\lambda_{k}', '#0000ff', 2, 0)], "lambda k")
plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[9], r'\lambda_{p}', '#0000ff', 2, 0)], "lambda p")
plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[10], r'\lambda_{q}', '#0000ff', 2, 0)], "lambda q")
plot2DLines([prim.XAndYPlottableLineData(graphTArray, solution.y[11], r'\lambda_{L}', '#0000ff', 2, 0)], "lambda Lon")

equiElements = []
for i in range(0, len(tArray)):    
    temp = mee.EquinoctialElementsHalfITrueLongitude(solution.y[0][i], solution.y[1][i], solution.y[2][i],solution.y[3][i],solution.y[4][i],solution.y[5][i],  muVal)
    
    #realEqui = scaleEquinoctialElements(temp, 1.0, 1.0)
    equiElements.append(temp)
finalKepElements = kep = equiElements[-1].ConvertToModifiedEquinoctial().ToKeplerian()
motions = mee.EquinoctialElementsHalfITrueLongitude.CreateEphemeris(equiElements)
satEphemeris = prim.EphemerisArrays()
satEphemeris.InitFromMotions(tArray, motions)
satPath = prim.PathPrimitive(satEphemeris)
satPath.color = "#ff00ff"

jh.showEquation("e", float(finalKepElements.Eccentricity))
jh.showEquation("i", float(finalKepElements.Inclination*180/math.pi))
jh.showEquation(r'\Omega', float(finalKepElements.RightAscensionOfAscendingNode*180/math.pi))
jh.showEquation(r'\omega', float(finalKepElements.ArgumentOfPeriapsis*180/math.pi))
jh.showEquation(r'M', float(finalKepElements.TrueAnomaly*180/math.pi))

for i in range(0, 6):
    if i == 6:
        jh.showEquation(z[i], (solution.y[i][-1]*180/math.pi)%360)
    else:
        jh.showEquation(z[i], solution.y[i][-1])
#%%


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
