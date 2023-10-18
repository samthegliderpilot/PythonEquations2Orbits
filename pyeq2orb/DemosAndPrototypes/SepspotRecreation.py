#%%
#type: ignore
import __init__ 
import sympy as sy
import os
import sys
import math
from collections import OrderedDict
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)
sy.init_printing()
from typing import Union, List, Optional, Sequence
from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
import pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule as mee
from IPython.display import display
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
import scipyPaperPrinter as jh #type: ignore
#jh.printMarkdown("# SEPSPOT Recreation")
#jh.printMarkdown("In working my way up through low-thrust modeling for satellite maneuvers, it is inevitable to run into Dr. Edelbaum's work.  Newer work such as Jean Albert Kechichian's practically requires understanding SEPSPOT as a prerequesit.  This writeup will go through the basics of SEPSPOT's algorithsm as described in the references below.")

#jh.printMarkdown("In other work in this python library, I have already created many helper types such as Equinoctial elements, their equations of motion, rotation matrices, and more. To start, we will define out set of equinoctial elements.  Unlike the orignial paper, I will be using the modified elements.  This replaces the semi-major axis with the parameter and reorders/renames some of the other elements.")
t=sy.Symbol('t', real=True)
mu = sy.Symbol(r'\mu', real=True, positive=True)
muVal = 3.986004418e5  
kepElements = KepModule.CreateSymbolicElements(t, mu)

simpleBoringEquiElements = mee.EquinoctialElementsHalfI.CreateSymbolicElements(t, mu)
simpleBoringEquiElements.SemiMajorAxis = sy.Function('a', real=True, positive=True)(t)
a = simpleBoringEquiElements.SemiMajorAxis
h = simpleBoringEquiElements.EccentricitySinTermH
k = simpleBoringEquiElements.EccentricityCosTermK
p = simpleBoringEquiElements.InclinationSinTermP
q = simpleBoringEquiElements.InclinationCosTermQ
n = sy.sqrt(mu/(a**3))

beta = 1/(1+sy.sqrt(1-h**2-k**2))
equiInTermsOfKep = mee.ConvertKeplerianToEquinoctial(kepElements)
# kepInTermsOfEqui = simpleEquiElements.ToKeplerian()
# jh.showEquation("p", equiInTermsOfKep.SemiParameter)
# jh.showEquation("f", equiInTermsOfKep.EccentricityCosTermF)
# jh.showEquation("g", equiInTermsOfKep.EccentricitySinTermG)
# jh.showEquation("h", equiInTermsOfKep.InclinationCosTermH)
# jh.showEquation("k", equiInTermsOfKep.InclinationSinTermK)
# jh.showEquation("L", equiInTermsOfKep.TrueLongitude)
eccentricAnomaly = sy.Symbol('E')
eccentricLongitude = sy.Function('F')(t)
simpleBoringEquiElements.F = eccentricLongitude
equiInTermsOfKep.F = eccentricAnomaly + kepElements.ArgumentOfPeriapsis + kepElements.RightAscensionOfAscendingNode
#jh.printMarkdown("We want our orbital elements to use the eccentric longitude which is:")
#jh.showEquation(eccentricLongitude, equiInTermsOfKep.F) #TODO: Look into how to better include this in the normal equi elements

#jh.printMarkdown("The rotation matrix of the axes being used for this analysis to inertial is:")
#jh.showEquation("R", simpleBoringEquiElements.CreateFgwToInertialAxes())
r = simpleBoringEquiElements.CreateFgwToInertialAxes()
#jh.printMarkdown("And with keplerian elements:")
#jh.showEquation("R", equiInTermsOfKep.CreateFgwToInertialAxes())

#jh.printMarkdown("And we need the position and velocity in the FGW axes.  Using the normal equinoctial elements (in order to better compare to the original paper):")
x1Sy = sy.Symbol('X_1')
x2Sy = sy.Symbol('X_2')
x1DotSy = sy.Symbol(r'\dot{X_1}')
x2DotSy = sy.Symbol(r'\dot{X_2}')

xSuperSimple = Cartesian(x1Sy, x2Sy, 0)
xDotSuperSimple = Cartesian(x1DotSy, x2DotSy, 0)
fullSubsDictionary = {}
[x1SimpleEqui, x2SimpleEqui] = simpleBoringEquiElements.RadiusInFgw(eccentricLongitude, fullSubsDictionary)
[x1DotSimpleEqui, x2DotSimpleEqui] = simpleBoringEquiElements.VelocityInFgw(eccentricLongitude, fullSubsDictionary)
# jh.showEquation(x1Sy, x1SimpleEqui)
# jh.showEquation(x2Sy, x2SimpleEqui)
# jh.showEquation(x1DotSy, x1SimpleEqui)
# jh.showEquation(x2DotSy, x2SimpleEqui)
normalEquiElementsInTermsOfKep = mee.EquinoctialElementsHalfI.FromModifiedEquinoctialElements(equiInTermsOfKep)
[x1Complete, x2Complete] = normalEquiElementsInTermsOfKep.RadiusInFgw(equiInTermsOfKep.F)
[x1DotComplete, x2DotComplete] = normalEquiElementsInTermsOfKep.VelocityInFgw(equiInTermsOfKep.F)
# jh.showEquation("X_1", x1Complete.trigsimp(deep=True))
# jh.showEquation("X_2", x2Complete.trigsimp(deep=True))
# jh.showEquation("\dot{X_1}", x1DotComplete.trigsimp(deep=True))
# jh.showEquation("\dot{X_2}", x2DotComplete.trigsimp(deep=True))

#jh.showEquation(x1Sy, x1SimpleEqui, False)
#showEquation(x1Sy, x1SimpleEqui, [k])

qOfT = sy.Function('q')(t)
dqdt = 5*qOfT.diff(t)
ddqdtt = 5*dqdt.diff(t)
showEquation(r'\frac{dq}{dt}', dqdt, [], True, t)


#x,y,z = sy.symbols('x y z', real=True)
#vx,vy,vz = sy.symbols('v_x v_y v_z', real=True)
#cart = MotionCartesian(Cartesian(x,y,z), Cartesian(vx,vy,vz))

#%%
def poisonBracket(exp, f, g, states) :
    sum = 0
    for a in states:
        for b in states:
            sum = sum - sy.diff(f, a) * sy.diff(g, b) + sy.diff(f, b)*sy.diff(g, a)
    return sum


#%%


fullSubsDictionary[x1Sy] = x1Complete
fullSubsDictionary[x2Sy] = x2Complete
fullSubsDictionary[x1DotSy] = x1DotComplete
fullSubsDictionary[x2DotSy] = x2DotComplete

meanAnomaly = sy.Function("M")(t)
kepElements.M = meanAnomaly
keplerianEquationLhs = kepElements.M + kepElements.ArgumentOfPeriapsis + kepElements.RightAscensionOfAscendingNode
keplerianEquationHhs = equiInTermsOfKep.F - equiInTermsOfKep.EccentricityCosTermF*sy.sin(eccentricLongitude) + equiInTermsOfKep.EccentricitySinTermG*sy.cos(eccentricLongitude)
kepEquation = sy.Eq(keplerianEquationLhs, keplerianEquationHhs)
jh.printMarkdown("And finally, we have Kepler's equation")
jh.showEquation(kepEquation)


#%%
jh.printMarkdown("### The Optimal Control Problem")
jh.printMarkdown("We will limit ourselves (for now) to the 5-state orbit averaged problem.  We will also for now stick to the 2-body problem with no oblateness of the Earth.")
jh.printMarkdown("The paper defines the Hamiltonian as")
jh.printMarkdown(r'$$H=\underline{\lambda}^{T}\underline{\dot{x}}$$')
jh.printMarkdown("Which is the standard Hamiltonian I've seen in other sources assuming no path cost.")

def makeMatrixOfSymbols(baseString : str, rows, cols, t=None) :
    endString = ''
    if baseString.endswith('}') :
        baseString = baseString[:-1]
        endString = '}'
    mat = sy.Matrix.zeros(rows, cols)
    for r in range(0, rows) :
        for c in range(0, cols):
            

            if t== None :
                mat[r,c] = sy.Symbol(baseString + "_{" + str(r) + "," + str(c)+"}" + endString)
            else:
                mat[r,c] = sy.Function(baseString + "_{" + str(r) + "," + str(c)+"}"+ endString)(t)
    return mat


n = 5
jh.printMarkdown("Staring with our x:")
x = sy.Matrix([[simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermK, simpleBoringEquiElements.InclinationSinTermP, simpleBoringEquiElements.InclinationCosTermQ]]).transpose()
xSy = sy.MatrixSymbol('x', n, 1)
jh.showEquation(xSy, x)
jh.printMarkdown(r'We write our $\underline{\dot{x}}$ with the assumed optimal control vector $\underline{\hat{u}}$ as:')
g1Sy = makeMatrixOfSymbols(r'g_{1}', n, 1, t)
aSy = sy.Function('a', commutative=True)(x, t)
uSy = sy.Matrix([["u1", "u2", "u3"]]).transpose()
g2Sy = makeMatrixOfSymbols('G_{2}', 5, 3)
display(g2Sy)
xDotSy = SymbolicProblem.CreateCoVector(x, r'\dot{x}', t)
xDot = g1Sy+ aSy*g2Sy*uSy
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
uOpt = lambdas.transpose()*g2Sy / ((lambdas.transpose()*g2Sy).norm())
display(uOpt)

jh.printMarkdown("Putting this back into our Hamiltonian, we get")
hStar = (lambdas.transpose() * g1Sy)[0,0] + aSy*(uOpt.norm())
jh.showEquation("H^{*}", hStar)
jh.printMarkdown("Although not as cleanly defined as in the paper, we will soon be substituting expressions into this to create our equations of motion and boundary conditions.")

#%%
x1 = x1Complete
y1 = x2Complete
xDot = x1DotComplete
yDot = x2DotComplete
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
m23 = k*(q*y1-p*x1)/(n*(a**2)*sy.sqrt(1-h**h-k**h))
m31 = -1*(sy.sqrt(1-h**2-k**2)/(n*a**2))*(dX1dh-(xDot/n)*(sy.cos(F)-k*beta))
m32 = -1*(sy.sqrt(1-h**2-k**2)/(n*a**2))*(dY1dh-(yDot/n)*(sy.cos(F)-k*beta))
m33 = -1*h*(q*y1-p*x1)/(n*(a**2)*sy.sqrt(1-h**h-k**h))
m41 = 0
m42 = 0
m43 = (1+p**2+q**2)*y1/(2*n*a**2*sy.sqrt(1-h**2-k**2))
m51 = 0
m52 = 0
m53 = (1+p**2+q**2)*x1/(2*n*a**2*sy.sqrt(1-h**2-k**2))

#%%
M = sy.Matrix([[m11, m12, m13], [m21, m22, m23],[m31, m32, m33],[m41, m42, m43],[m51, m52, m53]])
display(M)
display(M.shape)
#%%
#accel = sy.Symbol('a', real=True, nonnegative=True)


#lambdas = sy.Matrix([[sy.Symbol(r'\lambda_a')],[sy.Symbol(r'\lambda_h')],[sy.Symbol(r'\lambda_k')],[sy.Symbol(r'\lambda_p')],[sy.Symbol(r'\lambda_q')]])
acceleration= sy.Symbol('a')#sy.Matrix([[sy.Symbol('a_x'),sy.Symbol('a_y'),sy.Symbol('a_z')]])
MtimesLambdas = M.transpose()*lambdas
mTimesLambdasMagnitude = MtimesLambdas.norm()
MNormalized =(MtimesLambdas)/(mTimesLambdasMagnitude)
zDot = acceleration * (M*MNormalized)
display(zDot.shape)

#%%

s = (1-k*sy.cos(F)-h*sy.sin(F))/(2*sy.pi)
delSDelZ = sy.Matrix([[0, -sy.sin(F), -sy.cos(F), 0, 0]]) *2*sy.pi
zDotOverAnOrbit = -acceleration*sy.Integral((M*MNormalized)*s  , (F, -sy.pi, sy.pi))
delZDotDelZ = acceleration*M.applyfunc(lambda s: sy.diff(s, F))*MNormalized
display(delZDotDelZ.shape)


#%%

jh.printMarkdown("In implimenting equation 40, note that the delZDot del z is made on a per-row basis.")
z = [a, h, k, p, q]
part2 = lambdas.transpose()*zDot*delSDelZ
lmdDotArray = []
for i in range(0, 5) :
    delZDotDelZ = acceleration*M.applyfunc(lambda s: sy.diff(s, z[i]))*MNormalized 
    part1 = ((lambdas.transpose()*delZDotDelZ) * s)[0]
    fullIntegralOfThisEom = -sy.Integral(part1 + part2[0,i], (F, -sy.pi, sy.pi))
    lmdDotArray.append(fullIntegralOfThisEom)
    print("finished " + str(i))


print(lmdDotArray)
#%%
# now we try to integrate
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions
accelVal = 9.798e-4  #units are km, sec

eoms = []
for i in range(0, len(z)):
    eoms.append(sy.Eq(z[i].diff(t), zDot[i]))
for i in range(0, len(lambdas)):
    eoms.append(sy.Eq(lambdas[i].diff(t), lmdDotArray[i]))
lmdHelper = OdeLambdifyHelperWithBoundaryConditions(t, sy.Symbol('t_0', real=True), sy.Symbol('t_f', real=True), eoms, [], [], {mu: muVal, acceleration:accelVal})

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


lmdGuess = [10,0.1,0.1,200,200]

fullInitialState = [a0V, h0V, k0V, p0V, q0V]
fullInitialState.extend(lmdGuess)
#%%
integratorCallback = lmdHelper.CreateSimpleCallbackForSolveIvp()


#%%
# jh.printMarkdown("## Averaging of the Hamiltonian")

# jh.printMarkdown("In order to get the averaged Hamiltonian, we need to make the following transformation:")
# def createAveragedHamiltonian(h, lowerBound, upperBound,averageringVariabe, dtdAveragingVariable) :
#     T = upperBound - lowerBound
#     oneOverT = 1/T
#     hamltAveraged = oneOverT * sy.integrate(h*dtdAveragingVariable, (averageringVariabe, lowerBound, upperBound))
#     return hamltAveraged

# kepEquationEquiElementsRhs = eccentricLongitude - simpleBoringEquiElements.EccentricitySinTermH*sy.sin(eccentricLongitude) + simpleBoringEquiElements.EccentricityCosTermK*sy.cos(eccentricLongitude)
# jh.printMarkdown("The derivative of the left hand side of Keplers equation is the mean motion, where T is the period")
# period = sy.Symbol('T')
# dmdt = 2*sy.pi/period
# display(dmdt)
# jh.printMarkdown("And the right hand side will give us an expression for $\frac{dt}{dF}$")
# dKepDtRhs = sy.diff(kepEquationEquiElementsRhs, t)
# equToGetDFDt = sy.Eq(dmdt, dKepDtRhs)
# dtdF=1/sy.solve(equToGetDFDt, sy.diff(eccentricLongitude, t))[0]
# jh.showEquation(r'\frac{dt}{dF}', dtdF)
# hAveraged = createAveragedHamiltonian(hStar, -1*sy.pi, sy.pi, eccentricLongitude, dtdF)
# display(hAveraged)

# jh.printMarkdown("With this, we need to start filling in our G1 and G2 expressions.  After that, it is applying the Optimal Control Euler-Lagrange expressions.")

#display(simpleBoringEquiElements.CreatePerturbationMatrix())


#%%

# fullRDot = simpleBoringEquiElements.CreateFgwToInertialAxes()*sy.Matrix([[x1DotSimpleEqui],[x2DotSimpleEqui],[0]]) 
# #display(fullRDot)

# #display(fullRDot.diff(simpleEquiElements.InclinationCosTermH))

# m11 = 2*x1DotSimpleEqui/(simpleBoringEquiElements.SemiMajorAxis * sy.Symbol(r'\mu')/(simpleBoringEquiElements.SemiMajorAxis**3))
# maybeM11 = x1DotSimpleEqui.diff(simpleBoringEquiElements.SemiMajorAxis)
# #display(m11)
# #display(maybeM11)
# #display((m11-maybeM11))

# a = sy.Symbol('a')
# mu=sy.Symbol(r'\mu')

# simpM11 = 2*a*a/mu

# x1DotSuperSimple = sy.sqrt(mu/a)*a
# display(sy.diff(x1DotSuperSimple,a)-1/simpM11)

# x1S = sy.Symbol('X_1')
# x2S = sy.Symbol('X_2')
# x1DotS = sy.Symbol('\dot{X_1}')
# x2DotS = sy.Symbol('\dot{X_2}')

# xMag = sy.sqrt(x1S**2+x2S**2)
# xDotMag = sy.sqrt(x1DotS**2+x2DotS**2)
# a = (1/xMag-xDotMag**2/mu)**(-1)
# display(a.diff(x1DotS))

#%%
# innerX1Dot = sy.Symbol(r'\dot{X_1}')
# a = simpleBoringEquiElements.SemiMajorAxis
# #simpleX1Dot = sy.sqrt(sy.Symbol(r'\mu')/(a**3))*a**2*innerX1Dot
# simpleX1Dot = sy.Symbol('n')*a**2*innerX1Dot
# display(simpleX1Dot)
# display(sy.powsimp(simpleX1Dot.diff(simpleBoringEquiElements.SemiMajorAxis)))

#%%
# if '__file__' in globals() or '__file__' in locals():
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     thisFilePath = os.path.join(dir_path, "ModifiedEquinoctialElementsExplanation.py")
#     jh.ReportGeneratorFromPythonFileWithCells.WriteIpynbToDesiredFormatWithPandoc(thisFilePath, keepDirectoryClean=True)
#     jh.printMarkdown("done")

