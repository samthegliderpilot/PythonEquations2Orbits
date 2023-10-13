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

from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
import pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule as mee
from IPython.display import display
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
import scipyPaperPrinter as jh #type: ignore



def poisonBracket(f, g, ps, qs) :
    sum = 0
    for i in range(0, len(ps)):
        pi = ps[i]
        qi = qs[i]
        sum = sum + sy.diff(f, qi) * sy.diff(g, pi) - sy.diff(f, pi)*sy.diff(g, qi)
    return sum

def poisonBracketMatrix(fs, gs, ps, qs) : #This might be lagrange brackets
    m = sy.Matrix(len(fs), len(gs)) # should be square
    for r in range(0, len(fs)):
        for c in range(0, len(gs)):
            m[r,c] = poisonBracket(fs[r], gs[r], ps, qs)
            #m[c,r] = m[r,c]
    return m

q = sy.Symbol('q', real=True)
p = sy.Symbol('p', real=True)
f1 = p-q
f2 = sy.sin(q)

#display(poisonBracket(f1, f2, [p],[q]))


jh.printMarkdown("# SEPSPOT Recreation")
jh.printMarkdown("In working my way up through low-thrust modeling for satellite maneuvers, it is inevitable to run into Dr. Edelbaum's work.  Newer work such as Jean Albert Kechichian's practically requires understanding SEPSPOT as a prerequesit.  This writeup will go through the basics of SEPSPOT's algorithsm as described in the references below.")

jh.printMarkdown("In other work in this python library, I have already created many helper types such as Equinoctial elements, their equations of motion, rotation matrices, and more. To start, we will define out set of equinoctial elements.  Unlike the orignial paper, I will be using the modified elements.  This replaces the semi-major axis with the parameter and reorders/renames some of the other elements.")
t=sy.Symbol('t', real=True)
mu = sy.Symbol(r'\mu', real=True, positive=True)
muVal = 3.986004418e5  
#kepElements = KepModule.CreateSymbolicElements(t, mu)

equiElements = mee.EquinoctialElementsHalfI.CreateSymbolicElements(t, mu)
equiElements.SemiMajorAxis = sy.Function('a', real=True, positive=True)(t)
a = equiElements.SemiMajorAxis
h = equiElements.EccentricitySinTermH
k = equiElements.EccentricityCosTermK
p = equiElements.InclinationSinTermP
q = equiElements.InclinationCosTermQ
# n = sy.sqrt(mu/(a**3))

# beta = 1/(1+sy.sqrt(1-h**2-k**2))
#equiInTermsOfKep = mee.ConvertKeplerianToEquinoctial(kepElements)
# kepInTermsOfEqui = simpleEquiElements.ToKeplerian()
# jh.showEquation("p", equiInTermsOfKep.SemiParameter)
# jh.showEquation("f", equiInTermsOfKep.EccentricityCosTermF)
# jh.showEquation("g", equiInTermsOfKep.EccentricitySinTermG)
# jh.showEquation("h", equiInTermsOfKep.InclinationCosTermH)
# jh.showEquation("k", equiInTermsOfKep.InclinationSinTermK)
# jh.showEquation("L", equiInTermsOfKep.TrueLongitude)
eccentricAnomaly = sy.Symbol('E')
eccentricLongitude = sy.Function('F')(t)
equiElements.F = eccentricLongitude
F = equiElements.F
#equiInTermsOfKep.F = eccentricLongitude#eccentricAnomaly + kepElements.ArgumentOfPeriapsis + kepElements.RightAscensionOfAscendingNode
# jh.printMarkdown("We want our orbital elements to use the eccentric longitude which is:")
# jh.printMarkdown("The rotation matrix of the axes being used for this analysis to inertial is:")
# jh.showEquation("R", equiElements.CreateFgwToInertialAxes())
# r = equiElements.CreateFgwToInertialAxes()
# jh.printMarkdown("And with keplerian elements:")
#jh.showEquation("R", equiInTermsOfKep.CreateFgwToInertialAxes())

jh.printMarkdown("And we need the position and velocity in the FGW axes.  Using the normal equinoctial elements (in order to better compare to the original paper):")
# x1Sy = sy.Symbol('X_1')
# x2Sy = sy.Symbol('X_2')
# x1DotSy = sy.Symbol(r'\dot{X_1}')
# x2DotSy = sy.Symbol(r'\dot{X_2}')

# xSuperSimple = Cartesian(x1Sy, x2Sy, 0)
# xDotSuperSimple = Cartesian(x1DotSy, x2DotSy, 0)
fullSubsDictionary = {}
# [x1SimpleEqui, x2SimpleEqui] = equiElements.RadiusInFgw(eccentricLongitude, fullSubsDictionary)
# [x1DotSimpleEqui, x2DotSimpleEqui] = equiElements.VelocityInFgw(eccentricLongitude, fullSubsDictionary)
# jh.showEquation(x1Sy, x1SimpleEqui)
# jh.showEquation(x2Sy, x2SimpleEqui)
# jh.showEquation(x1DotSy, x1DotSimpleEqui)
# jh.showEquation(x2DotSy, x2DotSimpleEqui)






beta = sy.Function(r'\beta', real=True)(h, k)
betaExp = 1/(a+sy.sqrt(1-h**2-k**2))
fullSubsDictionary[beta]= betaExp

n = sy.Function(r'\mu', real=True, positive=True)(a)
nExp = sy.sqrt(mu/(a**3))
fullSubsDictionary[n]=nExp

r = sy.Function(r'r', real=True, positive=True)(a, k, h, F)
rExp = a*(1-k*sy.cos(F)-h*sy.sin(F))
fullSubsDictionary[r]=rExp

# not sure why the paper has _1 here...
x1 = sy.Function('X_1', real=True)(a, h, k, F, t)
y1 = sy.Function('Y_1', real=True)(a, h, k, F, t)
x1Dot = sy.Function('\dot{X}_1', real=True)(a, h, k, F, t)
y1Dot = sy.Function('\dot{Y}_1', real=True)(a, h, k, F, t)

x1Exp = a*((1-beta*h**2)*sy.cos(F)+h*k*beta*sy.sin(F)-k)
y1Exp = a*((1-beta*k**2)*sy.sin(F)+h*k*beta*sy.cos(F)-h)
#TODO: Can derive?
x1DotExp = n*a**2/r * (h*k*beta*sy.cos(F)-(1-beta*(h**2)*sy.sin(F)))
y1DotExp = n*a**2/r * (-h*k*beta*sy.sin(F)+(1-beta*(k**2)*sy.cos(F)))
fullSubsDictionary[x1]=x1Exp
fullSubsDictionary[y1]=y1Exp
fullSubsDictionary[x1Dot]=x1DotExp
fullSubsDictionary[y1Dot]=y1DotExp

a = sy.Function('a', real=True)(x1, y1, x1Dot, y1Dot)
h = sy.Function('h', real=True)(x1, y1, x1Dot, y1Dot)
k = sy.Function('k', real=True)(x1, y1, x1Dot, y1Dot)
p = sy.Function('p', real=True)(x1, y1, x1Dot, y1Dot)
q = sy.Function('q', real=True)(x1, y1, x1Dot, y1Dot)
F = sy.Function('F', real=True)(x1, y1, x1Dot, y1Dot)
f = sy.Matrix([[sy.Function('f_0', real=True)(p,q)], [sy.Function('f_1', real=True)(p,q)], [sy.Function('f_2', real=True)(p,q)]])
g = sy.Matrix([[sy.Function('g_0', real=True)(p,q)], [sy.Function('g_1', real=True)(p,q)], [sy.Function('g_2', real=True)(p,q)]])
w = sy.Matrix([[sy.Function('w_0', real=True)(p,q)], [sy.Function('w_1', real=True)(p,q)], [sy.Function('w_2', real=True)(p,q)]])
temp = 1/(1+p**2+q**2)
fExp = temp*sy.Matrix([[1-p**2+q**2], [2*p*q], [-2*p]])
gExp = temp*sy.Matrix([[2*p*q], [1+p**2-q**2], [-2*q]])
wExp = temp*sy.Matrix([[2*p], [-2*q], [1-p**2+q**2]])
fullSubsDictionary[f[0]]= fExp[0]
fullSubsDictionary[f[1]]= fExp[1]
fullSubsDictionary[f[2]]= fExp[2]
fullSubsDictionary[g[0]]= gExp[0]
fullSubsDictionary[g[1]]= gExp[1]
fullSubsDictionary[g[2]]= gExp[2]
fullSubsDictionary[w[0]]= wExp[0]
fullSubsDictionary[w[1]]= wExp[1]
fullSubsDictionary[w[2]]= wExp[2]

rotMatrix = sy.Matrix([[*f], [*g],[*w]]).transpose()
jh.showEquationNoFunctionsOf("R", rotMatrix)

rMag = sy.sqrt(x1**2+y1**2)
vMag = sy.sqrt(x1Dot**2+y1Dot**2)
positionVector = rotMatrix * sy.Matrix([[x1],[y1],[0]])  #TODO: Right order?
velocityVector = rotMatrix * sy.Matrix([[x1Dot],[y1Dot],[0]]) 
kep = KeplerianElements.FromMotionCartesian(MotionCartesian(positionVector, velocityVector), mu)

aExp = 1/((2/rMag)-vMag**2/mu)
hVec = positionVector.cross(velocityVector)
hVecNorm = hVec * (1/hVec.norm())
pExp = hVecNorm[0]/(1-hVecNorm[2]).simplify().expand().simplify()
qExp = -hVecNorm[1]/(1-hVecNorm[2]).simplify().expand().simplify()

eVec = -positionVector/rMag+velocityVector.cross(hVec)/mu
hExp = eVec.dot(g).simplify().expand().simplify()
kExp = eVec.dot(f).simplify().expand().simplify()


finalSubs = {a:aExp, h:hExp, k:kExp, p:pExp, q:qExp}

preM = sy.Matrix([[a, h, k, p, q]]).transpose()
MRow1 = preM.diff(x1Dot).applyfunc(lambda s : s.simplify())
jh.showEquationNoFunctionsOf("M_1", MRow1)

MRow1Subs = MRow1[2].subs(finalSubs)
jh.showEquationNoFunctionsOf("M_{13}", MRow1Subs)

MRow1SubsLevel2 = MRow1Subs.subs(fullSubsDictionary)
jh.showEquationNoFunctionsOf("M_{13}", MRow1SubsLevel2)

#%%
#normalEquiElementsInTermsOfKep = mee.EquinoctialElementsHalfI.FromModifiedEquinoctialElements(equiInTermsOfKep)
#[x1Complete, x2Complete] = normalEquiElementsInTermsOfKep.RadiusInFgw(equiInTermsOfKep.F)
#[x1DotComplete, x2DotComplete] = normalEquiElementsInTermsOfKep.VelocityInFgw(equiInTermsOfKep.F)
#jh.showEquation("X_1", x1Complete.trigsimp(deep=True))
#jh.showEquation("X_2", x2Complete.trigsimp(deep=True))
#jh.showEquation("\dot{X_1}", x1DotComplete.trigsimp(deep=True))
#jh.showEquation("\dot{X_2}", x2DotComplete.trigsimp(deep=True))


#x,y,z = sy.symbols('x y z', real=True)
#vx,vy,vz = sy.symbols('v_x v_y v_z', real=True)
#cart = MotionCartesian(Cartesian(x,y,z), Cartesian(vx,vy,vz))

#%%


#%%


fullSubsDictionary[x1Sy] = x1SimpleEqui
fullSubsDictionary[x2Sy] = x2SimpleEqui
fullSubsDictionary[x1DotSy] = x1DotSimpleEqui
fullSubsDictionary[x2DotSy] = x2DotSimpleEqui

#meanAnomaly = sy.Function("M")(t)
#kepElements.M = meanAnomaly
#keplerianEquationLhs = kepElements.M + kepElements.ArgumentOfPeriapsis + kepElements.RightAscensionOfAscendingNode
#keplerianEquationHhs = equiInTermsOfKep.F - equiInTermsOfKep.EccentricityCosTermF*sy.sin(eccentricLongitude) + equiInTermsOfKep.EccentricitySinTermG*sy.cos(eccentricLongitude)
#kepEquation = sy.Eq(keplerianEquationLhs, keplerianEquationHhs)
#jh.printMarkdown("And finally, we have Kepler's equation")
#jh.showEquation(kepEquation)


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
x = sy.Matrix([[equiElements.SemiMajorAxis, equiElements.EccentricitySinTermH, equiElements.EccentricityCosTermK, equiElements.InclinationSinTermP, equiElements.InclinationCosTermQ]]).transpose()
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
x1 = x1SimpleEqui
y1 = x2SimpleEqui
xDot = x1DotSimpleEqui
yDot = x2DotSimpleEqui
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
fullSubsDictionary[mu] = muVal
fullSubsDictionary[acceleration] = accelVal
eoms = []
for i in range(0, len(z)):
    eoms.append(sy.Eq(z[i].diff(t), zDot[i]))
    display(eoms[-1])
for i in range(0, len(lambdas)):
    eoms.append(sy.Eq(lambdas[i].diff(t), lmdDotArray[i]))
    display(eoms[-1])
#%%    
lmdHelper = OdeLambdifyHelperWithBoundaryConditions(t, sy.Symbol('t_0', real=True), sy.Symbol('t_f', real=True), eoms, [], [F], fullSubsDictionary)

z0 = SymbolicProblem.SafeSubs(z, {t: lmdHelper.t0})
zF = SymbolicProblem.SafeSubs(z, {t: lmdHelper.tf})

a0V = 10509.0
h0V = 0.325
k0V = 0
p0V = 28.5*math.pi/180.0
q0V = 0
f0V = 0
t0V = 2444239.0 * 86400

afV = 42241.19
hfV = 0
kfV = 0
pfV = 0
qfV = 0
ffV = 0
tfV = 0

lmdHelper.BoundaryConditionExpressions.append(zF[0]-afV)
lmdHelper.BoundaryConditionExpressions.append(zF[1])
lmdHelper.BoundaryConditionExpressions.append(zF[2])
lmdHelper.BoundaryConditionExpressions.append(zF[3])
lmdHelper.BoundaryConditionExpressions.append(zF[4])


lmdGuess = [10,0.1,0.1,200,200]

fullInitialState = [a0V, h0V, k0V, p0V, q0V]
fullInitialState.extend(lmdGuess)


integratorCallback = lmdHelper.CreateSimpleCallbackForSolveIvp()

#%%

print(integratorCallback(0, fullInitialState, (0)))

#%%
# jh.printMarkdown("## Averaging of the Hamiltonian")

# jh.printMarkdown("In order to get the averaged Hamiltonian, we need to make the following transformation:")
# def createAveragedHamiltonian(h, lowerBound, upperBound,averageringVariabe, dtdAveragingVariable) :
#     T = upperBound - lowerBound
#     oneOverT = 1/T
#     hamltAveraged = oneOverT * sy.integrate(h*dtdAveragingVariable, (averageringVariabe, lowerBound, upperBound))
#     return hamltAveraged

# kepEquationEquiElementsRhs = eccentricLongitude - equiElements.EccentricitySinTermH*sy.sin(eccentricLongitude) + equiElements.EccentricityCosTermK*sy.cos(eccentricLongitude)
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

#display(equiElements.CreatePerturbationMatrix())


#%%

# fullRDot = equiElements.CreateFgwToInertialAxes()*sy.Matrix([[x1DotSimpleEqui],[x2DotSimpleEqui],[0]]) 
# #display(fullRDot)

# #display(fullRDot.diff(simpleEquiElements.InclinationCosTermH))

# m11 = 2*x1DotSimpleEqui/(equiElements.SemiMajorAxis * sy.Symbol(r'\mu')/(equiElements.SemiMajorAxis**3))
# maybeM11 = x1DotSimpleEqui.diff(equiElements.SemiMajorAxis)
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
# a = equiElements.SemiMajorAxis
# #simpleX1Dot = sy.sqrt(sy.Symbol(r'\mu')/(a**3))*a**2*innerX1Dot
# simpleX1Dot = sy.Symbol('n')*a**2*innerX1Dot
# display(simpleX1Dot)
# display(sy.powsimp(simpleX1Dot.diff(equiElements.SemiMajorAxis)))

#%%
# if '__file__' in globals() or '__file__' in locals():
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     thisFilePath = os.path.join(dir_path, "ModifiedEquinoctialElementsExplanation.py")
#     jh.ReportGeneratorFromPythonFileWithCells.WriteIpynbToDesiredFormatWithPandoc(thisFilePath, keepDirectoryClean=True)
#     jh.printMarkdown("done")

