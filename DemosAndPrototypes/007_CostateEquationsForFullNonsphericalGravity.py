#%%

import sympy as sy
from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import ModifiedEquinoctialElements, CreateSymbolicElements
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb import SafeSubs
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelper
from pyeq2orb.HighLevelHelpers.EquinoctialElementsHelpers import ModifiedEquinoctialElementsHelpers
import scipyPaperPrinter as jh#type: ignore
import numpy as np
import math as math
from scipy.integrate import solve_ivp
from typing import Dict, Union
from IPython.display import display# import everything
from typing import List, Optional
from pyeq2orb.ForceModels.GravityField import gravityField, makeAccelerationMatrixFromPotential, createSphericalHarmonicGravityAcceleration, makeConstantForSphericalHarmonicCoefficient, makeOverallAccelerationExpression, makePotential

# form equi elements
# build 2-body equations
# buuld perturbation stack
# create spherical harmonic gravity equations
## create inertial elements from equi
## create inertial-to-fixed matrix
## create cos and sin of lat and lon
## create rotation matrix for whatever acceleration is to perturbation matrix
### check if other perturbation matrices might be in a more natural axes
## go



subsDict : Dict[Union[sy.Symbol, sy.Expr], SymbolOrNumber]= {}

t = sy.Symbol('t', real=True)
t0 = sy.Symbol('t_0', real=True)
tf = sy.Symbol('t_f', real=True)
mu = sy.Symbol(r'\mu', real=True, positive=True)

symbolicElements = CreateSymbolicElements(t, mu)
twoBodyOdeMatrix = CreateTwoBodyMotionMatrix(symbolicElements, subsDict)
twoBodyEvaluationHelper = OdeLambdifyHelper(t, symbolicElements.ToArray(), twoBodyOdeMatrix, [mu], subsDict)
twoBodyOdeCallback = twoBodyEvaluationHelper.CreateSimpleCallbackForSolveIvp()

jh.display(symbolicElements.ToArray())
jh.display(twoBodyOdeMatrix)


B = symbolicElements.CreatePerturbationMatrix(subsDict)
jh.display(B)


# \dot(x) = f + B * pert_forces_in_inertial

#%%

# cartesian state values
r, v = symbolicElements.ToCartesianArray()
inertial_elements = sy.Matrix([*r, *v])
inertial_r = sy.Matrix([*r])
jh.display(inertial_r)
r_mag = symbolicElements.Radius
jh.showEquation(symbolicElements.RadiusSymbol, r_mag)

jh.showEquation("R", inertial_r/r_mag)
#NOTE: Normalizing the R vector here seems to be better than doing it fixed
#%%
# earth to fixed matrix

FK = sy.MatrixSymbol("F", 3, 3)
FK_full = sy.Matrix([[FK[0,0], FK[0,1],FK[0,2]],[FK[1,0], FK[1,1], FK[1,2]],[FK[2,0], FK[2,1],FK[2,2]]])
jh.display(FK)
jh.display(FK_full)

rFixedSy = sy.Matrix([[sy.Symbol(r'r_{fx}', real=True)], [sy.Symbol(r'r_{fy}', real=True)], [sy.Symbol(r'r_{fz}', real=True)]])

r_fixed = FK_full*(inertial_r/r_mag) #TODO: Check order and alignment
jh.display(r_fixed)
jh.showEquation("R_f", r_fixed)

# remember, a unit vector already
xyMag = sy.sqrt(r_fixed[0]**2+r_fixed[1]**2)
lat = sy.Symbol(r"\gamma", real=True)
lon = sy.Symbol(r'\lambda', real=True)
cosLon = r_fixed[0]/xyMag # adjecent over hyp
sinLon = r_fixed[1]/xyMag # opposite over hyp
cosLat = xyMag 
sinLat = r_fixed[2]

jh.showEquation(r'cos(\lambda)', cosLon)
jh.showEquation(r'sin(\lambda)', sinLon)

jh.showEquation(r'cos(\delta)', cosLat)
jh.showEquation(r'sin(\delta)', sinLat)


rSy = symbolicElements.RadiusSymbol
rE = sy.Symbol("R_e", real=True, positive=True, communitive=False)
mu = sy.Symbol(r'\mu', real=True, positive=True)
#%%
lat = sy.Symbol(r'\phi', real=True)


        
#%%
c00 = makeConstantForSphericalHarmonicCoefficient("C", 0,0)
c10 = makeConstantForSphericalHarmonicCoefficient("C", 1,0)
c11 = makeConstantForSphericalHarmonicCoefficient("C", 1,1)
c20 = makeConstantForSphericalHarmonicCoefficient("C", 2,0)
c21 = makeConstantForSphericalHarmonicCoefficient("C", 2,1)

s00 = makeConstantForSphericalHarmonicCoefficient("S", 0,0)
s10 = makeConstantForSphericalHarmonicCoefficient("S", 1,0)
s11 = makeConstantForSphericalHarmonicCoefficient("S", 1,1)
s20 = makeConstantForSphericalHarmonicCoefficient("S", 2,0)
s21 = makeConstantForSphericalHarmonicCoefficient("S", 2,1)
s22 = makeConstantForSphericalHarmonicCoefficient("S", 2,2)

rFixedSy = sy.Matrix([[sy.Symbol('r_{fx}', real=True)], [sy.Symbol('r_{fy}', real=True)],[sy.Symbol('r_{fz}', real=True)]])
accelerationDerivative = makeAccelerationMatrixFromPotential(2, 0, mu, rSy, rE, lat, lon).subs({c00:0, c10:0, c11:0, c20:0, c21:0,   s00:0, s10:0, s11:0, s20:0, s21:0, s22:0}).subs(sy.sin(lat), rFixedSy[2]/rSy).subs(lon, 0)

display(accelerationDerivative[0])
display(accelerationDerivative[0].simplify())
display(accelerationDerivative[0].simplify().trigsimp(deep=True))
c2 = makeConstantForSphericalHarmonicCoefficient("C", 2,2)

expected_accel_ai_22 = (3*c2*mu*(rE**2)/(2*rSy**4))*(1-5*rFixedSy[2]**2/(rSy**2))*(rFixedSy[0]/rSy)
jh.showEquation("ai_{c2_v}", expected_accel_ai_22)
display(sy.Eq(0, (expected_accel_ai_22-accelerationDerivative[0].simplify().trigsimp(deep=True)).simplify().simplify()))

#%%

pot = makePotential(2, 2, mu, rSy, rFixedSy, rE, lat, lon)*rSy/mu
display(pot)
display(pot.diff(lat))

x = sy.Symbol('x', real=True)
p21 = sy.assoc_legendre(2, 1, x)
diffp21 = p21.diff(x)
p22 = sy.assoc_legendre(2, 2, x)
display(p21)

display((diffp21 - (sy.sqrt(1-x*x)*p22+x*p21)/(x*x-1)).simplify())


#%%
x = sy.Symbol('x', real=True)
p21_x = sy.assoc_legendre(2, 1, x)

display(p21_x)
display(p21_x.diff(x))
lhs = ((x**2)-1) * p21_x.diff(x)
rhs = sy.sqrt(1-x**2) * sy.assoc_legendre(2, 2, x) + 1*x*p21_x
display(lhs)
display(rhs)
display((lhs-rhs).simplify().replace(sy.Abs(sy.cos(lat)), sy.cos(lat)).trigsimp(deep=True))



#%%
display(potential.simplify())

# note that C = -J, so vallado has a negative here that I do not
fromVallado = 3*builder.makeConstant("C", 2, 2) * (mu/(2*rSy)) * ((rE/rSy)**2) * (sy.sin(lat)**2 - sy.Rational(1,3))
display(fromVallado.simplify())
display((fromVallado-potential).simplify())
display(sy.assoc_legendre(2, 0, sy.sin(lat)))
#%%

rbVec = sy.Matrix([sy.Symbol('x_b', real=True), sy.Symbol('y_b', real=True),sy.Symbol('z_b', real=True)])
rbVecSy = sy.MatrixSymbol("R_b", 1, 3)
rbSy = sy.Symbol(r'\hat{r_b}')

delXb_delRb = sy.Matrix([1.0, 0.0, 0.0]).transpose()
delYb_delRb = sy.Matrix([0.0, 1.0, 0.0]).transpose()
delZb_delRb = sy.Matrix([0.0, 0.0, 1.0]).transpose()

delR_delRb = rbVec.transpose()/rbSy
delLat_delRb = (1/sy.sqrt(rbVec[0]**2 + rbVec[1]**2)) * (delZb_delRb - rbVec[2]*rbVec.transpose()/(rbSy**2))
delLon_delRb = (1/ (rbVec[0]**2 + rbVec[1]**2)) * (rbVec[0]*delYb_delRb - rbVec[1]*delXb_delRb)
display(delR_delRb)
display(delLat_delRb)
display(delLon_delRb)

delU_delR = potential.diff(rSy)
delU_delLat = potential.diff(lat)
delU_delLon = potential.diff(lon)

accel = delU_delR * (delR_delRb.transpose()) + delU_delLat * (delLat_delRb.transpose()) + delU_delLon * (delLon_delRb.transpose())
display(accel[0].trigsimp().simplify())
display(accel[1].simplify())
display(accel[2].simplify())
#%%
ps = legendre_functions_goddard(sy.sin(lat), lat, 4)
l = 0
for arr in ps:
    m = 0
    for val in arr:
        jh.showEquation(f"P_{l}_{m}", val)
        m+=1
    l+=1

p31 = (1/2)*sy.cos(lat)*(15*sy.sin(lat)**2-3)
jh.display((p31-ps[3][1]).expand().simplify())



n_max = 4
m = sy.Symbol("m", integer=True, real=True, positive=True)
n = sy.Symbol("n", integer=True, real=True, positive=True)
Ps = []
Cs = []
Ss = []
for n_i in range(0, n_max):
    Ps.append([])
    Cs.append([])
    Ss.append([])
    for m_i in range(0, n_i+1): 
        Ps[-1].append(sy.Symbol("P^{" + str(m_i) + "}_{" + str(n_i) + "}", real=True))
        Cs[-1].append(sy.Symbol("C^{" + str(m_i) + "}_{" + str(n_i) + "}", real=True))
        Ss[-1].append(sy.Symbol("S^{" + str(m_i) + "}_{" + str(n_i) + "}", real=True))

print(Ps)


term = sy.Symbol(r'sin(bla)', real=True)
P = sy.Function('P')(term, n, m)
C = sy.Function('C')(n,m)
S = sy.Function('S')(n,m)


#U = mu/rSy * sy.Sum(((rE/rSy)**n) *  (P[n, 0]*P[n, 0] + P[n,m]*(S[n, m]*sy.sin(m*lon) + C[n,m]*sy.cos(m*lon)), (m, 1, n)), (n, 0, n_max))
sinMLon = 2*sy.cos(lon)*sy.sin((m-1)*lon)-sy.sin((m-2)*lon)
cosMLon = 2*sy.cos(lon)*sy.cos((m-1)*lon)-sy.cos((m-2)*lon)

betterP = lambda lone, ne, me : legendre_functions_goddard_single(sy.sin(lon), lon, ne, me, pCache)


U = mu/rSy * sy.Sum(((rE/rSy)**n) * C* P + sy.Sum(((rE/rSy)**n) *P*(S*sinMLon + C*cosMLon), (m, 0, n-1)), (n, 2, n_max))
jh.showEquation("U", U)

cVals = [[0.0],
         [0.0, 0.0],
         [0.001, 0.002, 0.003],
         [0.004, 0.005, 0.006, 0.007],
         [0.008, 0.009, 0.010, 0.011, 0.012]]

sVals = [[-0.0],
         [-0.0, -0.0],
         [-0.001, -0.002, -0.003],
         [-0.004, -0.005, -0.006, -0.007],
         [-0.008, -0.009, -0.010, -0.011, -0.012]]



def cFunc(n_i, m_i):
    if not isinstance(m_i, int):
        m_i = 0
    return cVals[n_i][m_i]

def sFunc(n_i, m_i):
    if not isinstance(m_i, int):
        m_i = 0
    return sVals[n_i][m_i]


moduleRedirect = {"P": betterP, "C": cFunc, "S": sFunc}
constants = [lon, rE, rSy, mu]
uFix = U.subs({term: sy.sin(lon)}).doit()
lambidified = sy.lambdify(constants, uFix, modules=moduleRedirect, cse=True, docstring_limit = 1000000)
print(lambidified.__doc__)
pCache = []
potential = lambidified(.5, 6378.137, 7000.000, 3.986004418e5)
display(potential)
