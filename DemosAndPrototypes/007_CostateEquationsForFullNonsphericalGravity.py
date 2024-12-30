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
# import everything
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

FK = sy.MatrixSymbol("FK", 3, 3)
FK_full = sy.Matrix([[FK[0,0], FK[0,1],FK[0,2]],[FK[1,0], FK[1,1], FK[1,2]],[FK[2,0], FK[2,1],FK[2,2]]])
jh.display(FK)
jh.display(FK_full)

r_fixed = FK_full*(inertial_r/r_mag) #TODO: Check order and alignment
jh.display(r_fixed)
jh.showEquation("R_f", r_fixed)

# remember, a unit vector already
xyMag = sy.sqrt(r_fixed[0]**2+r_fixed[1]**2)
cosLon = r_fixed[0]/xyMag # adjecent over hyp
sinLon = r_fixed[1]/xyMag # opposite over hyp
cosLat = xyMag 
sinLat = r_fixed[2]

jh.showEquation(r'cos(\lambda)', cosLon)
jh.showEquation(r'sin(\lambda)', sinLon)

jh.showEquation(r'cos(\delta)', cosLat)
jh.showEquation(r'sin(\delta)', sinLat)



#%%
lat = sy.Symbol(r'\phi', real=True)
def legendre_functions_vallado(expr, gma, lBound):
    P = []
    p0_0 = 1.0
    p1_0 = expr
    p1_1 = expr.diff(gma)

    P.append([p0_0])
    P.append([p1_0, p1_1])

    for l in range(2, lBound):
        pArray = []
        pl_0 = (2*l-1)*gma*P[l-1][0]-(l-1)*P[l-2][0]
        pArray.append(pl_0)
        for m in range(1, l):
            pl_m = P[l-1][m]+(2*l-1)*p1_1*P[l-1][m-1]
            pArray.append(pl_m)
        pl_l = (2*l-1)*p1_1*P[l-1][l-1]
        pArray.append(pl_l)

        p0_0 = pl_0
        # p1_0 = pArray[1]
        # p1_1 = pArray[-1]

        P.append(pArray)
    return P


def legendre_functions2(expr, gma, lBound):
    P = []
    # p0_0 = 1.0
    # p1_0 = expr
    # p1_1 = expr.diff(gma)

    # P.append([p0_0])
    # P.append([p1_0, p1_1])

    for l in range(0, lBound+1):
        pArray = []
        for m in range(0, l+1):
            pl_m = (((-1)**m)/(sy.factorial(l)* 2**l)) *(1-expr**2)**(m/2) * ((expr**2-1)**l).diff(gma, m)#  (1-expr**2)**(m/2)*P[l][0].diff(gma, m)
            pArray.append(pl_m.simplify().trigsimp(deep=True))
        P.append(pArray)
    return P


def legendre_functions_goddard(expr, gma, lBound):
    P = []
    p0_0 = 1.0
    p1_0 = expr
    p1_1 = expr.diff(gma)

    P.append([p0_0, 0])
    P.append([p1_0, p1_1, 0])

    for l in range(2, lBound):
        pArray = []
        pl_0 = ((2*l-1)*expr*P[l-1][0]-(l-1)*P[l-2][0])/l
        pArray.append(pl_0)
        for m in range(1, l):
            pl_m = P[l-2][m]+(2*l-1)*p1_1*P[l-1][m-1]
            pArray.append(pl_m)
        pl_l = (2*l-1)*p1_1*P[l-1][l-1]
        pArray.append(pl_l)

        #p0_0 = pl_0
        # p1_0 = pArray[1]
        # p1_1 = pArray[-1]

        P.append(pArray)
    return P


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

#%%
#summations

l_max = 10
l = sy.Symbol("l", real=True, positive=True, integer=True)
m = sy.IndexedBase("l")

Js = [sy.Symbol(f'J_{x}', real=True) for x in range(2, l_max)]
Cs = [sy.Symbol(f'C_{x}', real=True) for x in range(2, l_max)]
Ss = [sy.Symbol(f'S_{x}', real=True) for x in range(2, l_max)]

def jFunc(B):
    return Js[B]

def cFunc(B):
    return Cs[B]

def sFunc(B):
    return Ss[B]

J = sy.IndexedBase('J', real=True)
C = sy.IndexedBase('C', real=True)
S = sy.IndexedBase('S', real=True)
P = sy.IndexedBase(r'P[sin(\phi)]', real=True)

rSy = symbolicElements.RadiusSymbol
rE = sy.Symbol("R_e", real=True, positive=True)
mu = sy.Symbol(r'\mu', real=True, positive=True)
U = mu/rSy * (1-sy.Sum((J[l]*((rE/rSy)**l)*P[l, 0]), (l, 2, l_max)))
jh.showEquation("U", U)

