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
from typing import List
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


def legendre_functions_goddard_single(expr, gma, n, m, PCache):
    if n == 2 and m==0:
        PCache = []
        PCache.append([1.0])
        PCache.append([expr, expr.diff(gma)])

    # p0_0 = 1.0
    # p1_0 = expr
    # p1_1 = expr.diff(gma)

    if len(PCache) != n:
        PCache.append([0]*n)
    if m == 0:
        pVal= ((2*n-1)*expr*PCache[n-1][0]-(n-1)*PCache[n-2][0])/n
    elif n != m:
        pVal= 2.0#PCache[n-2][m]+(2*n-1)*PCache[n-1][m-1]
    else:
        pVal= (2*n-1)*PCache[n-1][n-1]
    #PCache[-1][m] = pVal
    print(str(n) + " and " + str(m))
    return pVal
        
#%%
class gravityField:
    def __init__(self, expr, gma):
        self._ps = []
        self.expr = expr
        self.gma = gma
        self.coefficientCache = {}

    def makeConstant(self, name, n, m):
        if not name in self.coefficientCache:
            self.coefficientCache[name] = []
        coef = sy.Symbol(name + "{^{" + str(m) + "}_" + str(n) + "}")#, commutative=False)
        self.coefficientCache[name].append(coef)
        return coef

    def makeIt(self, n_max: int, m_max :int, muSy : sy.Symbol, rSy : sy.Expr, rCbSy :sy.Symbol, latSy : sy.Symbol, lonSy :sy.Expr):
        n_max+=1
        m_max+=1
        self._ps = []
        self._ps.append([1.0, 0.0])
        self._ps.append([self.expr, self.expr.diff(self.gma)])

        overallTerms = []
        rCbDivR = rCbSy/rSy
        rCbDivRToN = rCbDivR
        
        for n in range(2, n_max):
            self._ps.append([])    
            mTerms = []
            rCbDivRToN=rCbDivRToN*rCbDivR
            pN0 = self.legendre_functions_goddard_single(n, 0)
            c = self.makeConstant("C", n, 0)
            firstTerm = c *rCbDivRToN * pN0
            for m in range(0, n): # M may start at 1, but it terms in Pcache start at 0
                s = self.makeConstant("S", n, m)
                pNM = self.legendre_functions_goddard_single(n, m)
                sNM = self.makeConstant("S", n, m)
                cNM = self.makeConstant("C", n, m)
                innerTerm = rCbDivRToN * pNM * (sNM* sy.sin(m*lonSy)+cNM*sy.cos(m*lonSy))
                mTerms.append(innerTerm)
            mTerms.reverse()
            totalTerm = firstTerm + sum(mTerms)
            overallTerms.append(totalTerm)
        overallTerms.reverse()
        return mu/rSy * sum(overallTerms)
        #return  sum(overallTerms)


    def legendre_functions_goddard_single(self, n, m):
        PCache = self._ps
        pVal = sy.assoc_legendre(n, m, self.expr)
        # if m == 0:
        #     pVal= ((2*n-1)*self.expr*PCache[n-1][0]-(n-1)*PCache[n-2][0])/n
        # elif n-1 != m:
        #     pVal= PCache[n-2][m]+(2*n-1)*self.expr.diff(self.gma)*PCache[n-1][m-1]
        # else:
        #     pVal= (2*n-1)*self.expr.diff(self.gma)*PCache[n-1][n-1]
        PCache[-1].append(pVal.doit())
        return pVal


builder = gravityField(sy.sin(lat), lat)
potential = builder.makeIt(2, 0, mu, rSy, rE, lat, lon)#.subs(sy.sin(lat), sy.Symbol('r_k', real=True)/rSy)
print(dir(potential))

display(potential.simplify())

# note that C = -J, so vallado has a negative here that I do not
fromVallado = 3*builder.makeConstant("C", 2, 0) * (mu/(2*rSy)) * ((rE/rSy)**2) * (sy.sin(lat)**2 - sy.Rational(1,3))
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
#%%
# read coefficients into an array so I can start trying to lambdify things
c_coefficients = []
s_coefficients = []