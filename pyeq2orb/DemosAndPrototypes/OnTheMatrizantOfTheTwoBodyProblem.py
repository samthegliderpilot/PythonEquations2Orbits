#%%
import __init__  #type:ignore
import sympy as sy
import os
import sys
import math
from collections import OrderedDict
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)
sy.init_printing()

from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements, CreateSymbolicElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
import pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule as mee
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs
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

t = sy.Symbol('t', real=True)
mu = sy.Symbol(r'\mu', real=True, positive=True)
x = sy.Function('x', real=True)(t)
y = sy.Function('y', real=True)(t)
z = sy.Function('z', real=True)(t)
r = sy.Function('r', real=True, positive=True)(t,x,y,z)
rExp = sy.sqrt(x**2+y**2+z**2)

subs = {}
subs[r] = rExp
f_x = -mu*x/r**3
f_y = -mu*y/r**3
f_z = -mu*z/r**3
A = sy.Matrix([[0,0,0, 1, 0, 0],[0,0,0, 0, 1, 0], [0,0,0, 0, 0, 1],[0,0,0, 0, 0, 0],[0,0,0, 0, 0, 0],[0,0,0, 0, 0, 0]])
A[3, 0] = f_x.diff(x)
A[3, 1] = f_x.diff(y)
A[3, 2] = f_x.diff(z)
A[4, 0] = f_y.diff(x)
A[4, 1] = f_y.diff(y)
A[4, 2] = f_y.diff(z)
A[5, 0] = f_z.diff(x)
A[5, 1] = f_z.diff(y)
A[5, 2] = f_z.diff(z)
display(A.applyfunc(lambda x : SafeSubs(x, subs)))


S = sy.Matrix([[0,0,0, 1, 0, 0],[0,0,0, 0, 1, 0], [0,0,0, 0, 0, 1],[-1,0,0, 0, 0, 0],[0,-1,0, 0, 0, 0],[0,0,-1, 0, 0, 0]])

#%%
# fundamental matrix
# finding some fundamental R matrix that is... FUNDAM<ENTAL!!!11!!

#%%
# fundamental matrix R of the Two Body Problem
kepElements = CreateSymbolicElements([t,x,y,z], mu)
a = kepElements.SemiMajorAxis
e = kepElements.Eccentricity
i = kepElements.Inclination
w = kepElements.ArgumentOfPeriapsis
raan = kepElements.RightAscensionOfAscendingNode
ta = kepElements.TrueAnomaly
n = kepElements.MeanMotion
subs[r] = kepElements.Radius
ea = sy.Function('E', real=True)(*[t,e,a])

perToInertial = kepElements.PerifocalToInertialRotationMatrix()
xPeri = a*(sy.cos(ea)-e)
yPeri = a*sy.sqrt(1-e**2)*sy.sin(ea)
P = perToInertial[:,0]
Q = perToInertial[:,1]
R = perToInertial[:,2]
xVector = P*xPeri + Q*yPeri

xPeriDot = -n*a**2*sy.sin(ea)/r
yPeriDot = n*a**2*sy.sqrt(1-e**2)*sy.cos(ea)/r
jh.showEquation("Y", yPeri)
jh.showEquation('dY/de', yPeri.diff(e))