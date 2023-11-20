#%%
import sympy as sy
import math
import os
import sys
sy.init_printing()

from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
import scipyPaperPrinter as jh #type: ignore
from scipy.optimize import fsolve, root #type: ignore
from IPython.display import display
t = sy.Symbol('t')
t0 = sy.Symbol('t_0')
tf = sy.Symbol('t_f')

kepElements = KepModule.CreateSymbolicElements()
kepElements.TrueAnomaly = sy.Symbol('ta')
kot =KepModule.CreateSymbolicElements(t)
kot.TrueAnomaly = sy.Function('ta')(t)
accel = Cartesian(sy.Symbol('F_r'), sy.Symbol('F_t'), sy.Symbol('F_c'))
keplerianEquationsOfMotion =  KepModule.GaussianEquationsOfMotion(kepElements, accel)
# jh.showEquation("dadt", keplerianEquationsOfMotion.SemiMajorAxisDot)
# jh.showEquation("dedt", keplerianEquationsOfMotion.EccentricityDot)
# jh.showEquation("dvdt", keplerianEquationsOfMotion.TrueAnomalyDot)

muVal = 3.986004418e5
subsDict = {accel.X: 0.001/3600, accel.Y:0, accel.Z:0, kepElements.GravitationalParameter:muVal}

aDotMaybeSolvable = keplerianEquationsOfMotion.SemiMajorAxisDot
aDotMaybeSolvable = aDotMaybeSolvable.subs(subsDict)
eDot = keplerianEquationsOfMotion.EccentricityDot.subs(subsDict)
taDot = keplerianEquationsOfMotion.TrueAnomalyDot.subs(subsDict)

jh.showEquation("\dot{ta}", taDot)
jh.showEquation("\dot{e}", eDot)
jh.showEquation("\dot{a}", aDotMaybeSolvable)

# since this is a top level script that knows what it is doing, don't do the casts on the symbols
lhs = [kot.SemiMajorAxis.diff(t).doit(), kot.Eccentricity.diff(t).doit(), kot.TrueAnomaly.diff(t).doit()]#type: ignore
rhs = sy.Matrix([[aDotMaybeSolvable], [eDot], [taDot]])
display(rhs)

#%%

lhs0 = sy.Matrix([[42164, 0.35, 0.01]]).transpose()
step = 1800
tF = 3600
lhsP1Guess = sy.Matrix([[42180, 0.36, 10*math.pi/180.0]]).transpose()
stateArray = sy.Matrix([[kepElements.SemiMajorAxis, kepElements.Eccentricity, kepElements.TrueAnomaly]]).transpose()
display(stateArray)
dydx = sy.lambdify(stateArray, rhs, 'sympy')

def toSolve(guess) :   
    print(guess)
    sma0 = lhs0[0,0]
    ecc0 = lhs0[1,0]
    ta0 = lhs0[2,0]

    smaPlusGuess = guess[0]
    eccPlusGuess = guess[1]
    taPlusGuess = guess[2]

    daGuess = smaPlusGuess - sma0
    deGuess = eccPlusGuess - ecc0
    dtaGuess = taPlusGuess - ta0

    f0=dydx(sma0, ecc0, ta0).evalf()
    fPlus1=dydx(smaPlusGuess, eccPlusGuess, taPlusGuess).evalf()
    trapAnswer = (step*0.5)*(f0+fPlus1)
    print("trap answer is " + str(trapAnswer))
    return [trapAnswer[0]-daGuess, trapAnswer[1]-deGuess, trapAnswer[2]-dtaGuess]
display(dydx(42164, 0.35, math.pi/2.0))
root = root(toSolve, [42170, 0.36, 10*math.pi/180.0], method='hybr', tol=1e-5)
#root = fsolve(toSolve, [42170, 0.36, 10*math.pi/180.0])
print(root)

# this might not seem like much, but I finally answered a huge question I've had, how do I 
# approximate a system of ODE's using some kind of discretization scheme. In this case, 
# the trapezoidal rule.  

#https://math.stackexchange.com/questions/2338477/trapezoidal-rule-for-system-of-ode

# it might seem overly simple.  If you are going to do multiple steps, you need to do this multiple times, or build a larger matrix.