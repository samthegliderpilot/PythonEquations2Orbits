#%%
from IPython.display import display
from scipy.integrate import solve_ivp #type: ignore
import matplotlib.pyplot as plt#type: ignore
import numpy as np
import sympy as sy
import plotly.express as px#type: ignore
from pandas import DataFrame #type: ignore
import math
from scipy.optimize import fsolve#type: ignore
from pyeq2orb.ProblemBase import Problem #type: ignore
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem #type: ignore
from pyeq2orb.Numerical import ScipyCallbackCreators #type: ignore
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions #type: ignore
from typing import List, Dict, Callable, Optional, Any, cast
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs
from pyeq2orb.Utilities.SolutionDictionaryFunctions import GetValueFromStateDictionaryAtIndex
import scipyPaperPrinter as jh#type: ignore

# constants
g = 9.80665
mu = 3.986004418e14  
thrust = 20.0
isp = 6000.0
m0 = 1500.0
# and their symbols
gSy = sy.Symbol('g', real=True, positive=True)
muSy = sy.Symbol(r'\mu', real=True, positive=True)
thrustSy = sy.Symbol('T', real=True, positive=True)
ispSy = sy.Symbol('I_{sp}', real=True, positive=True)
m0Sy = sy.Symbol('m_0', real=True, positive=True)
# start populating the substitution dictionary

# I know from many previous runs that this is the time needed to go from LEO to GEO.
# However, below works well wrapped in another fsolve to control the final time for a desired radius.
t0Val = 0.0
tfVal  = 3600*3.97152*24 
tfOrg = tfVal
tArray = np.linspace(0.0, tfOrg, 1200)
tSy = sy.Symbol('t', real=True)
t0Sy = sy.Symbol('t_0', real=True)
tfSy = sy.Symbol('t_f', real=True, positive=True)

# initial values
r0 = 6678000.0
u0 = 0.0
v0 = float(sy.sqrt(mu/r0)) # circular
lon0 = 0.0

# start making the equations of motion
rSy = sy.Function('r', real=True, positive=True)(tSy)
uSy = sy.Function('u', real=True, nonnegative=True)(tSy)
vSy = sy.Function('v', real=True, nonnegative=True)(tSy)
lonSy =  sy.Function(r'\theta', real=True, nonnegative=True)(tSy)
mSy = sy.Function('m', real=True, nonnegative=True)(tSy)

# the control variable
alpSy = sy.Function(r'\alpha', real=True)(tSy)

rDot = uSy
uDot = vSy*vSy/rSy - muSy/(rSy**2) + thrustSy*sy.sin(alpSy)/mSy
vDot = -vSy*uSy/rSy + thrustSy*sy.cos(alpSy)/mSy
lonDot = vSy/rSy
mDot = -1*thrustSy/(ispSy*gSy)

rEquationUnScaled = sy.Eq(rSy.diff(tSy), rDot)
uEquationUnScaled = sy.Eq(uSy.diff(tSy), uDot)
vEquationUnScaled = sy.Eq(vSy.diff(tSy), vDot)
lonEquationUnScaled = sy.Eq(lonSy.diff(tSy), lonDot)
mDotEquationUnscaled = sy.Eq(mSy.diff(tSy), mDot)

# problem specific boundary conditions
bc1 = uSy.subs(tSy, tfSy)
bc2 = vSy.subs(tSy, tfSy)-sy.sqrt(muSy/rSy.subs(tSy, tfSy))

terminalCost = rSy.subs(tSy, tfSy) # maximization problem
unintegratedPathCost = 0
# group things into the common pieces of data needed
eoms = [rEquationUnScaled, uEquationUnScaled, vEquationUnScaled, lonEquationUnScaled, mDotEquationUnscaled]
x = [rSy, uSy, vSy, lonSy, mSy]
x0=SafeSubs([rSy, uSy, vSy, lonSy], {tSy: t0Sy})
xf=SafeSubs([rSy, uSy, vSy, lonSy], {tSy: tfSy})
bcs = [bc1, bc2]
initialLambdaGuesses = [1, 0, 1, 0]
substitutionDictionary = {gSy:g, muSy:mu} #type: Dict[sy.Expr, SymbolOrNumber]
args = [isp, thrust]
scaleDictionary = {rSy:r0, uSy:3, vSy:3, lonSy:1, mSy:1}
initialValues = [r0, u0, v0, lon0, m0]
controlSymbols = [alpSy]

# this is the code I need to write/refactor to be this simple/general
costateVariables = Problem.CreateCostateVariables(x[0:4]) #type: Vector
costateGuesses = [1.0, 1.0, 1.0, 0.0]
hamiltonian = Problem.CreateHamiltonianStatic(tSy, sy.Matrix([x.rhs for x in eoms[0:4]]), unintegratedPathCost, costateVariables) #type: sy.Expr

optimalControl = Problem.CreateControlExpressionsFromHamiltonian(hamiltonian, controlSymbols) # an expression for the control variables
substitutionDictionary[alpSy] = optimalControl[alpSy] # loop

createCostateDifferentialEquations = Problem.CreateLambdaDotEquationsStatic(hamiltonian,tSy, x[0:4], costateVariables) #type: sy.Eq
eoms.extend(createCostateDifferentialEquations)

costateBoundaryConditions = Problem.CreateLambdaDotConditionStatic(hamiltonian, sy.Matrix(x[0:4]))
bcs.extend(costateBoundaryConditions)

dtf = sy.Symbol('dt_f', real=True)
[transversalityCondition, extraEndSymbols] = Problem.TransversalityConditionInTheDifferentialFormStatic(hamiltonian, dtf, SafeSubs(costateVariables, {tSy: tfSy}), rSy.subs(tSy, tfSy), tfSy, bcs, SafeSubs(x[0:4], {tSy: tfSy}) )
#transversalityCondition = IndirectHelper.TransversaitlyConditionInTheDifferentialForm(terminalCost, bcs, tf) # the other transversality condition option
bcs.extend(transversalityCondition)

# and here is where a new script would start to do custom things

# if we want to scale our expressions, do it here:
xFull = []
xFull.extend([r0, u0, v0, lon0, m0])
xFull.extend(initialLambdaGuesses)

#[decaleStateCallback, [xFull, eoms, bcs, initialValues, substitutionDictionary]] = ScaleState(scaleDictionary, [xFull, eoms, bcs, initialValues, substitutionDictionary])
#[decaleTimeCallback, [xFull, eoms, bcs, initialValues, substitutionDictionary]] = ScaleTime([tSy, tau/tfSy], [xFull, eoms, bcs, initialValues, substitutionDictionary])
#args.append(tfSy)
#substitutionDictionary = SafeSubsSubsDict(substitutionDictionary)
#[xFull, eoms, bcs] = SafeSubsEverything(scaleDictionary, [xFull, eoms, bcs, initialValues])

eomState = [tSy, xFull, args]
eomsCallback = sy.lambdify(eomState, eoms)
def propagate(state, t0, tf, args):
    return solve_ivp(eomsCallback, [t0, tf], state, args, dense_output=True)

eomInitialGuess = [0.0, [*initialValues, *costateGuesses], [isp, thrust]]
results = solve_ivp(eomsCallback, [0, 4*86400], eomInitialGuess, args, dense_output=True)

#%%

initialSymbols = SafeSubs(x, {tSy:t0Sy})
finalSymbols = SafeSubs(x, {tSy:tfSy})
directBcState = [[t0Sy,initialSymbols, tfSy, finalSymbols, args]]
fSolveState = xFull[4:]

setFSolveStateIntoSolveIvpState = CallbackHelpers.PutElementsInStateAIntoMatchingEleemntsInStateB(fSolveState, eomState) #lambda eomStateValues, fSolveGuess -> updatedEomStateValues
pullFullBcStateFromSolveIvpResults = lambda results: SolveIvpHelper.GetStandardBcState(bcState, solverGuess, results, args) # lambda solveIvpResults -> directBcState

innerBcCallback = sy.lambdify(bcs, directBcState)



def fsolveCallback(fSolveGuess, args=eomInitialGuess):
    thisIterationInitialEomState = [*args]#TODO, deeper copy
    thisIterationInitialEomState = setFSolveStateIntoSolveIvpState(thisIterationInitialEomState, fSolveGuess)
    results = propagate(thisIterationInitialEomState)
    bcState = pullFullBcStateFromSolveIvpResults(results)
    bcValues = innerBcCallback(bcState)
    return bcValues



helper = OdeLambdifyHelperWithBoundaryConditions(tSy, t0Sy, tfSy, [*x, *costateVariables], initialSymbols, finalSymbols,  eoms, [*bcs, *costateBoundaryConditions, transversalityCondition], [], substitutionDictionary)
#helper.ScaleEquationsOfMotionAndBoundaryConditions(scaleDictionary, {tfSy: 1})

#scaledInitialValues = helper.ScaleState(initialValues)


solveIvpCallback = helper.CreateSimpleCallbackForSolveIvp()
bcInitialState = helper.CreateDefaultStateForBoundaryConditions()
fsolveCallback = helper.CreateCallbackForBoundaryConditionsWithFullState(bcInitialState)

finalScaledSolution = solve_ivp(solveIvpCallback, [t0Val, tfVal], fullInitialStateWithInitialGuess, method='LSODA', rtol=1.49012e-8, atol=1.49012e-11)
fSolveAnswer = fsolve(fsolveCallback, fullInitialStateWithInitialGuess, full_output=True)



# and plot, do whatever with the final answer
# %%
