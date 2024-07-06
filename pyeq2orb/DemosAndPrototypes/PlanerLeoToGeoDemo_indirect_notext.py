#%%
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up
# these two appends do not conflict with each-other

from IPython.display import display
from scipy.integrate import solve_ivp #type:ignore
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import math #type:ignore
from scipy.optimize import fsolve #type:ignore
from pyeq2orb.ProblemBase import ProblemVariable, Problem
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from pyeq2orb.Numerical import ScipyCallbackCreators
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper, OdeLambdifyHelperWithBoundaryConditions
from pyeq2orb.Utilities.SolutionDictionaryFunctions import GetValueFromStateDictionaryAtIndex
from pyeq2orb import SafeSubs
import scipyPaperPrinter as jh #type:ignore
from datetime import datetime
from typing import List
print(str(datetime.now()))
# constants
g = 9.80665
mu = 3.986004418e14  
thrust = 20.0
isp = 6000.0
m0 = 1500.0

# initial values
r0 = 6678000.0
u0 = 0.0
v0 = sy.sqrt(mu/r0) # circular
lon0 = 0.0
# I know from many previous runs that this is the time needed to go from LEO to GEO.
# However, below works well wrapped in another fsolve to control the final time for a desired radius.
tfVal  = 3600*3.97152*24 
tfOrg = tfVal

# these are options to switch to try different things
scaleElements = True
scaleTime = scaleElements and False

# make the time array
tArray = np.linspace(0.0, tfOrg, 1200)
if scaleTime:
    tfVal = 1.0
    tArray = np.linspace(0.0, 1.0, 1200)


baseProblem = ContinuousThrustCircularOrbitTransferProblem()

initialStateValues = baseProblem.CreateVariablesAtTime0(baseProblem.StateVariables)
problem = None #Problem
problem = baseProblem

if scaleElements :
    newSvs = Problem.CreateBarVariables(problem.StateVariables, problem.TimeSymbol) 
    scaleTimeFactor = None
    if scaleTime :
        scaleTimeFactor = problem.TimeFinalSymbol
    problem = baseProblem.ScaleStateVariables(newSvs, {problem.StateVariables[0]: newSvs[0] * initialStateValues[0], 
                                                       problem.StateVariables[1]: newSvs[1] * initialStateValues[2], 
                                                       problem.StateVariables[2]: newSvs[2] * initialStateValues[2], 
                                                       problem.StateVariables[3]: newSvs[3]}) #type: ignore
    if scaleTime :
        tau = sy.Symbol('\tau', real=True)
        problem = problem.ScaleTime(tau, sy.Symbol('\tau_0', real=True), sy.Symbol('\tau_f', real=True), tau*problem.TimeFinalSymbol)  
        problem.OtherArguments.append(baseProblem.TimeFinalSymbol) 

stateAtTf = SafeSubs(problem.StateVariables, {problem.TimeSymbol: problem.TimeFinalSymbol})

# register constants (this dictionary gets shared around)
constantsSubsDict = problem.SubstitutionDictionary
constantsSubsDict[baseProblem.Isp] = isp
constantsSubsDict[baseProblem.MassInitial] = m0
constantsSubsDict[baseProblem.Gravity] = g
constantsSubsDict[baseProblem.Mu]= mu
constantsSubsDict[baseProblem.Thrust] = thrust

# register initial state values
constantsSubsDict.update(zip(initialStateValues, [r0, u0, v0, lon0]))
if scaleElements :
    originalProblem = problem
    # and reset the real initial values using tau_0 instead of time
    initialValuesAtTau0 = SafeSubs(initialStateValues, {baseProblem.TimeInitialSymbol: problem.TimeInitialSymbol})
    constantsSubsDict.update(zip(initialValuesAtTau0, [r0, u0, v0, lon0]))

    r0= r0/r0
    u0=u0/v0
    v0=v0/v0
    lon0=lon0/1.0
    # add the scaled initial values (at tau_0).  We should NOT need to add these at t_0
    initialScaledStateValues = problem.CreateVariablesAtTime0(problem.StateVariables)
    constantsSubsDict.update(zip(initialScaledStateValues, [r0, u0, v0, lon0]))     


jh.t = problem._timeSymbol # needed for cleaner printed equations

jh.showEquation("J", baseProblem.CostFunction, False)

for i in range(0, len(problem.StateVariableDynamics)) :
    jh.showEquation(problem.StateVariables[i].diff(problem.TimeSymbol), problem.StateVariableDynamics[i], [problem.TimeInitialSymbol])

for bc in problem.BoundaryConditions :
    jh.showEquation(0, bc, False)


# this next block does most of the problem, pretty standard optimal control actions

orgSvCount = len(problem.StateVariables)
costateSymbols = problem.CreateCoVector(problem.StateVariables, r'\lambda', problem.TimeSymbol) # preemptively making the costate values
hamiltonian = problem.CreateHamiltonian(costateSymbols)
jh.showEquation("H", hamiltonian)
lambdaDotExpressions = problem.CreateLambdaDotCondition(hamiltonian)
for i in range(0, 4):
    problem.AddCostateVariable(ProblemVariable(costateSymbols[i], lambdaDotExpressions[i]))
    jh.showEquation(costateSymbols[i].diff(problem.TimeSymbol), lambdaDotExpressions[i, 0])    


# NOTE that this call adds the lambdas to the integration state!
dHdu = problem.CreateHamiltonianControlExpressions(hamiltonian)[0]
jh.showEquation('\\frac{\\partial{H}}{\\partial{u}}=0', dHdu)

controlSolved = sy.solve(dHdu, problem.ControlVariables[0])[0] # something that may be different for other problems is when there are multiple control variables
jh.showEquation(problem.ControlVariables[0], controlSolved)

# update ALL equations of motion with the new expression for the control variable
controlSubsDict = {problem.ControlVariables[0]: controlSolved}
# the trig simplification needs the deep=True for this problem to make the equations even cleaner
for i in range(0, len(problem.StateVariableDynamics)):
    problem.StateVariableDynamics[i] = SafeSubs(problem.StateVariableDynamics[i],controlSubsDict).trigsimp(deep=True).simplify() # some simplification to make numerical code more stable later, and that is why this code forces us to do things somewhat manually.  There are often special things like this that we ought to do that you can't really automate.
    jh.showEquation(problem.StateVariables[i].diff(problem.TimeSymbol), problem.StateVariableDynamics[i], [problem.TimeInitialSymbol])
constantsSubsDict[problem.ControlVariables[0]]  =controlSolved

# your choice of the nu vector here controls which transversality condition we use
#nus = [sy.Symbol('B_{u_f}'), sy.Symbol('B_{v_f}')]
nus = [] #type: List[sy.Symbol]
lambdasFinal = SafeSubs(costateSymbols, {problem.TimeSymbol: problem.TimeFinalSymbol})
# make the transversality conditions
if len(nus) != 0:
    transversalityCondition = problem.TransversalityConditionsByAugmentation(nus, lambdasFinal)
else:
    transversalityCondition = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, sy.Symbol(r'dt_f'), lambdasFinal)
# and add them to the problem
for xvers in transversalityCondition :
    jh.showEquation(0, xvers, [problem.TimeInitialSymbol])



problem.BoundaryConditions.extend(transversalityCondition)


#TODO: This is a bad order of operations bug.  The final-time BC cannot be present in the base problem's BC's when the transversality conditions are made
if scaleTime : # add BC if we are working with the final time (not all solvers need this, but when the same number of BC's and variables are required by the solver [like fsolve does] then...)
    problem.BoundaryConditions.append(baseProblem.TimeFinalSymbol-tfOrg)


# lambda_lon is always 0, so do that cleanup
problem.BoundaryConditions.remove(transversalityCondition[-1])
lmdTheta = costateSymbols.pop()
problem._costateElements.pop()
problem.StateVariables.remove(problem.StateVariables[3])
constantsSubsDict[lmdTheta]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeFinalSymbol)]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeInitialSymbol)]=0

initialFSolveStateGuess = ContinuousThrustCircularOrbitTransferProblem.CreateInitialLambdaGuessForLeoToGeo(problem, controlSolved, costateSymbols)
if scaleTime :
    initialFSolveStateGuess.append(tfOrg)


otherArgs = []
if scaleTime :
    otherArgs.append(baseProblem.TimeFinalSymbol)
if len(nus) > 0 :
    otherArgs.extend(nus)
stateAndLambdas = []
stateAndLambdas.extend(problem.StateVariables)
stateAndLambdas.extend(costateSymbols)
odeState = [problem.TimeSymbol, stateAndLambdas, otherArgs]

integrationSymbols = [] #List[sy.Symbol]
integrationSymbols.extend(problem.StateVariables)
integrationSymbols.extend([x.Element for x in problem._costateElements])

equationsOfMotion = problem.EquationsOfMotionAsEquations
for i in range(0, len(problem._costateElements)) :
    equationsOfMotion.append(sy.Eq(problem._costateElements[i].Element.diff(problem.TimeSymbol), problem._costateElements[i].FirstOrderDynamics))

initialLambdaValues = SafeSubs(problem.CostateSymbols, {problem.TimeSymbol:problem.TimeInitialSymbol})

lambdifyHelper = OdeLambdifyHelperWithBoundaryConditions(problem.TimeSymbol, problem.TimeInitialSymbol, problem.TimeFinalSymbol, integrationSymbols, equationsOfMotion, initialLambdaValues, problem.BoundaryConditions, otherArgs, problem.SubstitutionDictionary)


# this next block is for when we are using the adjoined form of the transversality condition.  However it is also REALLY useful to just have 
# the equations of motion in a form that we can evaluate ourselves.
odeIntEomCallback = lambdifyHelper.CreateSimpleCallbackForSolveIvp()
if len(nus) > 0 :
    # run a test solution to get a better guess for the final nu values, this is a good technique, but 
    # it is still a custom-to-this-problem piece of code because it is still initial-guess work
    initialFSolveStateGuess.append(initialFSolveStateGuess[1])
    initialFSolveStateGuess.append(initialFSolveStateGuess[2])  
    argsForOde = []
    if scaleTime :
        argsForOde.append(tfOrg)
    argsForOde.append(initialFSolveStateGuess[1])
    argsForOde.append(initialFSolveStateGuess[2])  
    
    testSolution = solve_ivp(odeIntEomCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *initialFSolveStateGuess[0:3]], args=tuple(argsForOde), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
    #testSolution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, *initialFSolveStateGuess[0:3]], tArray, args=tuple(argsForOde))
    finalValues = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(testSolution)
    initialFSolveStateGuess[-2] = finalValues[5]
    initialFSolveStateGuess[-1] = finalValues[6]

stateForBoundaryConditions = []
stateForBoundaryConditions.extend(SafeSubs(problem.StateVariables, {problem.TimeSymbol: problem.TimeInitialSymbol}))
stateForBoundaryConditions.extend(SafeSubs(problem.StateVariables, {problem.TimeSymbol: problem.TimeFinalSymbol}))
stateForBoundaryConditions.extend(SafeSubs([x.Element for x in problem._costateElements], {problem.TimeSymbol: problem.TimeFinalSymbol}))
stateForBoundaryConditions.extend(SafeSubs([x.Element for x in problem._costateElements], {problem.TimeSymbol: problem.TimeFinalSymbol}))
stateForBoundaryConditions.extend(otherArgs)



fSolveCallback = ContinuousThrustCircularOrbitTransferProblem.createSolveIvpSingleShootingCallbackForFSolve(problem, integrationSymbols, [r0, u0, v0, lon0], tArray, odeIntEomCallback, problem.BoundaryConditions, SafeSubs(costateSymbols, {problem.TimeSymbol: problem.TimeInitialSymbol}), otherArgs)
#%%

thing = solve_ivp(odeIntEomCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, 1,2,3], args=tuple(), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
print(thing)
#%%
def ivpCallback(tArray, state):
    return solve_ivp(odeIntEomCallback, [tArray[0], tArray[-1]], state, args=tuple(), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)


from pyeq2orb.Numerical.SimpleProblemCallers import blackBoxSingleShootingFunctions, SimpleEverythingAnswer, fSolveSingleShootingSolver


initialStateValues = [r0, u0, v0, lon0, 1.0, 0.00010000000130634948, 1.0]

numerical = OdeLambdifyHelperWithBoundaryConditions.CreateFromProblem(problem)
numerical.SymbolsToSolveForWithBoundaryConditions.clear()
numerical.SymbolsToSolveForWithBoundaryConditions.extend(initialFSolveStateGuess)
numerical.SymbolsToSolveForWithBoundaryConditions.append(tfVal)
numerical.ApplySubstitutionDictionaryToExpressions()
ivpCallback = numerical.CreateSimpleCallbackForSolveIvp()

integrationVariables = []
integrationVariables.extend(problem.StateVariables)
integrationVariables.extend(problem.CostateSymbols)

def solve_ivp_wrapper(t, y, args):
    if isinstance(args, list):
        args = tuple(args)
    anAns = solve_ivp(ivpCallback, [t[0], t[-1]], y, dense_output=True, t_eval=t, args=args, method='LSODA')
    anAnsDict = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(integrationVariables, anAns)
    return anAnsDict

bcCallback = numerical.CreateCallbackForBoundaryConditionsWithFullState()
problemEvaluator = blackBoxSingleShootingFunctions(solve_ivp_wrapper, bcCallback[1], integrationVariables, problem.BoundaryConditions)
everything = problemEvaluator.EvaluateProblem(tArray, initialStateValues, None)
print(everything.BoundaryConditionValues)
print(everything.StateHistory)

tArray = np.linspace(0.0, 1.0, 1200)
fSolveSolver = fSolveSingleShootingSolver(problem, problemEvaluator, [ *problem.CostateSymbols[0:3]], problem.BoundaryConditions)
tfEst = 250.0
theAnswer = fSolveSolver.solve([*initialStateValues[4:]], tArray, initialStateValues, parameters=None, full_output=True,  factor=0.2,epsfcn=0.001)
print(theAnswer)







# fSolveCallbackAlt = lambdifyHelper.createCallbackToSolveForBoundaryConditions(ivpCallback, tArray, [r0, u0, v0, lon0, *initialFSolveStateGuess])    
# display(fSolveCallback([1,2,3]))
# display(fSolveCallbackAlt([1,2,3]))
#%%
# print(initialFSolveStateGuess)
# #initialFSolveStateGuess = [1, 0.001, 0.001]
# fSolveSol = fsolve(fSolveCallbackAlt, initialFSolveStateGuess, epsfcn=0.00001, full_output=True) # just to speed things up and see how the initial one works
# print(fSolveSol)


solution = solve_ivp(odeIntEomCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *fSolveSol[0][0:3]], args=tuple(fSolveSol[0][3:len(fSolveSol[0])]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(integrationSymbols, solution)
unscaledResults = solutionDictionary
unscaledTArray = tArray
unscaledResults = problem.DescaleResults(solutionDictionary)
if scaleTime:
    unscaledTArray=tfOrg*tArray

if scaleElements:    
    finalState = GetValueFromStateDictionaryAtIndex(solutionDictionary, -1)
    jh.showEquation(stateAtTf[0], finalState[problem.StateVariables[0]], False)
    jh.showEquation(stateAtTf[1], finalState[problem.StateVariables[1]], False)
    jh.showEquation(stateAtTf[2], finalState[problem.StateVariables[2]], False)
    jh.showEquation(stateAtTf[3], (finalState[problem.StateVariables[3]]%(2*math.pi))*180.0/(2*math.pi), False)

#%%
baseProblem.PlotSolution(tArray*tfOrg, unscaledResults, "Test")
jh.showEquation(baseProblem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledResults[problem.StateVariables[0]][-1], False)
jh.showEquation(baseProblem.StateVariables[1].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledResults[problem.StateVariables[1]][-1], False)
jh.showEquation(baseProblem.StateVariables[2].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledResults[problem.StateVariables[2]][-1], False)
jh.showEquation(baseProblem.StateVariables[3].subs(problem.TimeSymbol, problem.TimeFinalSymbol), (unscaledResults[problem.StateVariables[3]][-1]%(2*math.pi))*180.0/(2*math.pi), False)

[hamltVals, dhduValus, d2hdu2Valus] = problem.EvaluateHamiltonianAndItsFirstTwoDerivatives(solutionDictionary, tArray, hamiltonian, {problem.ControlVariables[0]: controlSolved}, {baseProblem.TimeFinalSymbol: tfOrg})
plt.title("Hamlitonion and its derivatives")
plt.plot(tArray/86400, hamltVals, label="Hamiltonian")
plt.plot(tArray/86400, dhduValus, label=r'$\frac{dH}{du}$')
plt.plot(tArray/86400, d2hdu2Valus, label=r'$\frac{d^2H}{du^2}$')

plt.tight_layout()
plt.grid(alpha=0.5)
plt.legend(framealpha=1, shadow=True)
plt.show()   
# %%
