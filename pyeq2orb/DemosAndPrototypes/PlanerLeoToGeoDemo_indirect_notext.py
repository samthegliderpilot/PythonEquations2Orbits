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
from pyeq2orb.Numerical.SimpleProblemCallers import SimpleIntegrationAnswer,SingleShootingFunctions
from pyeq2orb.Numerical.SimpleProblemCallers import BlackBoxSingleShootingFunctions, fSolveSingleShootingSolver, BlackBoxSingleShootingFunctionsFromLambdifiedFunctions

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
scaleElements = False
scaleTime = scaleElements and False

# make the time array
tArray = np.linspace(0.0, tfOrg, 1200)
if scaleTime:
    tfVal = 1.0
    tArray = np.linspace(0.0, 1.0, 1200)


baseProblem = ContinuousThrustCircularOrbitTransferProblem()
problem :Problem = baseProblem
initialStateValues = baseProblem.CreateVariablesAtTime0(baseProblem.StateVariables)

# register constants (this dictionary gets shared around)
constantsSubsDict = problem.SubstitutionDictionary
constantsSubsDict[baseProblem.Isp] = isp
constantsSubsDict[baseProblem.MassInitial] = m0
constantsSubsDict[baseProblem.Gravity] = g
constantsSubsDict[baseProblem.Mu]= mu
constantsSubsDict[baseProblem.Thrust] = thrust
constantsSubsDict.update(zip(initialStateValues, [r0, u0, v0, lon0]))


if scaleElements :
    originalProblem = problem
    newSvs = Problem.CreateBarVariables(problem.StateVariables, problem.TimeSymbol) 
    problem = baseProblem.ScaleStateVariables(newSvs, {problem.StateVariables[0]: newSvs[0] * initialStateValues[0], 
                                                       problem.StateVariables[1]: newSvs[1] * initialStateValues[2], 
                                                       problem.StateVariables[2]: newSvs[2] * initialStateValues[2], 
                                                       problem.StateVariables[3]: newSvs[3]}) #type: ignore
    if scaleTime :
        tau = sy.Symbol('\tau', real=True)
        problem = problem.ScaleTime(tau, sy.Symbol('\tau_0', real=True), sy.Symbol('\tau_f', real=True), tau*problem.TimeFinalSymbol)  
        jh.t = problem._timeSymbol # needed for cleaner printed equations

    constantsSubsDict=problem.SubstitutionDictionary
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

stateAtTf = SafeSubs(problem.StateVariables, {problem.TimeSymbol: problem.TimeFinalSymbol})

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
jh.printMarkdown('The transversality conditions')
for xvers in transversalityCondition :
    jh.showEquation(0, xvers, [problem.TimeInitialSymbol])



problem.BoundaryConditions.extend(transversalityCondition)

# lambda_lon is always 0, so do that cleanup
problem.BoundaryConditions.remove(transversalityCondition[-1])
lmdTheta = costateSymbols.pop()
problem._costateElements.pop()
problem.StateVariables.remove(problem.StateVariables[3])
constantsSubsDict[lmdTheta]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeFinalSymbol)]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeInitialSymbol)]=0


initialFSolveStateGuess = ContinuousThrustCircularOrbitTransferProblem.CreateInitialLambdaGuessForLeoToGeo(problem, controlSolved, costateSymbols)

#%%
numerical = OdeLambdifyHelperWithBoundaryConditions.CreateFromProblem(problem)
numerical.ApplySubstitutionDictionaryToExpressions()
ivpCallback = numerical.CreateSimpleCallbackForSolveIvp()

integrationVariables = []
integrationVariables.extend(problem.StateVariables)
integrationVariables.extend(problem.CostateSymbols)

def solve_ivp_wrapper(t, y, *args):
    if isinstance(args, list):
        args = tuple(args)
    if isinstance(args, float):
        args = (args,)
    anAns = solve_ivp(ivpCallback, [t[0], t[-1]], y, t_eval=t, dense_output=True, args=args, method='LSODA')
    anAnsDict = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(integrationVariables, anAns)
    return (anAnsDict, anAns)

boundaryConditionState = numerical.CreateDefaultStateForBoundaryConditions()
bcCallback = numerical.CreateCallbackForBoundaryConditionsWithFullState(boundaryConditionState)
betterFSolveCallback = SingleShootingFunctions.CreateBoundaryConditionCallbackFromLambdifiedCallback(bcCallback[1])
initialStateValues = [r0, u0, v0, lon0, *initialFSolveStateGuess]
problemEvaluator = BlackBoxSingleShootingFunctionsFromLambdifiedFunctions(solve_ivp_wrapper, bcCallback[1], integrationVariables, problem.BoundaryConditions, problem.OtherArguments)
fSolveInputSymbols = problem.CostateSymbols[:3]
if scaleTime:
    fSolveInputSymbols.append(originalProblem.TimeFinalSymbol)
solver = fSolveSingleShootingSolver(problemEvaluator, fSolveInputSymbols, problem.BoundaryConditions)
initialSolverGuess = initialStateValues[4:]
argsArray = None
if scaleTime:
    argsArray = []
    initialSolverGuess.append(tfOrg)
    argsArray.append(tfOrg)

ans = solver.solve(initialSolverGuess, tArray, initialStateValues, argsArray, full_output=True,  factor=0.2,epsfcn=0.001)
print(ans.SolverResult)
#%%




solution = ans.EvaluatedAnswer.RawIntegratorOutput# solve_ivp(odeIntEomCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *fSolveSol[0][0:3]], args=tuple(fSolveSol[0][3:len(fSolveSol[0])]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(integrationVariables, solution)
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
