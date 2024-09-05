#%%
from IPython.display import display
from scipy.integrate import solve_ivp #type:ignore
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import math #type:ignore
from scipy.optimize import fsolve #type:ignore
from pyeq2orb.ProblemBase import ProblemVariable, Problem #type:ignore
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem #type:ignore
from pyeq2orb.Numerical import ScipyCallbackCreators #type:ignore
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper, OdeLambdifyHelperWithBoundaryConditions #type:ignore
from pyeq2orb.Utilities.SolutionDictionaryFunctions import GetValueFromStateDictionaryAtIndex #type:ignore
from pyeq2orb import SafeSubs #type:ignore
import scipyPaperPrinter as jh #type:ignore
from datetime import datetime
from typing import List
from pyeq2orb.Numerical.SimpleProblemCallers import SimpleIntegrationAnswer,SingleShootingFunctions  #type:ignore
from pyeq2orb.Numerical.SimpleProblemCallers import BlackBoxSingleShootingFunctions, fSolveSingleShootingSolver, BlackBoxSingleShootingFunctionsFromLambdifiedFunctions
import matplotlib.pyplot as plot #type:ignore
import pyeq2orb.Graphics.Primitives as prim#type:ignore
from pyeq2orb.Graphics.Plotly2DModule import plot2DLines #type:ignore

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

r0_org = r0
u0_org = u0
v0_org = v0
lon0_org = lon0
# I know from many previous runs that this is the time needed to go from LEO to GEO.
# However, below works well wrapped in another fsolve to control the final time for a desired radius.
tfVal  = 3600*3.97152*24 
tfOrg = tfVal

# these are options to switch to try different things
scaleElements = False
scaleTime = scaleElements and False
# your choice of the nu vector here controls which transversality condition we use
nus = [sy.Symbol('B_{u_f}'), sy.Symbol('B_{v_f}')]
#nus = [] #type: List[sy.Symbol]

# make the time array
tArray = np.linspace(0.0, tfOrg, 1200)
if scaleTime:
    tfVal = 1.0
    tArray = np.linspace(0.0, 1.0, 1200)


baseProblem = ContinuousThrustCircularOrbitTransferProblem()
problem :Problem = baseProblem
initialStateSymbols = baseProblem.StateSymbolsInitial()

# register constants (this dictionary gets shared around)
constantsSubsDict = problem.SubstitutionDictionary
constantsSubsDict[baseProblem.Isp] = isp
constantsSubsDict[baseProblem.MassInitial] = m0
constantsSubsDict[baseProblem.Gravity] = g
constantsSubsDict[baseProblem.Mu]= mu
constantsSubsDict[baseProblem.Thrust] = thrust
constantsSubsDict.update(zip(initialStateSymbols, [r0, u0, v0, lon0]))

stateAtTf = baseProblem.StateSymbolsFinal()

jh.showEquation("J", baseProblem.CostFunction, False)

for i in range(0, len(problem.StateVariableDynamics)) :
    jh.showEquation(problem.StateSymbols[i].diff(problem.TimeSymbol), problem.StateVariableDynamics[i], [problem.TimeInitialSymbol])

for bc in problem.BoundaryConditions :
    jh.showEquation(0, bc, False)


# this next block does most of the problem, pretty standard optimal control actions

orgSvCount = len(problem.StateSymbols)
costateSymbols = problem.CreateCostateVariables(problem.StateSymbols, r'\lambda', problem.TimeSymbol) # preemptively making the costate values
hamiltonian = problem.CreateHamiltonian(costateSymbols)

lambdaDotExpressions = problem.CreateLambdaDotCondition(hamiltonian)
dHdu = problem.CreateHamiltonianControlExpressions(hamiltonian)[0]
for i in range(0, 4):
    problem.AddCostateVariable(ProblemVariable(costateSymbols[i], lambdaDotExpressions[i]))
controlSolved = sy.solve(dHdu, problem.ControlSymbols[0])[0] # something that may be different for other problems is when there are multiple control variables
eomWithControlSolved = problem.EquationsOfMotionAsEquations[1].rhs.subs(problem.ControlSymbols[0], controlSolved)
# update ALL equations of motion with the new expression for the control variable
controlSubsDict = {problem.ControlSymbols[0]: controlSolved}
for i in range(0, len(problem.StateVariableDynamics)):
    problem.StateVariableDynamics[i] = SafeSubs(problem.StateVariableDynamics[i],controlSubsDict).trigsimp(deep=True).simplify() # some simplification to make numerical code more stable later, and that is why this code forces us to do things somewhat manually.  There are often special things like this that we ought to do that you can't really automate.
constantsSubsDict[problem.ControlSymbols[0]]  =controlSolved
lambdasFinal = SafeSubs(costateSymbols, {problem.TimeSymbol: problem.TimeFinalSymbol})
if len(nus) != 0:
    transversalityCondition = problem.TransversalityConditionsByAugmentation(nus, lambdasFinal)
else:
    transversalityCondition = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, sy.Symbol(r'dt_f'), lambdasFinal)
problem.BoundaryConditions.extend(transversalityCondition)
if len(nus) > 0:
    problem.OtherArguments.extend(nus)




jh.showEquation("H", hamiltonian)
for i in range(0, 4):
    jh.showEquation(costateSymbols[i].diff(problem.TimeSymbol), lambdaDotExpressions[i, 0])    
jh.showEquation('\\frac{\\partial{H}}{\\partial{u}}=0', dHdu)
jh.showEquation(problem.ControlSymbols[0], controlSolved)
jh.showEquation("a", eomWithControlSolved.trigsimp(deep=True).simplify())
# the trig simplification needs the deep=True for this problem to make the equations even cleaner
for i in range(0, len(problem.StateVariableDynamics)):
    jh.showEquation(problem.StateSymbols[i].diff(problem.TimeSymbol), problem.StateVariableDynamics[i], [problem.TimeInitialSymbol])

# and add them to the problem
jh.printMarkdown('The transversality conditions')
for xvers in transversalityCondition :
    jh.showEquation(0, xvers, [problem.TimeInitialSymbol])

#if scaleTime : # add BC if we are working with the final time (not all solvers need this, but when the same number of BC's and variables are required by the solver [like fsolve does] then...)
#    problem.BoundaryConditions.append(stateAtTf[0]/r0_org-42162.0/r0_org)




#%%

if scaleElements :
    originalProblem = problem
    newSvs = Problem.CreateBarVariables(problem.StateSymbols, problem.TimeSymbol) 
    problem = baseProblem.ScaleStateVariables(newSvs, {problem.StateSymbols[0]: newSvs[0] * initialStateSymbols[0], 
                                                       problem.StateSymbols[1]: newSvs[1] * initialStateSymbols[2], 
                                                       problem.StateSymbols[2]: newSvs[2] * initialStateSymbols[2], 
                                                       problem.StateSymbols[3]: newSvs[3]}) #type: ignore
    if scaleTime :
        tau = sy.Symbol(r'\tau', real=True)
        problem = problem.ScaleTime(tau, sy.Symbol(r'\tau_0', real=True), sy.Symbol(r'\tau_f', real=True), tau*problem.TimeFinalSymbol)  
        jh.t = problem._timeSymbol # needed for cleaner printed equations
        controlSolved = SafeSubs(controlSolved, {originalProblem.TimeSymbol: tau})

    constantsSubsDict=problem.SubstitutionDictionary
    # and reset the real initial values using tau_0 instead of time
    initialValuesAtTau0 = SafeSubs(problem.StateSymbols, {baseProblem.TimeInitialSymbol: problem.TimeInitialSymbol})
    constantsSubsDict.update(zip(initialValuesAtTau0, [r0, u0, v0, lon0]))

    r0= r0/r0
    u0=u0/v0
    v0=v0/v0
    lon0=lon0/1.0
    # add the scaled initial values (at tau_0).  While we shouldn't need this in general, to make a guess for initial costate values, it is helpful
    initialScaledStateValues = problem.StateSymbolsInitial()
    constantsSubsDict.update(zip(initialScaledStateValues, [r0, u0, v0, lon0]))  

# lambda_lon is always 0, so do that cleanup
problem.BoundaryConditions.remove(problem.BoundaryConditions[-1])
lmdTheta = problem.CostateSymbols[-1]
problem._costateElements.pop()
constantsSubsDict[lmdTheta]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeFinalSymbol)]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeInitialSymbol)]=0


#%%
initialFSolveStateGuess = [26.227553300281922,1277.0845661479312,  23647.73525022148]# ContinuousThrustCircularOrbitTransferProblem.CreateInitialLambdaGuessForLeoToGeo(problem, controlSolved, problem.CostateSymbols)


for eom in problem.EquationsOfMotionAsEquations:
    jh.showEquation(eom.lhs, eom.rhs)

for co in problem.CostateVariables:
    jh.showEquation(sy.diff(co.Element, problem.TimeSymbol), co.FirstOrderDynamics.simplify())

for bc in problem.BoundaryConditions :
    jh.showEquation(0, bc)

#%%
numerical = OdeLambdifyHelperWithBoundaryConditions.CreateFromProblem(problem)
numerical.ApplySubstitutionDictionaryToExpressions()
ivpCallback = numerical.CreateSimpleCallbackForSolveIvp()

integrationVariables = []
integrationVariables.extend(problem.StateSymbols)
integrationVariables.extend(problem.CostateSymbols)

def solve_ivp_wrapper(t, y, *args):
    if isinstance(args, list):
        args = tuple(args)
    if isinstance(args, float):
        args = (args,)
    anAns = solve_ivp(ivpCallback, [t[0], t[-1]], y, t_eval=t, dense_output=True, args=args, method='LSODA', rtol=1.49012e-8, atol=1.49012e-11)
    anAnsDict = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(integrationVariables, anAns)
    return (anAnsDict, anAns)

bcCallback = numerical.CreateCallbackForBoundaryConditionsWithFullState()
#%%
betterFSolveCallback = SingleShootingFunctions.CreateBoundaryConditionCallbackFromLambdifiedCallback(bcCallback)
problemEvaluator = BlackBoxSingleShootingFunctionsFromLambdifiedFunctions(solve_ivp_wrapper, bcCallback, integrationVariables, problem.BoundaryConditions, problem.OtherArguments)
fSolveInputSymbols = problem.CostateSymbols[:3]
#if scaleTime:
#    fSolveInputSymbols.append(originalProblem.TimeFinalSymbol)
solver = fSolveSingleShootingSolver(problemEvaluator, fSolveInputSymbols, problem.BoundaryConditions[:-1])
#initialSolverGuess =  [1.0, 0.00010000000130634948, 1.0]
initialSolverGuess = ContinuousThrustCircularOrbitTransferProblem.CreateInitialLambdaGuessForLeoToGeo(problem, controlSolved, costateSymbols)
if len(nus) > 0:
    initialSolverGuess.append(initialSolverGuess[2])
    initialSolverGuess.append(initialSolverGuess[3])
initialStateValues = [r0, u0, v0, lon0, *initialFSolveStateGuess]

argsArray = None
if scaleTime:
    argsArray = []
    #initialSolverGuess.append(tfOrg)
    argsArray.append(tfOrg)

print(initialStateValues)
ans = solver.solve(initialSolverGuess, tArray, initialStateValues, full_output=True, factor=1.0, epsfcn=0.01)
print(ans.SolverResult)



#%%

solution = ans.EvaluatedAnswer.RawIntegratorOutput
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(integrationVariables, solution)
unscaledResults = solutionDictionary
unscaledTArray = tArray
unscaledResults = problem.DescaleResults(solutionDictionary)


# and validation
mDot = -1*thrust/(isp*g)
handWrittenDiffeq = ContinuousThrustCircularOrbitTransferProblem.scaledHandWrittenDifferentialEquations(r0_org, u0_org, v0_org, lon0_org, thrust, m0, mDot, 0.0, mu, scaleElements, scaleTime)
initialStateForValidation = []
for i in range(0, len(initialStateValues)):
    initialStateForValidation.append(ans.EvaluatedAnswer.RawIntegratorOutput.y[i][0])

#%%
handWrittenCallback = lambda t, y, args: handWrittenDiffeq.scaledDifferentialEquationCallback(t, y, args)
#nonScaledInitialStateThatWorkedLongAgo = [r0_org, u0_org, v0_org, lon0_org, 26.227553300281922, 1277.084566147931, 23647.73525022148]
validationAnswer = solve_ivp(handWrittenCallback, [tArray[0], tArray[-1]], initialStateForValidation, t_eval=tArray, dense_output=True, args=[tfOrg], method='LSODA', rtol=1.49012e-8, atol=1.49012e-11)
validationSolutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(integrationVariables, validationAnswer)
unscaledValidationResults = validationSolutionDictionary
unscaledValidationResults = problem.DescaleResults(unscaledValidationResults)



if scaleTime:
    unscaledTArray=tfOrg*tArray

if scaleElements:    
    finalState = GetValueFromStateDictionaryAtIndex(solutionDictionary, -1)
    jh.showEquation(stateAtTf[0], finalState[problem.StateSymbols[0]], False)
    jh.showEquation(stateAtTf[1], finalState[problem.StateSymbols[1]], False)
    jh.showEquation(stateAtTf[2], finalState[problem.StateSymbols[2]], False)
    jh.showEquation(stateAtTf[3], (finalState[problem.StateSymbols[3]]%(2*math.pi))*180.0/(2*math.pi), False)


baseProblem.PlotSolution(tArray*tfOrg, unscaledResults, "Test")

display('And the validation answer')

baseProblem.PlotSolution(tArray*tfOrg, unscaledValidationResults, "test_validation")
jh.showEquation(baseProblem.StateSymbols[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledValidationResults[problem.StateSymbols[0]][-1], False)
jh.showEquation(baseProblem.StateSymbols[1].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledValidationResults[problem.StateSymbols[1]][-1], False)
jh.showEquation(baseProblem.StateSymbols[2].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledValidationResults[problem.StateSymbols[2]][-1], False)
jh.showEquation(baseProblem.StateSymbols[3].subs(problem.TimeSymbol, problem.TimeFinalSymbol), (unscaledValidationResults[problem.StateSymbols[3]][-1]%(2*math.pi))*180.0/(2*math.pi), False)

[hamltVals, dhduValus, d2hdu2Valus] = problem.EvaluateHamiltonianAndItsFirstTwoDerivatives(validationSolutionDictionary, tArray, hamiltonian, {problem.ControlSymbols[0]: controlSolved}, {baseProblem.TimeFinalSymbol: tfOrg})
plt.title("Hamlitonion and its derivatives")
plt.plot(tArray/86400, hamltVals, label="Hamiltonian")
plt.plot(tArray/86400, dhduValus, label=r'$\frac{dH}{du}$')
plt.plot(tArray/86400, d2hdu2Valus, label=r'$\frac{d^2H}{du^2}$')

plt.tight_layout()
plt.grid(alpha=0.5)
plt.legend(framealpha=1, shadow=True)
plt.show()   

#%%

x = tArray
lines = []
color = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
for sv in problem.StateSymbols:
    thisDiff = []
    for j in range(0, len(tArray)):
        thisDiff.append(float(unscaledResults[sv][j] - unscaledValidationResults[sv][j]))
    lines.append(prim.XAndYPlottableLineData(tArray.tolist(), thisDiff, str(sv), next(color), 2))
plot2DLines(lines, "absolute differences")

#%%
#initialStateForValidation[1] = 0.2
#initialStateForValidation[2] = 0.3
initialStateForValidation = [0.9999999999999999,
 0.0,
 1.0,
 0.0,
26.227553300281922, 1277.0845661479311, 23647.73525022148]
print(initialStateForValidation)
print(ivpCallback(0.0, initialStateForValidation, tfOrg))
print(handWrittenCallback(0.0, initialStateForValidation, tfOrg))


