#%%
import sys
import os
thisFile = os.path.abspath(__file__)
sys.path.append(os.path.abspath(thisFile + '..\\..\\..\\'))

from IPython.display import display
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import math
from scipy.optimize import fsolve
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from pyeq2orb.Numerical import ScipyCallbackCreators
from pyeq2orb.Numerical.LambdifyModule import LambdifyHelper
from pyeq2orb.Utilities.SolutionDictionaryFunctions import GetValueFromStateDictionaryAtIndex
import JupyterHelper as jh

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
scaleTime = scaleElements and True
# your choice of the nu vector here controls which transversality condition we use
nus = [sy.Symbol('B_{u_f}'), sy.Symbol('B_{v_f}')]
#nus = []

baseProblem = ContinuousThrustCircularOrbitTransferProblem()
initialStateValues = baseProblem.CreateVariablesAtTime0(baseProblem.StateVariables)
problem = baseProblem

if scaleElements :
    newSvs = ScaledSymbolicProblem.CreateBarVariables(problem.StateVariables, problem.TimeSymbol) 
    problem = ScaledSymbolicProblem(baseProblem, newSvs, {problem.StateVariables[0]: initialStateValues[0], 
                                                          problem.StateVariables[1]: initialStateValues[2], 
                                                          problem.StateVariables[2]: initialStateValues[2], 
                                                          problem.StateVariables[3]: 1} , scaleTime) # note the int here for the scaling, not a float
stateAtTf = SymbolicProblem.SafeSubs(problem.StateVariables, {problem.TimeSymbol: problem.TimeFinalSymbol})
# make the time array
tArray = np.linspace(0.0, tfOrg, 1200)
if scaleTime:
    tfVal = 1.0
    tArray = np.linspace(0.0, 1.0, 1200)
jh.t = problem._timeSymbol # needed for cleaner printed equations

# register constants
constantsSubsDict = problem.SubstitutionDictionary
constantsSubsDict[baseProblem.Isp] = isp
constantsSubsDict[baseProblem.MassInitial] = m0
constantsSubsDict[baseProblem.Gravity] = g
constantsSubsDict[baseProblem.Mu]= mu
constantsSubsDict[baseProblem.Thrust] = thrust

# register initial state values
constantsSubsDict.update(zip(initialStateValues, [r0, u0, v0, lon0]))
if scaleElements :
    # and reset the real initial values using tau_0 instead of time
    initialValuesAtTau0 = SymbolicProblem.SafeSubs(initialStateValues, {baseProblem.TimeInitialSymbol: problem.TimeInitialSymbol})
    constantsSubsDict.update(zip(initialValuesAtTau0, [r0, u0, v0, lon0]))

    r0= r0/r0
    u0=u0/v0
    v0=v0/v0
    lon0=lon0/1.0
    # add the scaled initial values (at tau_0).  We should NOT need to add these at t_0
    initialScaledStateValues = problem.CreateVariablesAtTime0(problem.StateVariables)
    constantsSubsDict.update(zip(initialScaledStateValues, [r0, u0, v0, lon0])) 
    
# this next block does most of the problem, pretty standard optimal control actions
problem.Lambdas.extend(problem.CreateCoVector(problem.StateVariables, r'\lambda', problem.TimeSymbol))
lambdas = problem.Lambdas
hamiltonian = problem.CreateHamiltonian(lambdas)
lambdaDotExpressions = problem.CreateLambdaDotCondition(hamiltonian)
dHdu = problem.CreateHamiltonianControlExpressions(hamiltonian)[0]
controlSolved = sy.solve(dHdu, problem.ControlVariables[0])[0] # something that may be different for other problems is when there are multiple control variables

# you are in control of the order of integration variables and what EOM's get evaluated, start updating the problem
# NOTE that this call adds the lambdas to the integration state
problem.EquationsOfMotion.update(zip(lambdas, lambdaDotExpressions))
SymbolicProblem.SafeSubs(problem.EquationsOfMotion, {problem.ControlVariables[0]: controlSolved})
# the trig simplification needs the deep=True for this problem to make the equations even cleaner
for (key, value) in problem.EquationsOfMotion.items() :
    problem.EquationsOfMotion[key] = value.trigsimp(deep=True).simplify() # some simplification to make numerical code more stable later, and that is why this code forces us to do things somewhat manually.  There are often special things like this that we ought to do that you can't really automate.

## Start with the boundary conditions
if scaleTime : # add BC if we are working with the final time (kind of silly for this example, but we need an equal number of in's and out's for fsolve later)
    problem.BoundaryConditions.append(baseProblem.TimeFinalSymbol-tfOrg)

# make the transversality conditions
if len(nus) != 0:
    transversalityCondition = problem.TransversalityConditionsByAugmentation(nus)
else:
    transversalityCondition = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, sy.Symbol(r'dt_f'))
# and add them to the problem
problem.BoundaryConditions.extend(transversalityCondition)

initialFSolveStateGuess = ContinuousThrustCircularOrbitTransferProblem.CreateInitialLambdaGuessForLeoToGeo(problem, controlSolved)

# lambda_lon is always 0, so do that cleanup
del problem.EquationsOfMotion[lambdas[3]]
problem.BoundaryConditions.remove(transversalityCondition[-1])
lmdTheta = lambdas.pop()
problem.IntegrationSymbols.pop()
constantsSubsDict[lmdTheta]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeFinalSymbol)]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeInitialSymbol)]=0

# start the conversion to a numerical problem
if scaleTime :
    initialFSolveStateGuess.append(tfOrg)

otherArgs = []
if scaleTime :
    otherArgs.append(baseProblem.TimeFinalSymbol)
if len(nus) > 0 :
    otherArgs.extend(nus)
stateAndLambdas = []
stateAndLambdas.extend(problem.StateVariables)
stateAndLambdas.extend(lambdas)
odeState = [problem.TimeSymbol, stateAndLambdas, otherArgs]

lambdifyHelper = LambdifyHelper(problem.TimeSymbol, stateAndLambdas, problem.EquationsOfMotion.values(), otherArgs, problem.SubstitutionDictionary)

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
    print("solving ivp for final ajoined variable guess")
    testSolution = solve_ivp(odeIntEomCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *initialFSolveStateGuess[0:3]], args=tuple(argsForOde), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
    #testSolution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, *initialFSolveStateGuess[0:3]], tArray, args=tuple(argsForOde))
    finalValues = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(testSolution)
    initialFSolveStateGuess[-2] = finalValues[5]
    initialFSolveStateGuess[-1] = finalValues[6]

print(initialFSolveStateGuess)


stateForBoundaryConditions = []
stateForBoundaryConditions.extend(SymbolicProblem.SafeSubs(problem.IntegrationSymbols, {problem.TimeSymbol: problem.TimeInitialSymbol}))
stateForBoundaryConditions.extend(SymbolicProblem.SafeSubs(problem.IntegrationSymbols, {problem.TimeSymbol: problem.TimeFinalSymbol}))
stateForBoundaryConditions.extend(lambdas)
stateForBoundaryConditions.extend(otherArgs)
boundaryConditionHelper = LambdifyHelper(None, stateForBoundaryConditions, problem.BoundaryConditions, None, problem.SubstitutionDictionary)


fSolveCallback = ContinuousThrustCircularOrbitTransferProblem.createSolveIvpSingleShootingCallbackForFSolve(problem, problem.IntegrationSymbols, [r0, u0, v0, lon0], tArray, odeIntEomCallback, problem.BoundaryConditions, SymbolicProblem.SafeSubs(lambdas, {problem.TimeSymbol: problem.TimeInitialSymbol}), otherArgs)
fSolveSol = fsolve(fSolveCallback, initialFSolveStateGuess, epsfcn=0.000001, full_output=True) # just to speed things up and see how the initial one works
print(fSolveSol)

# final run with answer
solution = solve_ivp(odeIntEomCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *fSolveSol[0][0:3]], args=tuple(fSolveSol[0][3:len(fSolveSol[0])]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
#solution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, *fSolveSol[0][0:3]], tArray, args=tuple(fSolveSol[0][3:len(fSolveSol[0])]))
#solution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, 26.0, 1.0, 27.0], tArray, args=(tfOrg,))
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(problem.IntegrationSymbols, solution)
unscaledResults = solutionDictionary
unscaledTArray = tArray
unscaledResults = problem.DescaleResults(solutionDictionary)
if scaleTime:
    unscaledTArray=tfOrg*tArray

if scaleElements:    
    finalState = GetValueFromStateDictionaryAtIndex(solutionDictionary, -1)
    jh.showEquation(stateAtTf[0], finalState[problem.StateVariables[0]])
    jh.showEquation(stateAtTf[1], finalState[problem.StateVariables[1]])
    jh.showEquation(stateAtTf[2], finalState[problem.StateVariables[2]])
    jh.showEquation(stateAtTf[3], (finalState[problem.StateVariables[3]]%(2*math.pi))*180.0/(2*math.pi))

baseProblem.PlotSolution(tArray*tfOrg, unscaledResults, "Test")
jh.showEquation(baseProblem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledResults[baseProblem.StateVariables[0]][-1])
jh.showEquation(baseProblem.StateVariables[1].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledResults[baseProblem.StateVariables[1]][-1])
jh.showEquation(baseProblem.StateVariables[2].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaledResults[baseProblem.StateVariables[2]][-1])
jh.showEquation(baseProblem.StateVariables[3].subs(problem.TimeSymbol, problem.TimeFinalSymbol), (unscaledResults[baseProblem.StateVariables[3]][-1]%(2*math.pi))*180.0/(2*math.pi))

[hamltVals, dhduValus, d2hdu2Valus] = problem.EvaluateHamiltonianAndItsFirstTwoDerivatives(solutionDictionary, tArray, hamiltonian, {problem.ControlVariables[0]: controlSolved}, {baseProblem.TimeFinalSymbol: tfOrg})
plt.title("Hamlitonion and its derivatives")
plt.plot(tArray/86400, hamltVals, label="Hamiltonian")
plt.plot(tArray/86400, dhduValus, label=r'$\frac{dH}{du}$')
plt.plot(tArray/86400, d2hdu2Valus, label=r'$\frac{d^2H}{du^2}$')

plt.tight_layout()
plt.grid(alpha=0.5)
plt.legend(framealpha=1, shadow=True)
plt.show()   

xyz = np.zeros((len(tArray), 3))
for i in range(0, len(unscaledResults[baseProblem.StateVariables[0]])) :
    r = unscaledResults[baseProblem.StateVariables[0]][i]
    theta = unscaledResults[baseProblem.StateVariables[3]][i]
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    xyz[i,0] = x
    xyz[i,1] = y
    xyz[i,2] = 0

import plotly.express as px
from pandas import DataFrame
df = DataFrame(xyz)

x = np.array(xyz[:,0])
y = np.array(xyz[:,1])
z = np.array(xyz[:,2])
df = DataFrame({"x": x, "y":y, "z":z})
fig = px.line_3d(df, x="x", y="y", z="z")
fig.show()
