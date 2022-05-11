#%%
import sys
sys.path.append("..") # treating this as a jupyter-like cell requires adding one directory up
sys.path.append("../PythonOptimizationWithNlp") # and this line is needed for running like a normal python script
# these two appends do not conflict with eachother

from IPython.display import display
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import math
from scipy.optimize import fsolve
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
from PythonOptimizationWithNlp.ScaledSymbolicProblem import ScaledSymbolicProblem
from PythonOptimizationWithNlp.Problems.ContinuousThrustCircularOrbitTransferProblem import PlanerLeoToGeoProblem
from PythonOptimizationWithNlp.Numerical import ScipyCallbackCreators
import JupyterHelper as jh

g = 9.80665
mu = 3.986004418e14  
thrust = 20.0
isp = 6000.0
m0 = 1500.0

r0 = 6678000.0
u0 = 0.0
v0 = sy.sqrt(mu/r0) # circular
lon0 = 0.0
tfVal  = 3600*3.97152*24 
tArray = np.linspace(0, tfVal, 1200)

scale = True
scaleTime = scale and False
# your choice of the nu vector here controls which transversality condition we use
#nus = [sy.Symbol('B_{u_f}'), sy.Symbol('B_{v_f}')]
nus = []

baseProblem = PlanerLeoToGeoProblem()
problem = baseProblem
if scale :
    newSvs = [sy.Function(r'\bar{r}')(baseProblem.TimeSymbol), sy.Function(r'\bar{u}')(baseProblem.TimeSymbol), sy.Function(r'\bar{v}')(baseProblem.TimeSymbol), sy.Function(r'\bar{lon}')(baseProblem.TimeSymbol)]
    problem = ScaledSymbolicProblem(baseProblem, newSvs, {baseProblem.StateVariables[0]: baseProblem.StateVariables[0].subs(baseProblem.TimeSymbol, baseProblem.TimeInitialSymbol), 
                                                          baseProblem.StateVariables[1]: baseProblem.StateVariables[2].subs(baseProblem.TimeSymbol, baseProblem.TimeInitialSymbol), 
                                                          baseProblem.StateVariables[2]: baseProblem.StateVariables[2].subs(baseProblem.TimeSymbol, baseProblem.TimeInitialSymbol), 
                                                          baseProblem.StateVariables[3]: 1.0} , scaleTime)


baseProblem.RegisterConstantValue(baseProblem.Isp, isp)
baseProblem.RegisterConstantValue(baseProblem.MassInitial, m0)
baseProblem.RegisterConstantValue(baseProblem.Gravity, g)
baseProblem.RegisterConstantValue(baseProblem.Mu, mu)
baseProblem.RegisterConstantValue(baseProblem.Thrust, thrust)

if scaleTime:
    tfOrg = tfVal
    tfVal = 1.0
jh.t = problem._timeSymbol
constantsSubsDict = problem.SubstitutionDictionary
initialStateValues = baseProblem.CreateVariablesAtTime0()
initialScaledStateValues = problem.CreateVariablesAtTime0()

constantsSubsDict[initialStateValues[0]] = r0
constantsSubsDict[initialStateValues[1]] = u0
constantsSubsDict[initialStateValues[2]] = v0
constantsSubsDict[initialStateValues[3]] = lon0
# constantsSubsDict[initialStateValues[0].subs(problem.TimeInitialSymbol, 0.0)] = r0
# constantsSubsDict[initialStateValues[1].subs(problem.TimeInitialSymbol, 0.0)] = u0
# constantsSubsDict[initialStateValues[2].subs(problem.TimeInitialSymbol, 0.0)] = v0
# constantsSubsDict[initialStateValues[3].subs(problem.TimeInitialSymbol, 0.0)] = lon0
constantsSubsDict[initialScaledStateValues[0]] = r0
constantsSubsDict[initialScaledStateValues[1]] = u0
constantsSubsDict[initialScaledStateValues[2]] = v0
constantsSubsDict[initialScaledStateValues[3]] = lon0
if scale :
    constantsSubsDict[initialScaledStateValues[0]] = r0/r0
    constantsSubsDict[initialScaledStateValues[1]] = u0/v0
    constantsSubsDict[initialScaledStateValues[2]] = v0/v0
    constantsSubsDict[initialScaledStateValues[3]] = lon0/1.0

    constantsSubsDict[initialScaledStateValues[0].subs(problem.TimeInitialSymbol, 0.0)] = r0/r0
    constantsSubsDict[initialScaledStateValues[1].subs(problem.TimeInitialSymbol, 0.0)] = u0/v0
    constantsSubsDict[initialScaledStateValues[2].subs(problem.TimeInitialSymbol, 0.0)] = v0/v0
    constantsSubsDict[initialScaledStateValues[3].subs(problem.TimeInitialSymbol, 0.0)] = 0.0

    constantsSubsDict[baseProblem.StateVariables[0].subs(baseProblem.TimeSymbol,problem.TimeInitialSymbol)] = r0
    constantsSubsDict[baseProblem.StateVariables[1].subs(baseProblem.TimeSymbol, problem.TimeInitialSymbol)] = u0
    constantsSubsDict[baseProblem.StateVariables[2].subs(baseProblem.TimeSymbol, problem.TimeInitialSymbol)] = v0
    constantsSubsDict[baseProblem.StateVariables[3].subs(baseProblem.TimeSymbol, problem.TimeInitialSymbol)] = lon0

    r0= r0/r0
    u0=u0/v0
    v0=v0/v0
    lon0=lon0/1.0
    

# This is cheating, I know from many previous runs that this is the time needed to go from LEO to GEO.
# However, below works well wrapped in another fsolve to control the final time for a desired radius.

# this next block does most of the problem
lambdas = problem.CreateCoVector(problem.StateVariables, r'\lambda', problem.TimeSymbol)
hamiltonian = problem.CreateHamiltonian(lambdas)
jh.showEquation("H", hamiltonian)
dHdu = problem.CreateHamiltonianControlExpressions(hamiltonian).doit()[0]
d2Hdu2 = problem.CreateHamiltonianControlExpressions(dHdu).doit()[0]
controlSolved = sy.solve(dHdu, problem.ControlVariables[0])[0]
jh.showEquation(baseProblem.ControlVariables[0], controlSolved)

finalEquationsOfMotion = {}
for x in problem.StateVariables :
    finalEquationsOfMotion[x] = problem.EquationsOfMotion[x].subs(problem.ControlVariables[0], controlSolved).trigsimp(deep=True).simplify()
    jh.showEquation(sy.diff(x, problem.TimeSymbol), finalEquationsOfMotion[x])

lambdaDotExpressions = problem.CreateLambdaDotCondition(hamiltonian).doit()
for i in range(0, len(lambdas)) :
    finalEquationsOfMotion[lambdas[i]] = lambdaDotExpressions[i].subs(problem.ControlVariables[0], controlSolved).simplify()
    jh.showEquation(sy.diff(lambdas[i], problem.TimeSymbol), finalEquationsOfMotion[lambdas[i]])

lmdsF = problem.SafeSubs(lambdas, {problem.TimeSymbol: problem.TimeFinalSymbol})
if len(nus) != 0:
    transversalityCondition = problem.TransversalityConditionsByAugmentation(lmdsF, nus)
else:
    transversalityCondition = problem.CreateDifferentialTransversalityConditions(hamiltonian, lmdsF, sy.Symbol(r'dt_f'))

for xv in transversalityCondition :
    jh.showEquation(xv, 0)

#TODO: Throw if wrong number, expect 2?
lmdsAtT0 = problem.CreateVariablesAtTime0(lambdas)    
constantsSubsDict[lmdsAtT0[3]] = 0.0    

# creating the initial values is unique to each problem, it is luck that 
# my intuition pays off and we find a solution later
# We want initial alpha to be 0 (or really close to it) per intuition
# We can choose lmdv and solve for lmdu.  Start with lmdv to be 1
# solve for lmdu with those assumptions

constantsSubsDict[lmdsAtT0[2]] = 1.0 

initialLmdGuesses = []
controlAtT0 = problem.CreateVariablesAtTime0(controlSolved)
sinOfControlAtT0 = sy.sin(controlAtT0).trigsimp(deep=True).expand().simplify()
alphEq = sinOfControlAtT0.subs(lmdsAtT0[2], constantsSubsDict[lmdsAtT0[2]])
ans1 = sy.solveset(sy.Eq(0.00,alphEq), lmdsAtT0[1])
# doesn't like 0, so let's make it small
ans1 = sy.solveset(sy.Eq(0.0001,alphEq), lmdsAtT0[1])

for thing in ans1 :
    ansForLmdu = thing
constantsSubsDict[lmdsAtT0[1]] = float(ansForLmdu)

# if we assume that we always want to keep alpha small (0), we can solve dlmd_u/dt=0 for lmdr_0
lmdUDotAtT0 = problem.CreateVariablesAtTime0(finalEquationsOfMotion[lambdas[1]])
lmdUDotAtT0 = lmdUDotAtT0.subs(constantsSubsDict)
inter=sy.solve(sy.Eq(lmdUDotAtT0, 0), lmdsAtT0[0])
lambdaR0Value = inter[0].subs(constantsSubsDict) # we know there is just 1
constantsSubsDict[lmdsAtT0[0]] = float(lambdaR0Value) # later on, arrays will care that this MUST be a float

for lmdAtT0 in lmdsAtT0 :
    initialLmdGuesses.append(constantsSubsDict[lmdAtT0])
    del constantsSubsDict[lmdAtT0]


# lambda_lon is always 0, so do that cleanup
del finalEquationsOfMotion[lambdas[3]]
transversalityCondition.pop()
initialLmdGuesses.pop()
lmdTheta = lambdas.pop()
lmdsF.pop()
constantsSubsDict[lmdTheta]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeFinalSymbol)]=0
constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeInitialSymbol)]=0


# start the conversion to a numerical answer
integrationStateVariableArray = []
integrationStateVariableArray.extend(problem.StateVariables)
integrationStateVariableArray.extend(lambdas)
if scaleTime :
    constantsSubsDict[problem.TimeFinalSymbolOriginal] = tfOrg
odeIntEomCallback = ScipyCallbackCreators.CreateSimpleCallbackForOdeInt(problem.TimeSymbol, integrationStateVariableArray, finalEquationsOfMotion, constantsSubsDict)

stateForBoundaryConditions = []
stateForBoundaryConditions.extend(SymbolicProblem.SafeSubs(integrationStateVariableArray, {problem.TimeSymbol: problem.TimeInitialSymbol}))
stateForBoundaryConditions.extend(SymbolicProblem.SafeSubs(integrationStateVariableArray, {problem.TimeSymbol: problem.TimeFinalSymbol}))
if len(nus) > 0 :
    stateForBoundaryConditions.extend(nus)

allBcAndTransConditions = []
allBcAndTransConditions.extend(transversalityCondition)
allBcAndTransConditions.extend(problem.BoundaryConditions)
print("here")
for thing in allBcAndTransConditions :
    display(thing)
    display(thing.subs(constantsSubsDict))
boundaryConditionEvaluationCallbacks = ScipyCallbackCreators.createBoundaryConditionCallback(stateForBoundaryConditions, allBcAndTransConditions, constantsSubsDict)
if scaleTime :
    constantsSubsDict[problem.TimeFinalSymbolOriginal] = tfOrg
# run a test solution to get a better guess for the final nu values, this is a good technique, but 
# it is still a custom-to-this-problem piece of code because it is still initial-guess work
if len(nus) > 0 :
    initialLmdGuesses.append(initialLmdGuesses[1])
    initialLmdGuesses.append(initialLmdGuesses[2])
    testSolution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, *initialLmdGuesses[0:3]], tArray)
    initialLmdGuesses[3] = testSolution[:,5][-1]
    initialLmdGuesses[4] = testSolution[:,6][-1]

print(initialLmdGuesses)
def createCallbackForFSolve(initialEomState, timeArray, eomCallback, bcCallback, includeFSolveStateValuesInEomEquationBeforeThisIndex) :
    # this is a tricky thing to generalize.  This is combining several different things together:
    #  - evaluating the equations of motion
    #  - creating the values to be plugged into the boundary condition (xvers) equations
    #  - Evaluating the BC's/Xvers values
    # Combining those 3 things, although are common to many problems, don't really generalize well.  It is 
    # the meshing together of 3 things with the intent of it being run through a fourth (fsolve), and that 
    # kind of combining of several different things is something that ought to be brought together at the script
    # level.
    # BUT, I will keep thinking about how best to generalize this.  I keep thinking this ought to be some kind of 
    # helper function, but whenever I try it, I always end up with some overspecialized "helper" function, or 
    # hiding how it comes together (I still want the end user to call fsolve)
    def callbackForFsolve(costateAndCostateVariableGuesses) :
        z0 = []
        z0.extend(initialEomState)
        z0.extend(costateAndCostateVariableGuesses[0:includeFSolveStateValuesInEomEquationBeforeThisIndex])
        
        ans = odeint(eomCallback, z0, timeArray)
        finalState = []
        # add initial state
        finalState.extend(ans[0]) # the fact that this function needs to know how to create this overall state for the BC callback...
        # add final state
        finalState.extend(ans[-1]) 
        # add values in fSolve state after what is already there
        finalState.extend(costateAndCostateVariableGuesses[includeFSolveStateValuesInEomEquationBeforeThisIndex:])
        finalAnswers = []
        finalAnswers.extend(bcCallback(*finalState))
    
        return finalAnswers    
    return callbackForFsolve


fSolveCallback = createCallbackForFSolve([r0, u0, v0, lon0], tArray, odeIntEomCallback, boundaryConditionEvaluationCallbacks, len(lambdas))
fSolveSol = fsolve(fSolveCallback, initialLmdGuesses, epsfcn=0.00001, full_output=True) # just to speed things up and see how the initial one works
print(fSolveSol)
# final run with answer
solution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, *fSolveSol[0][0:3]], tArray)
#solution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, 26.0, 1.0, 27.0], tArray)
asDict = ScipyCallbackCreators.ConvertOdeIntResultsToDictionary(integrationStateVariableArray, solution)
if scale :
    asDict = problem.DescaleResults(asDict, constantsSubsDict)

baseProblem.PlotSolution(tArray, asDict, "Test")
jh.showEquation(baseProblem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol), asDict[baseProblem.StateVariables[0]][-1])
jh.showEquation(baseProblem.StateVariables[1].subs(problem.TimeSymbol, problem.TimeFinalSymbol), asDict[baseProblem.StateVariables[1]][-1])
jh.showEquation(baseProblem.StateVariables[2].subs(problem.TimeSymbol, problem.TimeFinalSymbol), asDict[baseProblem.StateVariables[2]][-1])
jh.showEquation(baseProblem.StateVariables[3].subs(problem.TimeSymbol, problem.TimeFinalSymbol), (asDict[baseProblem.StateVariables[3]][-1]%(2*math.pi))*180.0/(2*math.pi))

problem.plotHamiltonianProblemsFromSomeSetOfResults(lambdas, solution, tArray, hamiltonian, controlSolved)

