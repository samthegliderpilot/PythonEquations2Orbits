#%%
import sys
from typing import OrderedDict
from attr import asdict
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sqlalchemy import true
sys.path.append("..") # treating this as a jupyter-like cell requires adding one directory up
sys.path.append("../PythonOptimizationWithNlp") # and this line is needed for running like a normal python script
# these two appends do not conflict with eachother

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import math
from scipy.optimize import fsolve
from scipy.integrate import odeint

from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
from PythonOptimizationWithNlp.Problems.ScaledSymbolicProblem import ScaledSymbolicProblem
from PythonOptimizationWithNlp.Problems.PlanerLeoToGeoProblem import PlanerLeoToGeoProblem
from PythonOptimizationWithNlp.Symbolics.Vectors import Vector

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

scale = True
# your choice of the nu vector here controls which transversality condition we use
#nus = [sy.Symbol('B_{uf}'), sy.Symbol('B_{vf}')]
nus = []

baseProblem = PlanerLeoToGeoProblem()
problem = baseProblem
if scale :
    newSvs = [sy.Function(r'\bar{r}')(baseProblem.TimeSymbol), sy.Function(r'\bar{u}')(baseProblem.TimeSymbol), sy.Function(r'\bar{v}')(baseProblem.TimeSymbol), sy.Function(r'\bar{lon}')(baseProblem.TimeSymbol)]
    problem = ScaledSymbolicProblem(baseProblem, newSvs, {baseProblem.StateVariables[0]: baseProblem.StateVariables[0].subs(baseProblem.TimeSymbol, baseProblem.Time0Symbol), 
                                                          baseProblem.StateVariables[1]: baseProblem.StateVariables[2].subs(baseProblem.TimeSymbol, baseProblem.Time0Symbol), 
                                                          baseProblem.StateVariables[2]: baseProblem.StateVariables[2].subs(baseProblem.TimeSymbol, baseProblem.Time0Symbol), 
                                                          baseProblem.StateVariables[3]: 1.0} , False)

jh.t = problem.Ts
constantsSubsDict = {}
initialStateValues = baseProblem.CreateVariablesAtTime0()
initialScaledStateValues = problem.CreateVariablesAtTime0()

constantsSubsDict[initialStateValues[0]] = r0
constantsSubsDict[initialStateValues[1]] = u0
constantsSubsDict[initialStateValues[2]] = v0
constantsSubsDict[initialStateValues[3]] = lon0
constantsSubsDict[initialScaledStateValues[0]] = r0
constantsSubsDict[initialScaledStateValues[1]] = u0
constantsSubsDict[initialScaledStateValues[2]] = v0
constantsSubsDict[initialScaledStateValues[3]] = 0.0
if scale :
    constantsSubsDict[initialScaledStateValues[0]] = r0/r0
    constantsSubsDict[initialScaledStateValues[1]] = u0/v0
    constantsSubsDict[initialScaledStateValues[2]] = v0/v0
    constantsSubsDict[initialScaledStateValues[3]] = 0.0
tfVal  = 3600*3.97152*24 
# This is cheating, I know from many previous runs that this is the time needed to go from LEO to GEO.
# However, below works well wrapped in another fsolve to control the final time for a desired radius.


jh.printMarkdown('### Indirect Method')
jh.printMarkdown("# Two Dimensional LEO to GEO Transfer with Time Varying Mass")
jh.printMarkdown('The two dimensional LEO to GEO transfer has been a very useful problem to study in optimal control.  Having found solutions using both direct and indirect methods, it is complicated enough to learn insights to the various solving methods, while at the same time being tractable enough to not get stuck.  This script will show several ways to solve the problem as well as comparing the different solutions.')

jh.printMarkdown('To starts with, we state the problem.  There are many different ways to form the problem, but for a given final time, we will maximize the final radius:')
jh.showEquation(problem.CreateCostFunctionAsEquation())

jh.printMarkdown('Subject to the equations of motion:')
for eq in problem.CreateEquationOfMotionsAsEquations() :
    display(eq.rhs)
jh.printMarkdown(r'Where $\alpha$ is the control value to derive.')
jh.printMarkdown('Additionally, there is a full set of initial conditions for the four state variables, and all of the final conditions will be specified except for the final longitude.')
for bc in problem.FinalBoundaryConditions :
    display(bc)

# this next block does most of the problem
lambdas = problem.CreateCoVector(problem.StateVariables, r'\lambda', problem.TimeSymbol)
hamiltonian = problem.CreateHamiltonian(lambdas)
print("hamlt")
jh.showEquation("H", hamiltonian)
dHdu = problem.CreateHamiltonianControlExpressions(hamiltonian).doit()[0]
print("dHdu")
display(dHdu)

d2Hdu2 = problem.CreateHamiltonianControlExpressions(dHdu).doit()[0]
print("d2Hdu2")
display(d2Hdu2)

controlSolved = sy.solve(dHdu, problem.ControlVariables[0])[0]
print("solved control")
display(controlSolved)

finalEquationsOfMotion = []
for x in problem.StateVariables :
    finalEquationsOfMotion.append(problem.EquationsOfMotion[x].subs(problem.ControlVariables[0], controlSolved).trigsimp(deep=True))


lambdaDotExpressions = problem.CreateLambdaDotCondition(hamiltonian).doit()
for i in range(0, len(lambdas)) :
    finalEquationsOfMotion.append(lambdaDotExpressions[i].subs(problem.ControlVariables[0], controlSolved))

def otherWayToDoTransversalityCondition(problem : SymbolicProblem, lambdas, nus) :
    termFunc = problem.TerminalCost + (Vector.fromArray(nus).transpose()*Vector.fromArray(problem.FinalBoundaryConditions))[0,0]
    finalConditions = []
    i=0
    for x in problem.StateVariables :
        xf = x.subs(problem.TimeSymbol, problem.TimeFinalSymbol)
        cond = termFunc.diff(xf)
        finalConditions.append(lambdas[i]-cond)
        i=i+1

    return finalConditions
    

lmdsF = problem.SafeSubs(lambdas, {problem.TimeSymbol: problem.TimeFinalSymbol})

if len(nus) != 0:
    transversalityCondition = otherWayToDoTransversalityCondition(problem, lmdsF, nus)
else:
    transversalityCondition = problem.CreateDifferentialTransversalityConditions(hamiltonian, lambdas, sy.Symbol(r'dt_f'))

#TODO: Throw if wrong number, expect 2
print("xvers cond")
for xc in transversalityCondition :
    display(xc)
lmdsAtT0 = problem.CreateVariablesAtTime0(lambdas)    
constantsSubsDict[lmdsAtT0[3]] = 0.0    

# creating the initial values is still a manual process, it is luck that 
# my intuition pays off and we find a solution later
# want initial alpha to be 0 (or really close to it) per intuition
# we can choose lmdv and solve for lmdu.  Start with lmdv to be 1
# solve for lmdu with those assumptions

#constantsSubsDict[lmdsAtT0[0]] = 0.0
#constantsSubsDict[lmdsAtT0[1]] = 0.00
constantsSubsDict[lmdsAtT0[2]] = 1.0 

initialLmdGuesses = []

controlAtTo = problem.CreateVariablesAtTime0(controlSolved)

jh.showEquation(controlAtTo)
print("wrap in cos")
controlAtTo = sy.sin(controlAtTo).trigsimp(deep=true).expand().simplify()
print("cos done")
alphEq = controlAtTo.subs(lmdsAtT0[2], constantsSubsDict[lmdsAtT0[2]])
jh.showEquation(alphEq)
ans1 = sy.solveset(sy.Eq(0.00,alphEq), lmdsAtT0[1])
# doesn't like 0, so let's make it small
ans1 = sy.solveset(sy.Eq(0.02,alphEq), lmdsAtT0[1])
display(ans1)

for thing in ans1 :
    ansForLmdu = thing
constantsSubsDict[lmdsAtT0[1]] = float(ansForLmdu)

# if we assume that we always want to keep alpha small (0), we can solve dlmd_u/dt=0 for lmdr_0
lmdUDotAtT0 = problem.CreateVariablesAtTime0(finalEquationsOfMotion[5])
jh.showEquation(lmdUDotAtT0)
lmdUDotAtT0 = lmdUDotAtT0.subs(constantsSubsDict)
jh.showEquation(lmdUDotAtT0)
lambdaR0Value = sy.solve(sy.Eq(lmdUDotAtT0, 0), lmdsAtT0[0])[0].subs(constantsSubsDict) # we know there is just 1
jh.showEquation(lmdsAtT0[0], lambdaR0Value)
constantsSubsDict[lmdsAtT0[0]] = float(lambdaR0Value) # later on, arrays will care that this MUST be a float


for lmdAtT0 in lmdsAtT0 :
    initialLmdGuesses.append(constantsSubsDict[lmdAtT0])
    del constantsSubsDict[lmdAtT0]


# start the conversion to a numerical answer
finalEquationsOfMotion.pop()
for i in range(0, len(finalEquationsOfMotion)):
    finalEquationsOfMotion[i] = finalEquationsOfMotion[i].subs(lambdas[3], 0)
    finalEquationsOfMotion[i] = finalEquationsOfMotion[i].subs(lambdas[3].subs(problem.TimeSymbol, problem.Time0Symbol), 0)
    finalEquationsOfMotion[i] = finalEquationsOfMotion[i].subs(lambdas[3].subs(problem.TimeSymbol, problem.TimeFinalSymbol), 0)
transversalityCondition.pop()
initialLmdGuesses.pop()

# and add guesses for the adjoined constraints
for i in range(0, len(nus)) :
    initialLmdGuesses.append(initialLmdGuesses[i+1])


baseProblem.AppendConstantsToSubsDict(constantsSubsDict, mu, g, thrust, m0, isp)
stateForEom = [problem.TimeSymbol]
stateForEom.append(problem.StateVariables)
lmdTheta0 = lambdas.pop()
stateForEom.append(lambdas)
constantsSubsDict[lmdTheta0]=0
constantsSubsDict[lmdTheta0.subs(problem.TimeSymbol, problem.TimeFinalSymbol)]=0
constantsSubsDict[lmdTheta0.subs(problem.TimeSymbol, problem.Time0Symbol)]=0
eoms = []
for ep in finalEquationsOfMotion :
    eoms.append(ep.subs(constantsSubsDict))
integratableEoms = sy.lambdify(stateForEom, eoms)

stateForBoundaryConditions = []
for i in range(0, len(problem.StateVariables)) :
    stateForBoundaryConditions.append(problem.StateVariables[i].subs(problem.TimeSymbol, problem.Time0Symbol))
for i in range(0, len(lambdas)) :
    stateForBoundaryConditions.append(lambdas[i].subs(problem.TimeSymbol, problem.Time0Symbol))
for i in range(0, len(problem.StateVariables)) :
    stateForBoundaryConditions.append(problem.StateVariables[i].subs(problem.TimeSymbol, problem.TimeFinalSymbol))
for i in range(0, len(lambdas)) :
    stateForBoundaryConditions.append(lambdas[i].subs(problem.TimeSymbol, problem.TimeFinalSymbol))
for i in range(0, len(nus)) :
    stateForBoundaryConditions.append(nus[i])


evaluatableBoundaryConditions = []

print("xvers again")
for xVersBoundaryCondition in transversalityCondition:    
    display(xVersBoundaryCondition.subs(constantsSubsDict))
    evalTransCond = sy.lambdify(stateForBoundaryConditions, xVersBoundaryCondition.subs(constantsSubsDict))
    evaluatableBoundaryConditions.append(evalTransCond)

print("bcs")
count = 0
for bc in problem.FinalBoundaryConditions :
    display(bc.subs(constantsSubsDict))
    evalTransCond = sy.lambdify(stateForBoundaryConditions, bc.subs(constantsSubsDict))
    evaluatableBoundaryConditions.append(evalTransCond)

tArray = np.linspace(0, tfVal, 1200)

def integrateFromInitialValues(z0) :
    integratableCb = lambda z,t : integratableEoms(t, z[0:4], z[4:7])# this needs to be generalized
    return odeint(integratableCb, z0[0:7], tArray)

def createBoundaryConditionStateFromIntegrationResult(ans, lambdaGuesses) :
    finalState = []
    finalState.extend(ans[0])
    finalState.extend(ans[-1])
    
    return finalState

firstAns = None
def callbackForFsolve(lambdaGuesses) :
    global firstAns
    z0 = []
    for sv in initialScaledStateValues :
        z0.append(constantsSubsDict[sv])
    z0.extend(lambdaGuesses)
    
    ans = integrateFromInitialValues(z0)
    if firstAns is None :
        firstAns = ans
    finalState = createBoundaryConditionStateFromIntegrationResult(ans, lambdaGuesses)  
    for i in range(len(nus), 0, -1) :
        finalState.append(finalState[-1*i])   
    #finalState.append(lambdaGuesses[-1])
    finalAnswers = []
    for transCondition in evaluatableBoundaryConditions: 
        finalAnswers.append(transCondition(*finalState))
   
    return finalAnswers
#initialLmdGuesses = [-1*initialLmdGuesses[0], -1*initialLmdGuesses[1], -1*initialLmdGuesses[2]]#, 0]
#epsfcn is important, too large and the solution fails about
print(initialLmdGuesses)

fSolveSol = fsolve(callbackForFsolve, initialLmdGuesses, epsfcn=0.001, factor=1.0, full_output=True) # just to speed things up and see how the initial one works
print(fSolveSol)

finalInitialValues = []
for sv in initialScaledStateValues :
    finalInitialValues.append(constantsSubsDict[sv])
finalInitialValues.extend(fSolveSol[0])
# final run with answer
solution = integrateFromInitialValues(finalInitialValues)


#solution = integrateFromInitialValues([1.0, 0.0, 1.0, 0.0, 0.8952, -0.044999, 0.9223, -0.000735])
#solution = integrateFromInitialValues([1.0, 0.0, 1.0, 0.0, 0.768876, 0.0443323, 0.8905321, -0.045295])
#solution = firstAns

asDict = OrderedDict()
asDict[problem.StateVariables[0]] = solution[:,0]
asDict[problem.StateVariables[1]] = solution[:,1]
asDict[problem.StateVariables[2]] = solution[:,2]
asDict[problem.StateVariables[3]] = solution[:,3]
asDict[lambdas[0]] = solution[:,4]
asDict[lambdas[1]] = solution[:,5]
asDict[lambdas[2]] = solution[:,6]
if len(solution[:0]) > 7 :
    asDict[lambdas[3]] = solution[:,7]

unscaled = asDict
if scale :
    unscaled = problem.DescaleResults(asDict, constantsSubsDict)

baseProblem.PlotSolution(tArray, unscaled, "Test")
jh.showEquation(baseProblem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaled[baseProblem.StateVariables[0]][-1])
jh.showEquation(baseProblem.StateVariables[1].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaled[baseProblem.StateVariables[1]][-1])
jh.showEquation(baseProblem.StateVariables[2].subs(problem.TimeSymbol, problem.TimeFinalSymbol), unscaled[baseProblem.StateVariables[2]][-1])
jh.showEquation(baseProblem.StateVariables[3].subs(problem.TimeSymbol, problem.TimeFinalSymbol), (unscaled[baseProblem.StateVariables[3]][-1]%(2*math.pi))*180.0/(2*math.pi))


import matplotlib.pyplot as plt
stateForHaml = []
stateForHaml.extend(stateForEom)
stateForHaml.append(problem.TimeSymbol)
hamltEpx = sy.lambdify(stateForEom, hamiltonian.subs(problem.ControlVariables[0], controlSolved).trigsimp(deep=True).subs(constantsSubsDict).subs(lmdTheta0, 0))
hamltVals = hamltEpx(tArray, [solution[:,0],solution[:,1],solution[:,2],solution[:,3]],[solution[:,4],solution[:,5],solution[:,6]])

display(dHdu)
dhduExp = sy.lambdify(stateForEom, dHdu.subs(problem.ControlVariables[0], controlSolved).trigsimp(deep=True).subs(constantsSubsDict).subs(lmdTheta0, 0))
dhduValus = dhduExp(tArray, [solution[:,0],solution[:,1],solution[:,2],solution[:,3]],[solution[:,4],solution[:,5],solution[:,6]])
display(dhduValus)
if float(dhduValus) == 0 :
    dhduValus = np.zeros(len(tArray))


d2hdu2Exp = sy.lambdify(stateForEom, d2Hdu2.subs(problem.ControlVariables[0], controlSolved).trigsimp(deep=True).subs(constantsSubsDict).subs(lmdTheta0, 0))
d2hdu2Valus = d2hdu2Exp(tArray, [solution[:,0],solution[:,1],solution[:,2],solution[:,3]],[solution[:,4],solution[:,5],solution[:,6]])

plt.title("Hamlitonion")
plt.plot(tArray, hamltVals, label="Hamlt")
plt.plot(tArray, dhduValus, label="dH\du")
plt.plot(tArray, d2hdu2Valus, label="d2H\du2")

plt.tight_layout()
plt.grid(alpha=0.5)
plt.legend(framealpha=1, shadow=True)
plt.show()
jh.printMarkdown("Since t appears explicitly in the Hamiltonian (as the mass is not its own state variable), the hamiltonian will not be constant.")

