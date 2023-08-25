#%%
import __init__ #type: ignore
from IPython.display import display
from scipy.integrate import solve_ivp #type: ignore
import matplotlib.pyplot as plt#type: ignore
import numpy as np
import sympy as sy
import plotly.express as px#type: ignore
from pandas import DataFrame #type: ignore
import math
from scipy.optimize import fsolve#type: ignore
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from pyeq2orb.Numerical import ScipyCallbackCreators
from pyeq2orb.DemosAndPrototypes.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions

import sympy as sy
from typing import List, Dict, Callable, Optional, Any
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.Utilities.Typing import SymbolOrNumber


def plotSolution(helper, solution):

    xyz = np.zeros((len(tArray), 3))
    for i in range(0, len(solution[helper.NonTimeLambdifyArguments[0]])) :
        r = solution[helper.NonTimeLambdifyArguments[0]][i]
        theta = solution[helper.NonTimeLambdifyArguments[3]][i]
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        xyz[i,0] = x
        xyz[i,1] = y
        xyz[i,2] = 0


    df = DataFrame(xyz)

    xf = np.array(xyz[:,0])
    yf = np.array(xyz[:,1])
    zf = np.array(xyz[:,2])
    df = DataFrame({"x": xf, "y":yf, "z":zf})
    fig = px.line_3d(df, x="x", y="y", z="z")
    fig.show()

from pyeq2orb.Utilities.SolutionDictionaryFunctions import GetValueFromStateDictionaryAtIndex
import scipyPaperPrinter as jh#type: ignore
from typing import cast
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
tArray = np.linspace(0.0, tfOrg, 1200)
#if scaleTime:
#    tfVal = 1.0
#    tArray = np.linspace(0.0, 1.0, 1200)
    

# your choice of the nu vector here controls which transversality condition we use
nus = [sy.Symbol('B_{u_f}'), sy.Symbol('B_{v_f}')]
#nus = []

mus = sy.Symbol(r'\mu', real=True, positive=True)
thrusts = sy.Symbol('T', real=True, positive=True)
m0s = sy.Symbol('m_0', real=True, positive=True)

gs = sy.Symbol('g', real=True, positive=True)

ispS = sy.Symbol('I_{sp}', real=True, positive=True)
ts = sy.Symbol('t', real=True)
t0s = sy.Symbol('t_0', real=True)
tfs = sy.Symbol('t_f', real=True, positive=True)

rs = sy.Function('r', real=True, positive=True)(ts)
us = sy.Function('u', real=True, nonnegative=True)(ts)
vs = sy.Function('v', real=True, nonnegative=True)(ts)
lonS =  sy.Function(r'\theta', real=True, nonnegative=True)(ts)

alps = sy.Function(r'\alpha', real=True)(ts)

bc1 = us.subs(ts, tfs)
bc2 = vs.subs(ts, tfs)-sy.sqrt(mu/rs.subs(ts, tfs))

terminalCost = rs.subs(ts, tfs) # maximization problem



mFlowRate = -1*thrusts/(ispS*gs)
mEq = m0s+ts*mFlowRate

rEom = us
uEom = vs*vs/rs - mus/(rs*rs) + thrusts*sy.sin(alps)/mEq
vEom = -vs*us/rs + thrusts*sy.cos(alps)/mEq
lonEom = vs/rs


rEquation = sy.Eq(rs.diff(ts), rEom)
uEquation = sy.Eq(us.diff(ts), uEom)
vEquation = sy.Eq(vs.diff(ts), vEom)
lonEquation = sy.Eq(lonS.diff(ts), lonEom)



helper = OdeLambdifyHelperWithBoundaryConditions(ts, t0s, tfs, [rEquation, uEquation, vEquation, lonEquation], [bc1, bc2], [], {gs:g, ispS:isp, m0s: m0, thrusts:thrust, mus:mu})

lmds = SymbolicProblem.CreateCoVector(helper.NonTimeLambdifyArguments, None, ts)
hamlt = SymbolicProblem.CreateHamiltonianStatic(helper.NonTimeLambdifyArguments, ts, helper.GetExpressionToLambdifyInMatrixForm(), 0, lmds)
lambdaEoms = SymbolicProblem.CreateLambdaDotEquationsStatic(hamlt, ts, helper.NonTimeArgumentsArgumentsInMatrixForm(), lmds)
helper.AddMoreEquationsOfMotion(lambdaEoms)
dHdu = SymbolicProblem.CreateHamiltonianControlExpressionsStatic(hamlt, alps)
controlSolved = sy.solve(dHdu, alps)[0] 
helper.SubstitutionDictionary[alps] =  controlSolved

#TODO: Get the transversality conditions more generally
bc3 = 1-lmds[0].subs(ts, tfs)+0.5*lmds[2].subs(ts, tfs)*sy.sqrt(1/(rs.subs(ts, tfs)**3))
bc4 = lmds[3].subs(ts, tfs)
newBcs = [bc3, bc4]
helper.BoundaryConditionExpressions.extend(newBcs)
helper.SymbolsToSolveForWithBoundaryConditions.extend(SymbolicProblem.SafeSubs(lmds, {ts: t0s}))

del helper.BoundaryConditionExpressions[-1]
del helper.NonTimeLambdifyArguments[-1]
del helper.EquationsOfMotion[-1]
del helper.ExpressionsToLambdify[-1]
del helper.SymbolsToSolveForWithBoundaryConditions[-1]
helper.SubstitutionDictionary[lmds[3]] =0
helper.SubstitutionDictionary[lmds[3].subs(ts, tfs)]=0
helper.SubstitutionDictionary[lmds[3].subs(ts, t0s)]=0


ipvCallback = helper.CreateSimpleCallbackForSolveIvp()
def realIpvCallback(initialState) :
    solution = solve_ivp(ipvCallback, [tArray[0], tArray[-1]], initialState, t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
    solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, solution)
    return solution





initialaFSolveState = [0.0011569091762708, 0.00010000000130634948, 1.0]

justBcCb = helper.CreateCallbackForBoundaryConditionsWithFullState()
display(justBcCb[1](0, 6000, 7000, 8000, 0, *initialaFSolveState, 5000000, 42164000, -2000, -3000, 17, 10000, 20000, 30000))

solverCb = helper.createCallbackToSolveForBoundaryConditions(realIpvCallback, tArray, [r0, u0, v0, lon0, *initialaFSolveState])

#display(solverCb([1.0, 0.001, 0.001, 0.0]))
#display(helper.GetExpressionToLambdifyInMatrixForm())
#print(ipvCallback(0, [r0, u0, v0, lon0, 1.0, 0.001, 0.001, 0.0]))
solution = solve_ivp(ipvCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *initialaFSolveState], t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, solution)
plotSolution(helper, solutionDictionary)

fSolveSol = fsolve(solverCb, initialaFSolveState, epsfcn=0.000001, full_output=True)
display(fSolveSol)
fSolveSolution = solve_ivp(ipvCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *fSolveSol[0]], t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
fSolveSolutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, fSolveSolution)


plotSolution(helper, fSolveSolutionDictionary)

#%%
# these are options to switch to try different things
scaleElements = True
scaleTime = scaleElements and True

baseProblem = ContinuousThrustCircularOrbitTransferProblem()
initialStateValues = baseProblem.CreateVariablesAtTime0(baseProblem.StateVariables)
problem = cast(SymbolicProblem, baseProblem)

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

# you are in control of the order of integration variables and what equations of motion get evaluated, start updating the problem
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

def safeSubs(exprs, toBeSubbed):
    tbr = []
    for eom in exprs :
        if hasattr(eom, "subs"):
            tbr.append(eom.subs(toBeSubbed))
        else :
            tbr.append(eom)    
    return tbr

class OdeHelper :
    lambdifyStateFlattenOption = "flatten"
    lambdifyStateGroupedAllOption = "group"
    lambdifyStateGroupedAllButParametersOption = "groupFlattenParameters"

    lambdifyStateOrderOptionTimeFirst = "Time,StateVariables,MissingInitialValues,Parameters"
    lambdifyStateOrderOptionTimeMiddle = "StateVariables,Time,MissingInitialValues,Parameters"
    def __init__(self, t) :
        self.equationsOfMotion = []
        self.initialSymbols = []
        self.stateFunctionSymbols = []
        self.t = t
        self.constants = {}
        self.lambdifyParameterSymbols = []

    def setStateElement(self, sympyFunctionSymbol, symbolicEom, initialSymbol) :
        self.equationsOfMotion.append(symbolicEom)
        self.stateFunctionSymbols.append(sympyFunctionSymbol)
        self.initialSymbols.append(initialSymbol)

    def makeStateForLambdifiedFunction(self, groupOrFlatten=lambdifyStateGroupedAllButParametersOption, orderOption=lambdifyStateOrderOptionTimeFirst):
        arrayForLmd = []
        if orderOption == OdeHelper.lambdifyStateOrderOptionTimeFirst :
            arrayForLmd.append(self.t)
        stateArray = []    
        for svf in self.stateFunctionSymbols :
            stateArray.append(svf)
        if groupOrFlatten != OdeHelper.lambdifyStateFlattenOption :
            arrayForLmd.append(stateArray)    
        else :
            arrayForLmd.extend(stateArray)
        if orderOption == OdeHelper.lambdifyStateOrderOptionTimeMiddle :
            arrayForLmd.append(self.t)

        if len(self.lambdifyParameterSymbols) != 0 :
            if groupOrFlatten == OdeHelper.lambdifyStateGroupedAllButParametersOption or groupOrFlatten == OdeHelper.lambdifyStateFlattenOption:
                arrayForLmd.extend(self.lambdifyParameterSymbols)
            elif groupOrFlatten == OdeHelper.lambdifyStateGroupedAllOption :
                arrayForLmd.append(self.lambdifyParameterSymbols)
        return arrayForLmd

    def _createParameterOptionalWrapperOfLambdifyCallback(self, baseLambdifyCallback) :
        def callbackWrapper(a, b, *args) :
            if len(self.lambdifyParameterSymbols) == 0 :
                return baseLambdifyCallback(a, b)
            else :
                return baseLambdifyCallback(a, b, *args)
        return callbackWrapper

    def createLambdifiedCallback(self, groupOrFlatten=lambdifyStateGroupedAllButParametersOption, orderOption=lambdifyStateOrderOptionTimeFirst) :
        arrayForLmd=self.makeStateForLambdifiedFunction(groupOrFlatten, orderOption)
        subbedEom = safeSubs(self.equationsOfMotion, self.constants)
        baseLambdifyCallback = sy.lambdify(arrayForLmd, subbedEom, 'numpy')
        return self._createParameterOptionalWrapperOfLambdifyCallback(baseLambdifyCallback)

thisOdeHelper = OdeHelper(problem.TimeSymbol)
for key, value in problem.EquationsOfMotion.items() :
    thisOdeHelper.setStateElement(key, value, key.subs(problem.TimeSymbol, problem.TimeInitialSymbol) )

if scaleTime:
    thisOdeHelper.lambdifyParameterSymbols.append(baseProblem.TimeFinalSymbol)

if len(nus) != 0:
    thisOdeHelper.lambdifyParameterSymbols.append(problem.CostateSymbols[1])
    thisOdeHelper.lambdifyParameterSymbols.append(problem.CostateSymbols[2])

thisOdeHelper.constants = problem.SubstitutionDictionary
display(thisOdeHelper.makeStateForLambdifiedFunction())
odeIntEomCallback = thisOdeHelper.createLambdifiedCallback()

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
    print("solving ivp for final adjoined variable guess")
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

[hamiltonVals, dhduValues, d2hdu2Values] = problem.EvaluateHamiltonianAndItsFirstTwoDerivatives(solutionDictionary, tArray, hamiltonian, {problem.ControlVariables[0]: controlSolved}, {baseProblem.TimeFinalSymbol: tfOrg})
plt.title("Hamiltonian and its derivatives")
plt.plot(tArray/86400, hamiltonVals, label="Hamiltonian")
plt.plot(tArray/86400, dhduValues, label=r'$\frac{dH}{du}$')
plt.plot(tArray/86400, d2hdu2Values, label=r'$\frac{d^2H}{du^2}$')

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


df = DataFrame(xyz)

xf = np.array(xyz[:,0])
yf = np.array(xyz[:,1])
zf = np.array(xyz[:,2])
df = DataFrame({"x": xf, "y":yf, "z":zf})
fig = px.line_3d(df, x="x", y="y", z="z")
fig.show()
