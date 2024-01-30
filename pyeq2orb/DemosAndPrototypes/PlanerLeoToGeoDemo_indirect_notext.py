#%%
#import sys
#sys.path.append('..\\')
#sys.path.append('..\..\\')
from IPython.display import display
from scipy.integrate import solve_ivp #type: ignore
import matplotlib.pyplot as plt#type: ignore
import numpy as np
import sympy as sy
import plotly.express as px#type: ignore
from pandas import DataFrame #type: ignore
import math
from scipy.optimize import fsolve#type: ignore
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem #type: ignore
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem #type: ignore
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem #type: ignore
from pyeq2orb.Numerical import ScipyCallbackCreators #type: ignore
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions #type: ignore

import sympy as sy
from typing import List, Dict, Callable, Optional, Any
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs

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
# and their symbols
gSy = sy.Symbol('g', real=True, positive=True)
muSy = sy.Symbol(r'\mu', real=True, positive=True)
thrustSy = sy.Symbol('T', real=True, positive=True)
ispSy = sy.Symbol('I_{sp}', real=True, positive=True)
m0Sy = sy.Symbol('m_0', real=True, positive=True)
# start populating the substitution dictionary

# I know from many previous runs that this is the time needed to go from LEO to GEO.
# However, below works well wrapped in another fsolve to control the final time for a desired radius.
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

# problem specific boundary conditions, =0
bc1 = uSy.subs(tSy, tfSy)
bc2 = vSy.subs(tSy, tfSy)-sy.sqrt(muSy/rSy.subs(tSy, tfSy))

terminalCost = rSy.subs(tSy, tfSy) # maximization problem
unintegratedPathCost = 0
# group things into the common pieces of data needed
eoms = [rEquationUnScaled, uEquationUnScaled, vEquationUnScaled, lonEquationUnScaled, mDotEquationUnscaled]
x = [rSy, uSy, vSy, lonSy]
x0=SafeSubs([rSy, uSy, vSy, lonSy], {tSy: t0Sy})
xf=SafeSubs([rSy, uSy, vSy, lonSy], {tSy: tfSy})
bcs = [bc1, bc2]
initialLambdaGuesses = [1, 0, 1, 0]
substitutionDictionary = {gSy:g, ispSy:isp, m0Sy: m0, thrustSy:thrust, muSy:mu}
scaleDictionary = {rSy:r0, uSy:3, vSy:3, lonSy:1, mSy:1}
initialValues = [r0, u0, v0, lon0, m0]
controlSymbols = [alpSy]




helper = OdeLambdifyHelperWithBoundaryConditions(tSy, t0Sy, tfSy, eoms, bcs, [],substitutionDictionary)

costateVariables = SymbolicProblem.CreateCoVector(x, None, tSy)
hamiltonian = SymbolicProblem.CreateHamiltonianStatic(x, tSy, helper.GetExpressionToLambdifyInMatrixForm(), 0, costateVariables)
lambdaEquationsOfMotion = SymbolicProblem.CreateLambdaDotEquationsStatic(hamiltonian, tSy, helper.NonTimeArgumentsArgumentsInMatrixForm(), costateVariables)

helper.AddMoreEquationsOfMotion(lambdaEquationsOfMotion)
optU = SymbolicProblem.CreateControlExpressionsFromHamiltonian(hamiltonian, controlSymbols)
for (k,v) in optU.items() :
    helper.SubstitutionDictionary[k] =  v

#TODO: Get the transversality conditions more generally
SymbolicProblem.TransversalityConditionInTheDifferentialForm(hamiltonian, sy.Symbol(r'dt_f'))
bc3 = 1-costateVariables[0].subs(tSy, tfSy)+0.5*costateVariables[2].subs(tSy, tfSy)*sy.sqrt(1/(rSy.subs(tSy, tfSy)**3))
bc4 = costateVariables[3].subs(tSy, tfSy)
newBcs = [bc3, bc4]
helper.BoundaryConditionExpressions.extend(newBcs)
helper.SymbolsToSolveForWithBoundaryConditions.extend(SafeSubs(costateVariables, {tSy: t0Sy}))

# From working the problem enough, we find that one of the costate variables is constant and 0. 
# As such, the related BC can be removed, one of the costates can be made constant, an EOM removed, etc..
# This needs to be improved, but for now, just do manually
del helper.BoundaryConditionExpressions[-1]
del helper.NonTimeLambdifyArguments[-1]
del helper.EquationsOfMotion[-1]
del helper.ExpressionsToLambdify[-1]
del helper.SymbolsToSolveForWithBoundaryConditions[-1]
helper.SubstitutionDictionary[costateVariables[3]] =0
helper.SubstitutionDictionary[costateVariables[3].subs(tSy, tfSy)]=0
helper.SubstitutionDictionary[costateVariables[3].subs(tSy, t0Sy)]=0


ipvCallback = helper.CreateSimpleCallbackForSolveIvp()
def realIpvCallback(initialState) :
    solution = solve_ivp(ipvCallback, [tArray[0], tArray[-1]], initialState, t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
    solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, solution)
    return solution




# In earlier iterations, I've found a way for this problem to get a not-horrible initial guess for the initial costates.  Not reproducing here 
# for now
initialFSolveState = [0.0011569091762708, 0.00010000000130634948, 1.0]

justBcCb = helper.CreateCallbackForBoundaryConditionsWithFullState()
display(justBcCb[1](0, 6000, 7000, 8000, 0, *initialFSolveState, 5000000, 42164000, -2000, -3000, 17, 10000, 20000, 30000))

solverCb = helper.createCallbackToSolveForBoundaryConditions(realIpvCallback, tArray, [r0, u0, v0, lon0, *initialFSolveState])

#display(solverCb([1.0, 0.001, 0.001, 0.0]))
#display(helper.GetExpressionToLambdifyInMatrixForm())
#print(ipvCallback(0, [r0, u0, v0, lon0, 1.0, 0.001, 0.001, 0.0]))
solution = solve_ivp(ipvCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *initialFSolveState], t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(helper.NonTimeLambdifyArguments, solution)
plotSolution(helper, solutionDictionary)

fSolveSol = fsolve(solverCb, initialFSolveState, epsfcn=0.000001, full_output=True)
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
stateAtTf = SafeSubs(problem.StateVariables, {problem.TimeSymbol: problem.TimeFinalSymbol})
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
    initialValuesAtTau0 = SafeSubs(initialStateValues, {baseProblem.TimeInitialSymbol: problem.TimeInitialSymbol})
    constantsSubsDict.update(zip(initialValuesAtTau0, [r0, u0, v0, lon0]))

    r0= r0/r0
    u0=u0/v0
    v0=v0/v0
    lon0=lon0/1.0
    # add the scaled initial values (at tau_0).  We should NOT need to add these at t_0
    initialScaledStateValues = problem.CreateVariablesAtTime0(problem.StateVariables)
    constantsSubsDict.update(zip(initialScaledStateValues, [r0, u0, v0, lon0])) 
    
# this next block does most of the problem, pretty standard optimal control actions
lambdas = CreateCoVector(problem.StateVariables, r'\lambda', problem.TimeSymbol)
problem.StateVariables.extend(lambdas)
hamiltonian = problem.CreateHamiltonian(lambdas)
lambdaDotExpressions = problem.CreateLambdaDotCondition(hamiltonian)
dHdu = problem.CreateHamiltonianControlExpressions(hamiltonian)[0]
controlSolved = sy.solve(dHdu, problem.ControlVariables[0])[0] # something that may be different for other problems is when there are multiple control variables

# you are in control of the order of integration variables and what equations of motion get evaluated, start updating the problem
# NOTE that this call adds the lambdas to the integration state
problem.StateVariableDynamics.extend(lambdaDotExpressions)
SafeSubs(problem.StateVariableDynamics, {problem.ControlVariables[0]: controlSolved})
# the trig simplification needs the deep=True for this problem to make the equations even cleaner
for i in range(0, len(problem.StateVariableDynamics)) :
    problem.StateVariableDynamics[i] = problem.StateVariableDynamics[i].trigsimp(deep=True).simplify() # some simplification to make numerical code more stable later, and that is why this code forces us to do things somewhat manually.  There are often special things like this that we ought to do that you can't really automate.

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
for i in range(0, len(problem.StateVariableDynamics)) :
    thisOdeHelper.setStateElement(problem.StateVariables[i], problem.StateVariableDynamics[i], key.subs(problem.TimeSymbol, problem.TimeInitialSymbol) )

if scaleTime:
    thisOdeHelper.lambdifyParameterSymbols.append(baseProblem.TimeFinalSymbol)

if len(nus) != 0:
    thisOdeHelper.lambdifyParameterSymbols.append(problem.StateVariables[5])
    thisOdeHelper.lambdifyParameterSymbols.append(problem.StateVariables[6])

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
stateForBoundaryConditions.extend(SafeSubs(problem.IntegrationSymbols, {problem.TimeSymbol: problem.TimeInitialSymbol}))
stateForBoundaryConditions.extend(SafeSubs(problem.IntegrationSymbols, {problem.TimeSymbol: problem.TimeFinalSymbol}))
stateForBoundaryConditions.extend(lambdas)
stateForBoundaryConditions.extend(otherArgs)

fSolveCallback = ContinuousThrustCircularOrbitTransferProblem.createSolveIvpSingleShootingCallbackForFSolve(problem, problem.IntegrationSymbols, [r0, u0, v0, lon0], tArray, odeIntEomCallback, problem.BoundaryConditions, SafeSubs(lambdas, {problem.TimeSymbol: problem.TimeInitialSymbol}), otherArgs)
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
