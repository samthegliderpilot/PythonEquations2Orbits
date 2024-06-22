#%%
from pyeq2orb.ProblemBase import Problem, ProblemVariable
from pyeq2orb import SafeSubs
from pyeq2orb.NumericalOptimizerProblem import NumericalOptimizerProblemBase
import sympy as sy
from IPython.display import display
from scipyPaperPrinter import printMarkdown, showEquation #type: ignore
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import pyeq2orb.Numerical.ScipyCallbackCreators as ScipyCallbackCreators
from scipy.integrate import solve_ivp
class EverythingAnswer(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod    
    def StateHistory(self) -> Dict[sy.Symbol, List[float]]:
        pass

    @property
    @abstractmethod
    def BoundaryConditionValues(self) -> List[float]:
        pass

class SimpleEverythingAnswer(EverythingAnswer):
    def __init__(self, StateHistory : Dict[sy.Symbol, List[float]], bcAnswer : List[float]):
        self._stateHistory = StateHistory
        self._boundaryConditionValues = bcAnswer

    @property
    def StateHistory(self) -> Dict[sy.Symbol, List[float]]:
        return self._stateHistory

    @property
    def BoundaryConditionValues(self) -> List[float]:
        return self._boundaryConditionValues




class EverythingProblem(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def StateVariables(self) ->List[sy.Symbol]:
        pass

    @abstractmethod
    def EvaluateProblem(self, time, initialState : List[float], parameters : Tuple[float]) ->EverythingAnswer:
        pass

    def EvaluateIndirectProblem(self, time, initialStateValues: List[float], initialCostateValues: List[float], parameters : Tuple[float]) ->EverythingAnswer:
        fullInitialState = []
        fullInitialState.extend(initialStateValues)
        fullInitialState.extend(initialCostateValues)
        return self.EvaluateProblem(time, fullInitialState, parameters)


class singleShootingFunctions(EverythingProblem, ABC):
    def __init__(self):
        pass

    @abstractmethod
    def differentialEquation(self, t, y, args) -> Dict[sy.Symbol, Tuple[float]]:
        pass

    def buildBoundaryConditionState(self, bcInitialState, bcFinalState):
        stateNow = []
        stateNow.extend(bcInitialState)
        stateNow.extend(bcFinalState)
        stateNow.append(bcTimeValue)
        return stateNow

    def buildBoundaryConditionStateFromSolutionDict(self, solutionDict):
        bcState = []
        bcState.append(0)
        for k, v in solutionDict.items():
            bcState.append(v[0])
        bcState.append(1)
        for k, v in solutionDict.items():
            bcState.append(v[-1])            
        return bcState

    @abstractmethod
    def boundaryConditionEvaluation(self, fullBcState, args) -> List[float]:
        pass
    
    def EvaluateProblem(self, time, initialState : List[float], parameters : Tuple[float]) -> EverythingAnswer:
        ivpAns = self.differentialEquation(time, initialState, parameters)
        bcState = self.buildBoundaryConditionStateFromSolutionDict(ivpAns)
        bcAns = self.boundaryConditionEvaluation(bcState, parameters)
        return SimpleEverythingAnswer(ivpAns, bcAns)


class blackBoxSingleShootingFunctions(singleShootingFunctions):
    def __init__(self, diffeqCallback, boundaryConditionCallback,  stateVariables):
        self._difeqCallback = diffeqCallback
        self._boundaryConditionCallback = boundaryConditionCallback
        self._stateVariables = stateVariables

    def differentialEquation(self, t, y, args):
        return self._difeqCallback(t, y, args)

    def boundaryConditionEvaluation(self, fullBcState, args):
        return self._boundaryConditionCallback(fullBcState, args)

    @property
    def StateVariables(self) ->List[sy.Symbol]:
        return self._stateVariables


#%%
t = sy.Symbol('t', real=True, positive=True)
t0 = sy.Symbol('t_0', real=True, positive=True)
tf = sy.Symbol('t_f', real=True, positive=True)
x = sy.Function('x', real=True)
y = sy.Function('y', real=True)
vx = sy.Function('v_x', real=True)
vy = sy.Function('v_y', real=True)
accel = sy.Symbol('a', real=True, positive=True)
m = sy.Function('m', real=True, positive=True)
g = sy.Symbol('g', real=True, positive=True)

alpha = sy.Function(r'\alpha', real=True)
rCb = sy.Symbol('r_{e}', real=True, positive=True)
rc = sy.Symbol('r_c', real=True, positive=True)
vxf = sy.Symbol('v_{x_f}', real=True, positive=True)
mDotS = sy.Symbol('\dot{m}', real=True)
m0 = sy.Symbol("m_0", real=True, positive=True)

xDot = vx(t)
yDot = vy(t)
vxDot = accel*sy.cos(alpha(t))
vyDot = accel*sy.sin(alpha(t)) - g
mDot = t*mDotS

display(vxDot)

bc1 = y(tf)-rc+rCb#=0
bc2 = vxf-vx(tf)#=0
bc3 = vy(tf) #=0


######## Standard problem setup
pr_t = t
pr_y = [x(t), y(t), vx(t), vy(t)] 
pr_eom = [xDot, yDot, vxDot, vyDot]
pr_initialGuesses = [0, 0, 0, 0]
pr_bcs = [bc1, bc2, bc3]
pr_j = tf
pr_sense = -1 # minimize
pr_controls = [alpha(t)]
pr_subsDict = {vxf:1.6270, rCb:1740.0, rc:1852.000, g:0.00162, accel:3*g, mDotS:-0.001, m0:2000.0} #type: ignore

problem = Problem()
for (k,v) in pr_subsDict.items():
    problem.SubstitutionDictionary[k] = v

for i in range(0, len(pr_y)):
    problem.AddStateVariable(ProblemVariable(pr_y[i], pr_eom[i]))

problem.ControlVariables.append(pr_controls[0])

problem.BoundaryConditions.extend(pr_bcs)
problem.TimeFinalSymbol = tf
problem.TimeInitialSymbol = t0
problem.TimeSymbol = t

tau = sy.Symbol(r'\tau')
tau0 = sy.Symbol(r'\tau_0')
tauf = sy.Symbol(r'\tau_f')
problem = problem.ScaleTime(tau, tau0, tauf, tau*tf)


######## Indirect specific solving
lambdas = Problem.CreateCoVector(pr_y, None, tau)
hamiltonian = problem.CreateHamiltonian(lambdas)
showEquation("H", hamiltonian)
lambdasEom = Problem.CreateLambdaDotEquationsStatic(hamiltonian, tau, problem.StateVariables, lambdas)
for i in range(0, len(pr_y)):
    problem.AddCostateVariable(ProblemVariable(lambdas[i], lambdasEom[i]))

controlExpressions = problem.CreateControlExpressionsFromHamiltonian(hamiltonian, problem.ControlVariables)
i=0
for (k,v) in controlExpressions.items(): #type: ignore
    problem.SubstitutionDictionary[k] = v #type: ignore
    showEquation(pr_controls[i], v)
    i=i+1

i=0
for eom in problem.EquationsOfMotionAsEquations:
    display(eom)
    display(problem.CostateDynamicsEquations[i])
    i=i+1

for bc in problem.BoundaryConditions:
    display(sy.Eq(0, bc))

## pick xversality
lambdas_f = SafeSubs(lambdas, {tau:tauf})
lambdas_0 = SafeSubs(lambdas, {tau:tau0})
xversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, sy.Symbol('dt_f'), lambdas_f)
for i in range(0, len(xversality)):
    problem.BoundaryConditions.append(xversality[i])

###### Indirect Problem is fully setup, time to start doing numerical things.....
#%%
import numpy as np
initialGuess = [0,0,0,0,  -1.0,-1.0,-0.01,-10.0]
tArray = np.linspace(0.0, 1.0, 400)

from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions

numerical = OdeLambdifyHelperWithBoundaryConditions.CreateFromProblem(problem)
numerical.SymbolsToSolveForWithBoundaryConditions.clear()
numerical.SymbolsToSolveForWithBoundaryConditions.extend(lambdas_0)
numerical.SymbolsToSolveForWithBoundaryConditions.append(tf)
numerical.ApplySubstitutionDictionaryToExpressions()
ivpCallback = numerical.CreateSimpleCallbackForSolveIvp()

integrationVariables = []
integrationVariables.extend(problem.StateVariables)
integrationVariables.extend(problem.CostateSymbols)

def solve_ivp_wrapper(t, y, args):
    if isinstance(args, list):
        args = tuple(args)
    anAns = solve_ivp(ivpCallback, [t[0], t[-1]], y, dense_output=True, args=args, method='LSODA')
    anAnsDict = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(integrationVariables, anAns)
    return anAnsDict

bcCallback = numerical.CreateCallbackForBoundaryConditionsWithFullState()

#bcCallback = numerical.createCallbackToSolveForBoundaryConditionsBetter(solve_ivp_wrapper, [lambdas_0[0], lambdas_0[1], lambdas_0[2], lambdas_0[3], tf], tArray, initialGuess, (480,))
def tempBcCallback(state, args):
    return bcCallback[1](*state, *args)
problemEvaluator = blackBoxSingleShootingFunctions(solve_ivp_wrapper, tempBcCallback, integrationVariables)
everything = problemEvaluator.EvaluateProblem(tArray, initialGuess, (3600,))
print(everything.BoundaryConditionValues)
print(everything.StateHistory)
#%%

from scipy.integrate import solve_ivp #type: ignore
from scipy.optimize import fsolve  #type: ignore

fSolveInitialGuess = initialGuess[4:]
fSolveInitialGuess.append(480)
fsolveAns = fsolve(bcCallback, fSolveInitialGuess, full_output=True,  factor=0.2,epsfcn=0.001 )
print(fsolveAns)
finalInitialState = initialGuess[:4]
finalInitialState.extend(fsolveAns[0][:4])
tfReal = fsolveAns[0][-1]
anAns = solve_ivp(ivpCallback, [0,1], finalInitialState, dense_output=True, args=(fsolveAns[0][-1],), method='LSODA')
print(anAns)


from pyeq2orb.Graphics.Plotly2DModule import plot2DLines
import pyeq2orb.Graphics.Primitives as prim

def plotAThing(title, label1, t1, dataset1):
    plot2DLines([prim.XAndYPlottableLineData(t1, dataset1, label1, '#0000ff', 2, 0)], title)

titles = ["x", "y", "vx", "vy"]
i=0
for title in titles:
    plotAThing(titles[i], titles[i], anAns.t*tfReal, anAns.y[i])
    i=i+1
i=0

plotAThing("XY Path", "path", anAns.y[0], anAns.y[0])

print(fsolveAns)



