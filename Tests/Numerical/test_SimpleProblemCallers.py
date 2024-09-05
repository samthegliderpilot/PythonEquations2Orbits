import pytest
import sympy as sy
from scipy.integrate import odeint, solve_ivp # type: ignore
import pyeq2orb.Numerical.ScipyCallbackCreators as scipyCreator #type: ignore
from pyeq2orb.Numerical.SimpleProblemCallers import SimpleEverythingAnswer, SingleShootingFunctions, BlackBoxSingleShootingFunctions, SimpleIntegrationAnswer, IIntegrationAnswer, singleShootingSolver, fSolveSingleShootingSolver #type: ignore
import pyeq2orb.Numerical.ScipyCallbackCreators as ScipyCallbackCreators #type: ignore
import math
from typing import Tuple, List, Callable, Optional, Dict, Iterable
from types import MethodType

#class testSimpleEverythingAnswer(unittest.TestCase) :

def testHoldingValues() :
    t =[1,2,3]
    sh = {sy.Symbol("a") : [4,5,6], sy.Symbol("b") : [7,8,9]}
    bc = [-1]
    toTest = SimpleEverythingAnswer("Blah", t, sh, bc) #type: ignore
    assert t == toTest.TimeHistory
    assert sh == toTest.StateHistory
    assert bc == toTest.BoundaryConditionValues
    assert "Blah" == toTest.OriginalProblem

#class testBlackBoxSingleShootingFunctions(unittest.TestCase):

def testBasicOperationNoArgs():
    runSimpleCase(None)

def testBasicOperationArgs():
    runSimpleCase((4.0, ))

def runSimpleCase(passedInArgs):    
    stateSymbols = [sy.Symbol('x'), sy.Symbol('y')]
    bcSymbol = [stateSymbols[0]**2, stateSymbols[0]**3]
    otherArgSymbols = None
    if passedInArgs == None or len(passedInArgs) == 0:
        odeCallback = lambda times, x0, args : SimpleIntegrationAnswer(problem, times, {stateSymbols[0]: [x0[0], x0[0]+x0[0]*math.cos(times[0])],    stateSymbols[1]: [x0[1], x0[1]+x0[1]*math.sin(times[0])] }, "Finished")
        bcCallback = lambda t0, x0, y0, tf, xf, yf, *args: [xf**2, yf**3]
        argValue = 0.0
        passedInArgs = (),
    else:
        odeCallback = lambda times, x0, args : SimpleIntegrationAnswer(problem, times, {stateSymbols[0]: [x0[0], x0[0]+x0[0]*math.cos(times[0])],    stateSymbols[1]: [x0[1], x0[1]+x0[1]*math.sin(times[0])] }, "Finished")
        bcCallback = lambda t0, x0, y0, tf, xf, yf, *args: [args[0] + xf**2, args[0] - yf**3]
        argValue = passedInArgs[0]
        otherArgSymbols = [sy.Symbol("a")]
    problem = BlackBoxSingleShootingFunctions(odeCallback, SingleShootingFunctions.CreateBoundaryConditionCallbackFromLambdifiedCallback(bcCallback), stateSymbols, bcSymbol, otherArgSymbols)

    # test integration
    difeqEvaluated = problem.IntegrateDifferentialEquations([0.0, 1.0], [2.0, 3.0], passedInArgs)
    expectedDifeqAnswer = odeCallback([0.0, 1.0], [2.0, 3.0], passedInArgs)
    assert len(difeqEvaluated.TimeHistory) == len(expectedDifeqAnswer.TimeHistory)
    for i in range(0, len(difeqEvaluated.TimeHistory)):
        for j in range(0, len(stateSymbols)) :
            assert difeqEvaluated.StateHistory[stateSymbols[j]][i] == expectedDifeqAnswer.StateHistory[stateSymbols[j]][i]
        assert difeqEvaluated.TimeHistory[i] == expectedDifeqAnswer.TimeHistory[i]

    # test bc state creation
    bcState = IIntegrationAnswer.BuildBoundaryConditionStateFromIntegrationAnswer(difeqEvaluated)
    assert bcState[0] == 0.0 #t0
    assert bcState[1] == difeqEvaluated.StateHistory[stateSymbols[0]][0] #x0
    assert bcState[2] == difeqEvaluated.StateHistory[stateSymbols[1]][0] #y0
    assert bcState[3] == 1.0 #tf
    assert bcState[4] == difeqEvaluated.StateHistory[stateSymbols[0]][-1] # xf
    assert bcState[5] == difeqEvaluated.StateHistory[stateSymbols[1]][-1] # yf
    assert len(bcState) == 6

    # test bc evaluation
    bcValues = problem.BoundaryConditionEvaluation(difeqEvaluated, passedInArgs)
    expectedBcValues = bcCallback(*bcState, *passedInArgs)
    assert len(bcValues) == 2
    assert bcValues[0] == expectedBcValues[0]
    assert bcValues[0] == expectedBcValues[0]

    # test evaluating overall problem
    answer = problem.EvaluateProblem([0.0, 1.0], [2.0, 3.0], passedInArgs)
    assert len(answer.BoundaryConditionValues) == 2
    assert answer.BoundaryConditionValues[0] == bcValues[0]
    assert answer.BoundaryConditionValues[1] == bcValues[1]
    assert answer.RawIntegratorOutput == "Finished"
    assert len(answer.TimeHistory) == 2
    assert answer.TimeHistory[0] == 0.0
    assert answer.TimeHistory[1] == 1.0
    for i in range(0, len(difeqEvaluated.TimeHistory)):
        for j in range(0, len(stateSymbols)) :
            assert answer.StateHistory[stateSymbols[j]][i] == expectedDifeqAnswer.StateHistory[stateSymbols[j]][i]
        assert answer.TimeHistory[i] == expectedDifeqAnswer.TimeHistory[i]

class fakeSingleShooting(SingleShootingFunctions):
    def __init__(self):
        self._stateCount =2
        self._argCount = 2
        self._bcCount =2 

    @property
    def StateSymbols(self) ->List[sy.Symbol]:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        symbols = []
        for i in range(0, self._stateCount) :
            symbols.append(sy.Symbol(alphabet[i], real=True))
        return symbols


    @property
    def OtherArgumentSymbols(self) -> List[sy.Symbol]:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        symbols = []
        for i in range(0, self._stateCount) :
            symbols.append(sy.Symbol(alphabet[i+5], real=True))
        return symbols
    @property
    def BoundaryConditionExpressions(self) -> List[sy.Expr]: # this is a sympy expression, it must equal 0
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        symbols = self.StateSymbols
        argSymbols = self.OtherArgumentSymbols
        exprs = []
        for i in range(0, self._stateCount) :
            exprs.append(symbols[i]-argSymbols[i])
        return exprs


    def evaluateFakeIntegrationHistory(self, time : Iterable[float], y0 : List[float], args  : List[float]) -> Dict[sy.Symbol, List[float]]:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        solution : Dict[sy.Symbol, List[float]]= {}
        for i in range(0, len(y0)):
            thisSymbol = sy.Symbol(alphabet[i], real=True)
            solution[thisSymbol] = []
            for t in time:
                solution[thisSymbol].append(y0[i]+args[i])
        return solution

    def IntegrateDifferentialEquations(self, time : Iterable[float], y0 : List[float], args  : List[float]) -> IIntegrationAnswer:
        results = self.evaluateFakeIntegrationHistory(time, y0, args)
        answer = SimpleIntegrationAnswer(self, time, results, None)    
        return answer

    def BoundaryConditionEvaluation(self, integrationAnswer: IIntegrationAnswer, args : List[float]) -> List[float]:    
        bcValues : List[float]= []
        i=0
        for key, values in integrationAnswer.StateHistory.items():
            bcValues.append(values[-1]-10)
            i=i+1
        return bcValues

def testBasics():

    t = sy.Symbol('t')
    t0 = sy.Symbol('t_0')
    tf = sy.Symbol('t_f')

    basicProblem = fakeSingleShooting()
    
    stateSymbols = basicProblem.StateSymbols
    argSymbols = basicProblem.OtherArgumentSymbols
    boundaryConditionExpressions = basicProblem.BoundaryConditionExpressions
    solver = fSolveSingleShootingSolver(basicProblem, [stateSymbols[1], argSymbols[0]], boundaryConditionExpressions[:2])
    problemEvaluated = solver.EvaluatableProblem.EvaluateProblem([0.0, 5.0, 10.0], [30, 20], [15, 5])
    print(problemEvaluated)
    valuesFromOneOffAnswer = solver.getSolverValuesFromEverythingAns(problemEvaluated)
    assert valuesFromOneOffAnswer[0] == problemEvaluated.BoundaryConditionValues[0]
    assert valuesFromOneOffAnswer[1] == problemEvaluated.BoundaryConditionValues[1]
    assert len(valuesFromOneOffAnswer) == 2

    newInitialState = [0.0, 5.0]
    solver.updateInitialStateWithSolverValues([-1.0, -2.0], newInitialState)
    assert newInitialState[0] == 0.0
    assert newInitialState[1] == -1

    newInitialArgs = [15, 5]
    solver.updateInitialParametersWithSolverValues([-1.0, -2.0], newInitialArgs)
    assert newInitialArgs[0] == -2.0
    assert newInitialArgs[1] == 5
    
    sampleEverythingAnswer = SimpleEverythingAnswer(basicProblem, [0.0, 5.0, 10.0], {stateSymbols[0]: [2.0, 3.0, 4.0], stateSymbols[1]: [-11.0, -10.0, -8.0]}, [5.0, 6.0, 8.0], "finished")
    solverAns = solver.getSolverValuesFromEverythingAns(sampleEverythingAnswer)
    assert len(solverAns) == 2
    assert solverAns[0] == 5.0
    assert solverAns[1] == 6.0

    ans = solver.solve([4.0, 5.0], [0.0, 5.0, 10.0], [4.0, 5.0], full_output=True)
    assert 10.0 == ans.SolvedControls[0]
    assert 6.0 == ans.SolvedControls[1]
    assert 0.0 == ans.ConstraintValues[0]
    assert 0.0 == ans.ConstraintValues[1]

    
    
    