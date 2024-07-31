import unittest
import sympy as sy
from scipy.integrate import odeint, solve_ivp # type: ignore
import pyeq2orb.Numerical.ScipyCallbackCreators as scipyCreator
from pyeq2orb.Numerical.SimpleProblemCallers import SimpleEverythingAnswer, SingleShootingFunctions, BlackBoxSingleShootingFunctions, SimpleIntegrationAnswer, IIntegrationAnswer, singleShootingSolver, fSolveSingleShootingSolver
import pyeq2orb.Numerical.ScipyCallbackCreators as ScipyCallbackCreators
import math
from typing import Tuple, List, Callable, Optional
from types import MethodType

class testSimpleEverythingAnswer(unittest.TestCase) :

    def testHoldingValues(self) :
        t =[1,2,3]
        sh = {sy.Symbol("a") : [4,5,6], sy.Symbol("b") : [7,8,9]}
        bc = [-1]
        toTest = SimpleEverythingAnswer("Blah", t, sh, bc) #type: ignore
        assert t == toTest.TimeHistory
        assert sh == toTest.StateHistory
        assert bc == toTest.BoundaryConditionValues
        assert "Blah" == toTest.OriginalProblem

class testBlackBoxSingleShootingFunctions(unittest.TestCase):

    def testBasicOperationNoArgs(self):
        self.runSimpleCase(None)

    def testBasicOperationArgs(self):
        self.runSimpleCase((4.0, ))

    def runSimpleCase(self, passedInArgs):    
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
        bcValues = problem.BoundaryConditionEvaluation(difeqEvaluated, *passedInArgs)
        expectedBcValues = bcCallback(*bcState, *passedInArgs)
        assert len(bcValues) == 2
        assert bcValues[0] == expectedBcValues[0]
        assert bcValues[0] == expectedBcValues[0]

        # test evaluating overall problem
        answer = problem.EvaluateProblem([0.0, 1.0], [2.0, 3.0], *passedInArgs)
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

        


class testFSolveSingleShootingSolver(unittest.TestCase):

    @staticmethod
    def simpleOdeCallback(t:float, xState:List[float], *args : float) ->List[float]:
        dx1 = xState[1] + args[0] + 0.0
        dx2 = xState[0] + args[1] + 1.0
        return [dx1, dx2]

    @staticmethod
    def boundaryConditionCallback(integrationAnswer : IIntegrationAnswer, *args : float) ->List[float]:
        desiredAnswer1 = integrationAnswer.StateVariableHistoryByIndex(0)[-1] - args[0]
        desiredAnswer2 = integrationAnswer.StateVariableHistoryByIndex(1)[-1] - args[1]
        return [desiredAnswer1, desiredAnswer2]

    def testBasics(self):
        def solve_ivp_wrapper(t, y, *args):
            realArgs = args
            # if isinstance(args, list):
            #     realArgs = *args
            anAns = solve_ivp(testFSolveSingleShootingSolver.simpleOdeCallback, [t[0], t[-1]], y, dense_output=True, t_eval=t, args=realArgs, method='LSODA')
            anAnsDict = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(stateSymbols, anAns)
            return SimpleIntegrationAnswer(basicProblem, t, anAnsDict)

        t = sy.Symbol('t')
        t0 = sy.Symbol('t_0')
        tf = sy.Symbol('t_f')
        stateSymbols = [sy.Function('x')(t), sy.Function('y')(t)]
        argSymbols = [sy.Symbol('a'), sy.Symbol('b')]
        boundaryConditionExpressions  = [sy.Function('x')(tf) - argSymbols[0], sy.Function('y')(tf) - argSymbols[1], 30]
        
        basicProblem = BlackBoxSingleShootingFunctions(solve_ivp_wrapper, testFSolveSingleShootingSolver.boundaryConditionCallback, stateSymbols, boundaryConditionExpressions, argSymbols)
        solver = fSolveSingleShootingSolver(basicProblem, [stateSymbols[1], argSymbols[0]], boundaryConditionExpressions[:2])
        problemEvaluated = solver.EvaluatableProblem.EvaluateProblem([0.0, 5.0, 10.0], [30, 20], *(15, 5))
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

        ans = solver.solve([2.0, 3.0], [0.0, 5.0, 10.0], [4.0, 5.0], *(5.0, 6.0), full_output=True)
        # def interceptFsolveRun(self, solverFunc, solverState, **kwargs) :
        #     print(solverState)
        #     return "Solved"

        # solver.fsolveRun = MethodType(interceptFsolveRun, solver)
        solver.solve([2.0, 3.0], [0.0, 5.0, 10.0], [2.0, 3.0], *(5.0, 6.0), full_output=True)
        
        
        