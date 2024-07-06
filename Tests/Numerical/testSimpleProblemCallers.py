import unittest
import sympy as sy
from scipy.integrate import odeint, solve_ivp # type: ignore
import pyeq2orb.Numerical.ScipyCallbackCreators as scipyCreator
from pyeq2orb.Numerical.SimpleProblemCallers import SimpleEverythingAnswer, SingleShootingFunctions, BlackBoxSingleShootingFunctions, SimpleIntegrationAnswer
import math

class testSimpleEverythingAnswer(unittest.TestCase) :

    def testHoldingValues(self) :
        t =[1,2,3]
        sh = {sy.Symbol("a") : [4,5,6], sy.Symbol("b") : [7,8,9]}
        bc = [-1]
        toTest = SimpleEverythingAnswer(t, sh, bc)
        assert t == toTest.TimeHistory
        assert sh == toTest.StateHistory
        assert bc == toTest.BoundaryConditionValues

class testBlackBoxSingleShootingFunctions(unittest.TestCase):

    def testBasicOperationNoArgs(self):
        self.runSimpleCase(None)

    def testBasicOperationArgs(self):
        self.runSimpleCase((4.0, ))

    def runSimpleCase(self, passedInArgs):    
        stateSymbols = [sy.Symbol('x'), sy.Symbol('y')]
        bcSymbol = [stateSymbols[0]**2, stateSymbols[0]**3]
        if passedInArgs == None or len(passedInArgs) == 0:
            odeCallback = lambda times, x0, args : SimpleIntegrationAnswer(times, {stateSymbols[0]: [x0[0], x0[0]+x0[0]*math.cos(times[0])],    stateSymbols[1]: [x0[1], x0[1]+x0[1]*math.sin(times[0])] }, "Finished")
            bcCallback = lambda bcState, args: [bcState[-2]**2, bcState[-1]**3]
            argValue = 0.0
        else:
            odeCallback = lambda times, x0, args : SimpleIntegrationAnswer(times, {stateSymbols[0]: [x0[0], x0[0]+x0[0]*math.cos(times[0])],    stateSymbols[1]: [x0[1], x0[1]+x0[1]*math.sin(times[0])] }, "Finished")
            bcCallback = lambda bcState, args: [args[0] + bcState[-2]**2,  args[0]-bcState[-1]**3]
            argValue = passedInArgs[0]
        problem = BlackBoxSingleShootingFunctions(odeCallback, bcCallback, stateSymbols, bcSymbol)

        # test integration
        difeqEvaluated = problem.IntegrateDifferentialEquations([0.0, 1.0], [2.0, 3.0], passedInArgs)
        expectedDifeqAnswer = odeCallback([0.0, 1.0], [2.0, 3.0], passedInArgs)
        assert len(difeqEvaluated.TimeHistory) == len(expectedDifeqAnswer.TimeHistory)
        for i in range(0, len(difeqEvaluated.TimeHistory)):
            for j in range(0, len(stateSymbols)) :
                assert difeqEvaluated.StateHistory[stateSymbols[j]][i] == expectedDifeqAnswer.StateHistory[stateSymbols[j]][i]
            assert difeqEvaluated.TimeHistory[i] == expectedDifeqAnswer.TimeHistory[i]

        # test bc state creation
        bcState = problem.BuildBoundaryConditionStateFromIntegrationAnswer(difeqEvaluated)
        assert bcState[0] == 0.0 #t0
        assert bcState[1] == difeqEvaluated.StateHistory[stateSymbols[0]][0] #x0
        assert bcState[2] == difeqEvaluated.StateHistory[stateSymbols[1]][0] #y0
        assert bcState[3] == 1.0 #tf
        assert bcState[4] == difeqEvaluated.StateHistory[stateSymbols[0]][-1] # xf
        assert bcState[5] == difeqEvaluated.StateHistory[stateSymbols[1]][-1] # yf
        assert len(bcState) == 6

        # test bc evaluation
        bcValues = problem.BoundaryConditionEvaluation(bcState, passedInArgs)
        expectedBcValues = bcCallback(bcState, passedInArgs)
        assert len(bcValues) == 2
        assert bcValues[0] == expectedBcValues[0]

        # test evaluating overall problem
        answer = problem.EvaluateProblem([0.0, 1.0], [2.0, 3.0], passedInArgs)
        assert len(answer.BoundaryConditionValues) == 2
        assert answer.BoundaryConditionValues[0] == bcValues[0]
        assert answer.RawIntegratorOutput == "Finished"
        assert len(answer.TimeHistory) == 2
        assert answer.TimeHistory[0] == 0.0
        assert answer.TimeHistory[1] == 1.0
        for i in range(0, len(difeqEvaluated.TimeHistory)):
            for j in range(0, len(stateSymbols)) :
                assert answer.StateHistory[stateSymbols[j]][i] == expectedDifeqAnswer.StateHistory[stateSymbols[j]][i]
            assert answer.TimeHistory[i] == expectedDifeqAnswer.TimeHistory[i]

        
