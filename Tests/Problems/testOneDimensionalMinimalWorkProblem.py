import unittest
from PythonOptimizationWithNlp.Problems.OneDimensionalMinimalWorkProblem import OneDWorkSymbolicProblem, OneDWorkProblem, AnalyticalAnswerToProblem

class testOneDimensionalMinimalWorkProblem(unittest.TestCase) :

    def testConstructor(self) :
        oneDWorkProblem = OneDWorkProblem()
        self.assertEqual(2, oneDWorkProblem.NumberOfStateVariables, msg="state variable count")
        self.assertEqual(1, oneDWorkProblem.NumberOfControlVariables, msg="control variable count")
        self.assertEqual(0, len(oneDWorkProblem.BoundaryConditionCallbacks), msg="number of bc callbacks")
        self.assertEqual(0.0, oneDWorkProblem.KnownInitialConditions[oneDWorkProblem.State[0]], msg="x_0 bc")
        self.assertEqual(0.0, oneDWorkProblem.KnownInitialConditions[oneDWorkProblem.State[1]], msg="vx_0 bc")
        self.assertEqual(1.0, oneDWorkProblem.KnownFinalConditions[oneDWorkProblem.State[0]], msg="x_f bc")
        self.assertEqual(0.0, oneDWorkProblem.KnownFinalConditions[oneDWorkProblem.State[1]], msg="vx_f bc")
        self.assertEqual(0.0, oneDWorkProblem.T0, msg="t0 value")
        self.assertEqual(1.0, oneDWorkProblem.Tf, msg="tf value")
        self.assertIsNotNone(oneDWorkProblem.Time, "time object is set")

    def testInitialGuessCallbackStart(self) :
        oneDWorkProblem = OneDWorkProblem()
        expectedIc = [0.0, 0.0, 0.0]
        ic = oneDWorkProblem.InitialGuessCallback(0.0)
        for i in range(0, len(expectedIc)) :
            self.assertEqual(expectedIc[i], ic[i], msg="IC at index " + str(i))

    def testInitialGuessCallbackEarlyMidRange(self) :
        oneDWorkProblem = OneDWorkProblem()
        expectedIc = [0.25, 1.0, 0.1]
        ic = oneDWorkProblem.InitialGuessCallback(0.25)
        for i in range(0, len(expectedIc)) :
            self.assertEqual(expectedIc[i], ic[i], msg="IC at index " + str(i))

    def testInitialGuessCallbackLateMidRange(self) :
        oneDWorkProblem = OneDWorkProblem()
        expectedIc = [0.75, -1.0, -0.1]
        ic = oneDWorkProblem.InitialGuessCallback(0.75)
        for i in range(0, len(expectedIc)) :
            self.assertEqual(expectedIc[i], ic[i], msg="IC at index " + str(i))

    def testInitialGuessCallbackFinal(self) :
        oneDWorkProblem = OneDWorkProblem()
        expectedIc = [1.0, -1.0, 0.0]
        ic = oneDWorkProblem.InitialGuessCallback(1.0)
        for i in range(0, len(expectedIc)) :
            self.assertEqual(expectedIc[i], ic[i], msg="IC at index " + str(i))

    def testEquationsOfMotion(self) :
        oneDWorkProblem = OneDWorkProblem()
        dxdt = oneDWorkProblem.EquationOfMotion(0, [1.0, 2.0, 3.0])
        self.assertEqual(2, len(dxdt), msg="length of dxdt")
        self.assertEqual(2.0, dxdt[0], msg="x dot")
        self.assertEqual(3.0, dxdt[1], msg="v dot")

    def testCostFunction(self) :
        oneDWorkProblem = OneDWorkProblem()
        history = {}
        history[oneDWorkProblem.State[0]]=[1.0, 1.0, 1.0, 1.0]
        history[oneDWorkProblem.State[1]]=[0.1, 0.1, 0.1, 0.1]
        history[oneDWorkProblem.Control[0]]=[0.25, 0.25, 1.0, 1.0]

        cost = oneDWorkProblem.CostFunction(oneDWorkProblem.CreateTimeRange(3), history)
        self.assertAlmostEqual(0.53125,  cost, delta=0.00001, msg="cost")

class testOneDimensionalMinimalWorkProblemAnalyticalAnswerToProblem(unittest.TestCase) :

    def testOptimalX(self) :
        analyticalAnswerEvaluator = AnalyticalAnswerToProblem()
        optX = analyticalAnswerEvaluator.OptimalX(0.25)
        assert optX == 0.15625

    def testOptimalV(self) :
        analyticalAnswerEvaluator = AnalyticalAnswerToProblem()
        optV = analyticalAnswerEvaluator.OptimalV(0.25)
        assert optV == 1.125

    def testOptimalU(self) :
        analyticalAnswerEvaluator = AnalyticalAnswerToProblem()
        optU = analyticalAnswerEvaluator.OptimalControl(0.25)
        assert optU == 3.0

    def testAnalyticalSolution(self) :
        oneDWorkProblem = OneDWorkProblem()
        t = oneDWorkProblem.CreateTimeRange(5)
        analyticalAnswerEvaluator = AnalyticalAnswerToProblem()
        analyticalAnswer = analyticalAnswerEvaluator.EvaluateAnswer(oneDWorkProblem, t)
        self.AssertEqualityOfAnalyticalSolution(oneDWorkProblem, analyticalAnswer)

    def testAnalyticalSolutionDefaultT(self) :
        oneDWorkProblem = OneDWorkProblem()
        t = oneDWorkProblem.CreateTimeRange(5)
        analyticalAnswerEvaluator = AnalyticalAnswerToProblem()
        analyticalAnswer = analyticalAnswerEvaluator.EvaluateAnswer(oneDWorkProblem, t)
        self.AssertEqualityOfAnalyticalSolution(oneDWorkProblem, analyticalAnswer)

    def AssertEqualityOfAnalyticalSolution(self, oneDWorkProblem, analyticalAnswer):
        expectedX = [0.0, 0.104, 0.352, 0.648, 0.896, 1.0]
        expectedV = [0.0, 0.96, 1.44, 1.44, 0.96, 0.0]
        expectedU = [6.0, 3.6, 1.2, -1.2, -3.6, -6.0]
        for i in range(0, 6) :
            self.assertAlmostEqual(analyticalAnswer[oneDWorkProblem.State[0]][i], expectedX[i], places=6, msg="X at index " + str(i))
            self.assertAlmostEqual(analyticalAnswer[oneDWorkProblem.State[1]][i], expectedV[i], places=6, msg="V at index " + str(i))
            self.assertAlmostEqual(analyticalAnswer[oneDWorkProblem.Control[0]][i], expectedU[i], places=6, msg="U at index " + str(i))

class testOneDWorkSymbolicProblem(unittest.TestCase) :
    def testBoilerplate(self) :
        problem = OneDWorkSymbolicProblem()
        self.assertEqual(0, problem.TerminalCost, msg="terminal cost")
        self.assertEqual(2, len(problem.StateVariables), msg="count of state variables")
        self.assertEqual(1, len(problem.ControlVariables), msg="count of control variables")
        self.assertEqual(problem.ControlVariables[0]**2, problem.UnIntegratedPathCost, msg="unintegrated path cost")
        self.assertEqual("t" , problem.TimeSymbol.name, msg="time symbol")
        self.assertEqual("t_f" , problem.TimeFinalSymbol.name, msg="time final symbol")
        self.assertEqual("t_0" , problem.TimeInitialSymbol.name, msg="time initial symbol")
        self.assertEqual(0, len(problem.ConstantSymbols), msg="number of constants")
        self.assertEqual(2, problem.EquationsOfMotion[problem.StateVariables[0]].subs(problem.StateVariables[1], 2), msg="eom 1")
        self.assertEqual(3, problem.EquationsOfMotion[problem.StateVariables[1]].subs(problem.ControlVariables[0], 3), msg="eom 2")
        self.assertEqual(0, problem.BoundaryConditions[0].subs(problem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol), 1), msg="bc1")
        self.assertEqual(1, problem.BoundaryConditions[1].subs(problem.StateVariables[1].subs(problem.TimeSymbol, problem.TimeFinalSymbol), 1), msg="bc2")
