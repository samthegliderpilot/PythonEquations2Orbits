import pytest
from pyeq2orb.Problems.OneDimensionalMinimalWorkProblem import OneDWorkSymbolicProblem, OneDWorkProblem, AnalyticalAnswerToProblem #type: ignore

def testConstructor() :
    oneDWorkProblem = OneDWorkProblem()
    assert 2== oneDWorkProblem.NumberOfStateVariables, "state variable count"
    assert 1== oneDWorkProblem.NumberOfControlVariables, "control variable count"
    assert 0== len(oneDWorkProblem.BoundaryConditionCallbacks), "number of bc callbacks"
    assert 0.0== oneDWorkProblem.KnownInitialConditions[oneDWorkProblem.State[0]], "x_0 bc"
    assert 0.0== oneDWorkProblem.KnownInitialConditions[oneDWorkProblem.State[1]], "vx_0 bc"
    assert 1.0== oneDWorkProblem.KnownFinalConditions[oneDWorkProblem.State[0]], "x_f bc"
    assert 0.0== oneDWorkProblem.KnownFinalConditions[oneDWorkProblem.State[1]], "vx_f bc"
    assert 0.0== oneDWorkProblem.T0, "t0 value"
    assert 1.0== oneDWorkProblem.Tf, "tf value"
    assert oneDWorkProblem.Time is not None, "time object is set"

def testInitialGuessCallbackStart() :
    oneDWorkProblem = OneDWorkProblem()
    expectedIc = [0.0, 0.0, 0.0]
    ic = oneDWorkProblem.InitialGuessCallback(0.0)
    for i in range(0, len(expectedIc)) :
        assert expectedIc[i]== ic[i], "IC at index " + str(i)

def testInitialGuessCallbackEarlyMidRange() :
    oneDWorkProblem = OneDWorkProblem()
    expectedIc = [0.25, 1.0, 0.1]
    ic = oneDWorkProblem.InitialGuessCallback(0.25)
    for i in range(0, len(expectedIc)) :
        assert expectedIc[i]== ic[i], "IC at index " + str(i)

def testInitialGuessCallbackLateMidRange() :
    oneDWorkProblem = OneDWorkProblem()
    expectedIc = [0.75, -1.0, -0.1]
    ic = oneDWorkProblem.InitialGuessCallback(0.75)
    for i in range(0, len(expectedIc)) :
        assert expectedIc[i]== ic[i], "IC at index " + str(i)

def testInitialGuessCallbackFinal() :
    oneDWorkProblem = OneDWorkProblem()
    expectedIc = [1.0, -1.0, 0.0]
    ic = oneDWorkProblem.InitialGuessCallback(1.0)
    for i in range(0, len(expectedIc)) :
        assert expectedIc[i]== ic[i], "IC at index " + str(i)

def testEquationsOfMotion() :
    oneDWorkProblem = OneDWorkProblem()
    dxdt = oneDWorkProblem.EquationOfMotion(0, [1.0, 2.0, 3.0])
    assert 2== len(dxdt), "length of dxdt"
    assert 2.0== dxdt[0], "x dot"
    assert 3.0== dxdt[1], "v dot"

def testCostFunction() :
    oneDWorkProblem = OneDWorkProblem()
    history = {}
    history[oneDWorkProblem.State[0]]=[1.0, 1.0, 1.0, 1.0]
    history[oneDWorkProblem.State[1]]=[0.1, 0.1, 0.1, 0.1]
    history[oneDWorkProblem.Control[0]]=[0.25, 0.25, 1.0, 1.0]

    cost = oneDWorkProblem.CostFunction(oneDWorkProblem.CreateTimeRange(3), history)
    assert pytest.approx(cost, 0.00001) == 0.5052083333333334, "cost"



def testOptimalX() :
    analyticalAnswerEvaluator = AnalyticalAnswerToProblem()
    optX = analyticalAnswerEvaluator.OptimalX(0.25)
    assert optX == 0.15625

def testOptimalV() :
    analyticalAnswerEvaluator = AnalyticalAnswerToProblem()
    optV = analyticalAnswerEvaluator.OptimalV(0.25)
    assert optV == 1.125

def testOptimalU() :
    analyticalAnswerEvaluator = AnalyticalAnswerToProblem()
    optU = analyticalAnswerEvaluator.OptimalControl(0.25)
    assert optU == 3.0

def testAnalyticalSolution() :
    oneDWorkProblem = OneDWorkProblem()
    t = oneDWorkProblem.CreateTimeRange(5)
    analyticalAnswerEvaluator = AnalyticalAnswerToProblem()
    analyticalAnswer = analyticalAnswerEvaluator.EvaluateAnswer(oneDWorkProblem, t)
    AssertEqualityOfAnalyticalSolution(oneDWorkProblem, analyticalAnswer)

def testAnalyticalSolutionDefaultT() :
    oneDWorkProblem = OneDWorkProblem()
    t = oneDWorkProblem.CreateTimeRange(5)
    analyticalAnswerEvaluator = AnalyticalAnswerToProblem()
    analyticalAnswer = analyticalAnswerEvaluator.EvaluateAnswer(oneDWorkProblem, t)
    AssertEqualityOfAnalyticalSolution(oneDWorkProblem, analyticalAnswer)

def AssertEqualityOfAnalyticalSolution(oneDWorkProblem, analyticalAnswer):
    expectedX = [0.0, 0.104, 0.352, 0.648, 0.896, 1.0]
    expectedV = [0.0, 0.96, 1.44, 1.44, 0.96, 0.0]
    expectedU = [6.0, 3.6, 1.2, -1.2, -3.6, -6.0]
    for i in range(0, 6) :
        assert pytest.approx(expectedX[i], 0.0000001) == analyticalAnswer[oneDWorkProblem.State[0]][i], "X at index " + str(i)
        assert pytest.approx(expectedV[i], 0.000001) == analyticalAnswer[oneDWorkProblem.State[1]][i], "V at index " + str(i)
        assert pytest.approx(expectedU[i], 0.000001) == analyticalAnswer[oneDWorkProblem.Control[0]][i], "U at index " + str(i)

def testIntialTrajectoryGuessCallback() :
    oneDWorkProblem = OneDWorkProblem()
    n = 5
    initialState = [0.0, 1.0, 2.0]
    finalState = [10.0, 11.0, 12.0]
    trajectoryGuess = oneDWorkProblem.InitialTrajectoryGuess(n, 0.0, initialState, 10.0, finalState)
    assert 4== len(trajectoryGuess), "should be 4 elements"
    assert [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]== trajectoryGuess[oneDWorkProblem.State[0]], "x guess"
    assert [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]== trajectoryGuess[oneDWorkProblem.State[1]], "vx guess"
    assert [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]== trajectoryGuess[oneDWorkProblem.Control[0]], "u guess"


def testBoilerplate() :
    problem = OneDWorkSymbolicProblem()
    assert 0== problem.TerminalCost, "terminal cost"
    assert 2== len(problem.StateSymbols), "count of state variables"
    assert 1== len(problem.ControlSymbols), "count of control variables"
    assert problem.ControlSymbols[0]**2== problem.UnIntegratedPathCost, "unintegrated path cost"
    assert "t" == problem.TimeSymbol.name, "time symbol"
    assert "t_f" == problem.TimeFinalSymbol.name, "time final symbol"
    assert "t_0" == problem.TimeInitialSymbol.name, "time initial symbol"
    assert 0== len(problem.ConstantSymbols), "number of constants"
    assert 2== problem.StateVariableDynamics[0].subs(problem.StateSymbols[1], 2), "eom 1"
    assert 3== problem.StateVariableDynamics[1].subs(problem.ControlSymbols[0], 3), "eom 2"
    assert 0== problem.BoundaryConditions[0].subs(problem.StateSymbols[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol), 1), "bc1"
    assert 1== problem.BoundaryConditions[1].subs(problem.StateSymbols[1].subs(problem.TimeSymbol, problem.TimeFinalSymbol), 1), "bc2"
