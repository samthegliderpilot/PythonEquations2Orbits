import pytest
from pytest import approx
from pyeq2orb.Solvers.ScipyDiscretizationMinimizeWrapper import ScipyDiscretizationMinimizeWrapper  # type: ignore
from pyeq2orb.Problems.OneDimensionalMinimalWorkProblem import OneDWorkProblem, AnalyticalAnswerToProblem  # type: ignore
from scipy.interpolate import interp1d # type: ignore
import numpy as np
from pyeq2orb.Utilities.utilitiesForTest import assertAlmostEquals

#TODO: Test needs work
def testSolution() :
    oneDWorkProblem = OneDWorkProblem() # picking a larger steps size to get closer to ideal (but still not exact)
    n= 20
    t=oneDWorkProblem.CreateTimeRange(n)

    truth = AnalyticalAnswerToProblem()
    evaledAns = truth.EvaluateAnswer(oneDWorkProblem, t)        
    interpTruth = interp1d(t, evaledAns[oneDWorkProblem.State[0]], kind='cubic')

    scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)
    ans = scipySolver.ScipyOptimize(n)
    
    assert ans is not None, "answer is not None"
    
    interpAnswer = interp1d(t, ans.x[0:n+1], kind='cubic')
    for i in np.linspace(0,1,11) :
        #assert abs(interpTruth(i) - interpAnswer(i)) <= 0.001 
        # this line can be useful for debugging, leaving it in
        print(str(i) + " " + str(interpTruth(i)) + " " + str(interpAnswer(i)) + " " + str(interpTruth(i)-interpAnswer(i)))

def testCreateDiscretizedInitialGuess() :
    oneDWorkProblem = OneDWorkProblem()
    scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)
    initialGuess = scipySolver.CreateDiscretizedInitialGuess(oneDWorkProblem.CreateTimeRange(4))
    expected = [0.0, 0.25, 0.5, 0.75, 1.0,   0.0, 1.0, 1.0, -1.0, -1.0,   0.0, 0.1, 0.1, -0.1, 0.0]
    for i in range(0, len(expected))  :
        assert expected[i] == initialGuess[i], "value at " + str(i)

def testConvertScipyOptimizerOutputToDictionary() :
    oneDWorkProblem = OneDWorkProblem()
    scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)
    ans = scipySolver.ScipyOptimize(4)
    dictAnswer =scipySolver.ConvertScipyOptimizerOutputToDictionary(ans)
    assert 3== len(dictAnswer), "make sure we get 3 items back"
    assert 5== len(dictAnswer[oneDWorkProblem.State[0]]), "check x is there and is the right length"
    assert 5== len(dictAnswer[oneDWorkProblem.State[1]]), "check v is there and is the right length"
    assert 5== len(dictAnswer[oneDWorkProblem.Control[0]]), "check u is there and is the right length"
    for i in range(0, 5) :
        assert ans.x[i]== dictAnswer[oneDWorkProblem.State[0]][i], "check x at index " + str(i)
        assert ans.x[5+i]== dictAnswer[oneDWorkProblem.State[1]][i], "check x at index " + str(i)
        assert ans.x[10+i]== dictAnswer[oneDWorkProblem.Control[0]][i], "check x at index " + str(i)

def ConvertDiscretizedStateToDict() :
    #TODO: I don't like the copy/paste of the above test here, consider ways to really test this
    oneDWorkProblem = OneDWorkProblem(4)
    scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)
    ans = scipySolver.ScipyOptimize()
    dictAnswer =scipySolver.ConvertDiscretizedStateToDict(ans.x)
    assert 3== len(dictAnswer), "make sure we get 3 items back"
    assert 5== len(dictAnswer[oneDWorkProblem.State[0]]), "check x is there and is the right length"
    assert 5== len(dictAnswer[oneDWorkProblem.State[1]]), "check v is there and is the right length"
    assert 5== len(dictAnswer[oneDWorkProblem.Control[0]]), "check u is there and is the right length"
    for i in range(0, 5) :
        assert ans.x[i]== dictAnswer[oneDWorkProblem.State[0]][i], "check x at index " + str(i)
        assert ans.x[5+i]== dictAnswer[oneDWorkProblem.State[1]][i], "check x at index " + str(i)
        assert ans.x[10+i]== dictAnswer[oneDWorkProblem.Control[0]][i], "check x at index " + str(i)
    

def testEquationOfMotionFromOptimizationState() :
    problem = OneDWorkProblem()
    scipySolver = ScipyDiscretizationMinimizeWrapper(problem)
    equationOfMotionValue = scipySolver.EquationOfMotionInTermsOfOptimizerState(2, [2,12, 22])
    assert 12==equationOfMotionValue[0]
    assert 22==equationOfMotionValue[1]

def testInitialAndFinalStateConstraints() :
    # TODO: This is a fairly weak test due to the nature of OneDWorkProblem, update when a better 
    # problem is available
    problem = OneDWorkProblem()
    scipySolver = ScipyDiscretizationMinimizeWrapper(problem)
    constraints = scipySolver.CreateIndividualBoundaryValueConstraintCallbacks()
    assert 0== len(constraints), "number of additional constraints for this problem"
    state = [1.0, 2.0, 3.0, 4.0,   5.0, 6.0, 7.0, 8.0,   9.0, 10.0, 11.0, 12.0]
    constraints = scipySolver.CreateInitialStateConstraintCallbacks()
    assert 2== len(constraints), "2 initial state constraints returned"
    assert 0.0-state[0]== constraints[0]['fun'](state), "x_0"
    assert 0.0-state[4]== constraints[1]['fun'](state), "v_0"
    constraints = scipySolver.CreateFinalStateConstraintCallbacks()
    assert 2== len(constraints), "2 final state constraints returned"
    assert 1.0-state[3]== constraints[0]['fun'](state), "x_f"
    assert 0.0-state[7]== constraints[1]['fun'](state), "v_f"

def testCostFunctionInTermsOfZ() :
    oneDWorkProblem = OneDWorkProblem()
    scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)

    cost = scipySolver.CostFunctionInTermsOfZ(oneDWorkProblem.CreateTimeRange(3), [1.0, 1.0, 1.0, 1.0,   0.1, 0.1, 0.1, 0.1,   0.25, 0.25, 1.0, 1.0 ])
    assertAlmostEquals(0.5052083333333334 , cost, 0.000001, msg="cost")

def testCreateIndividualColocationConstraints() :
    #TODO: Incomplete
    problem = OneDWorkProblem()
    scipySolver = ScipyDiscretizationMinimizeWrapper(problem)
    t = problem.CreateTimeRange(4)

    constraints = scipySolver.CreateIndividualCollocationConstraints(t)
    assert 8== len(constraints), "8 constraints back"
    z = OptimizerStateForExpectedAnswerFor4LengthConstraint()
    expectedAns = getExpectedAnswersFor4LengthConstraint()

    for i in range(0, len(constraints)) :
        assert expectedAns[i]== constraints[i]['fun'](z), "index " + str(i)

def testGetOptimizerStateAtIndex() :
    problem = OneDWorkProblem()
    scipySolver = ScipyDiscretizationMinimizeWrapper(problem)
    sampleOptimizerState = [0,1,2,3,4,10,11,12,13,14,20,21,22,23,24]
    returnedState = scipySolver.GetOptimizerStateAtIndex(sampleOptimizerState, 2)
    assert returnedState[0] == 2
    assert returnedState[1] == 12
    assert returnedState[2] == 22        

 
def OptimizerStateForExpectedAnswerFor4LengthConstraint() :
    return [0,1,3,6,4,  10,13,17,22,29, 33,43,55,67,80]

def getExpectedAnswersFor4LengthConstraint() :
    # the first few were calculated by hand, the others are answers from an earlier run (making using this a regression test)
    return [1.875, 1.75, 1.875, 8.375, 6.5, 8.25, 10.25, 11.375]

def testColocationConstraintIntegrationRule() :
    problem = OneDWorkProblem()
    scipySolver = ScipyDiscretizationMinimizeWrapper(problem)
    n=4
    t = problem.CreateTimeRange(n)
    z = OptimizerStateForExpectedAnswerFor4LengthConstraint()
    expectedAns = getExpectedAnswersFor4LengthConstraint()
    for i in range(0, 4) :
        assert expectedAns[i]== scipySolver.CollocationConstraintIntegrationRule(t, z, i, 0), "X at index " + str(i)
        assert expectedAns[i+n]== scipySolver.CollocationConstraintIntegrationRule(t, z, i, 1), "v at index " + str(i)

    # these could be better, look for the specific exception and error message, probably even best as a separate function...
    with pytest.raises(Exception):
        scipySolver.CollocationConstraintIntegrationRule(t, z, 4, 1)
    with pytest.raises(Exception):
        scipySolver.CollocationConstraintIntegrationRule(t, z, 2, 2)


