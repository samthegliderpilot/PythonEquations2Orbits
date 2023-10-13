import unittest

from pytest import approx
from pyeq2orb.Solvers.ScipyDiscretizationMinimizeWrapper import ScipyDiscretizationMinimizeWrapper
from pyeq2orb.Problems.OneDimensionalMinimalWorkProblem import OneDWorkProblem, AnalyticalAnswerToProblem
from scipy.interpolate import interp1d # type: ignore
import numpy as np

class testScipyWrapper(unittest.TestCase) :

    def testSolution(self) :
        oneDWorkProblem = OneDWorkProblem() # picking a larger steps size to get closer to ideal (but still not exact)
        n= 20
        t=oneDWorkProblem.CreateTimeRange(n)

        truth = AnalyticalAnswerToProblem()
        evaledAns = truth.EvaluateAnswer(oneDWorkProblem, t)        
        interpTruth = interp1d(t, evaledAns[oneDWorkProblem.State[0]], kind='cubic')

        scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)
        ans = scipySolver.ScipyOptimize(n)
        
        self.assertIsNotNone(ans, msg="answer is not None")
        
        interpAnswer = interp1d(t, ans.x[0:n+1], kind='cubic')
        for i in np.linspace(0,1,11) :
            #assert abs(interpTruth(i) - interpAnswer(i)) <= 0.001 
            # this line can be useful for debugging, leaving it in
            print(str(i) + " " + str(interpTruth(i)) + " " + str(interpAnswer(i)) + " " + str(interpTruth(i)-interpAnswer(i)))

    def testCreateDiscretizedInitialGuess(self) :
        oneDWorkProblem = OneDWorkProblem()
        scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)
        initialGuess = scipySolver.CreateDiscretizedInitialGuess(oneDWorkProblem.CreateTimeRange(4))
        expected = [0.0, 0.25, 0.5, 0.75, 1.0,   0.0, 1.0, 1.0, -1.0, -1.0,   0.0, 0.1, 0.1, -0.1, 0.0]
        for i in range(0, len(expected))  :
            self.assertEqual(expected[i], initialGuess[i], msg="value at " + str(i))

    def testConvertScipyOptimizerOutputToDictionary(self) :
        oneDWorkProblem = OneDWorkProblem()
        scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)
        ans = scipySolver.ScipyOptimize(4)
        dictAnswer =scipySolver.ConvertScipyOptimizerOutputToDictionary(ans)
        self.assertEqual(3, len(dictAnswer), msg="make sure we get 3 items back")
        self.assertEqual(5, len(dictAnswer[oneDWorkProblem.State[0]]), msg="check x is there and is the right length")
        self.assertEqual(5, len(dictAnswer[oneDWorkProblem.State[1]]), msg="check v is there and is the right length")
        self.assertEqual(5, len(dictAnswer[oneDWorkProblem.Control[0]]), msg="check u is there and is the right length")
        for i in range(0, 5) :
            self.assertEqual(ans.x[i], dictAnswer[oneDWorkProblem.State[0]][i], msg="check x at index " + str(i))
            self.assertEqual(ans.x[5+i], dictAnswer[oneDWorkProblem.State[1]][i], msg="check x at index " + str(i))
            self.assertEqual(ans.x[10+i], dictAnswer[oneDWorkProblem.Control[0]][i], msg="check x at index " + str(i))

    def ConvertDiscretizedStateToDict(self) :
        #TODO: I don't like the copy/paste of the above test here, consider ways to really test this
        oneDWorkProblem = OneDWorkProblem(4)
        scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)
        ans = scipySolver.ScipyOptimize()
        dictAnswer =scipySolver.ConvertDiscretizedStateToDict(ans.x)
        self.assertEqual(3, len(dictAnswer), msg="make sure we get 3 items back")
        self.assertEqual(5, len(dictAnswer[oneDWorkProblem.State[0]]), msg="check x is there and is the right length")
        self.assertEqual(5, len(dictAnswer[oneDWorkProblem.State[1]]), msg="check v is there and is the right length")
        self.assertEqual(5, len(dictAnswer[oneDWorkProblem.Control[0]]), msg="check u is there and is the right length")
        for i in range(0, 5) :
            self.assertEqual(ans.x[i], dictAnswer[oneDWorkProblem.State[0]][i], msg="check x at index " + str(i))
            self.assertEqual(ans.x[5+i], dictAnswer[oneDWorkProblem.State[1]][i], msg="check x at index " + str(i))
            self.assertEqual(ans.x[10+i], dictAnswer[oneDWorkProblem.Control[0]][i], msg="check x at index " + str(i))
        

    def testEquationOfMotionFromOptimizationState(self) :
        problem = OneDWorkProblem()
        scipySolver = ScipyDiscretizationMinimizeWrapper(problem)
        equationOfMotionValue = scipySolver.EquationOfMotionInTermsOfOptimizerState(2, [2,12, 22])
        assert 12==equationOfMotionValue[0]
        assert 22==equationOfMotionValue[1]

    def testInitialAndFinalStateConstraints(self) :
        # TODO: This is a fairly weak test due to the nature of OneDWorkProblem, update when a better 
        # problem is available
        problem = OneDWorkProblem()
        scipySolver = ScipyDiscretizationMinimizeWrapper(problem)
        constraints = scipySolver.CreateIndividualBoundaryValueConstraintCallbacks()
        self.assertEqual(0, len(constraints), msg="number of additional constriants for this problem")
        state = [1.0, 2.0, 3.0, 4.0,   5.0, 6.0, 7.0, 8.0,   9.0, 10.0, 11.0, 12.0]
        constraints = scipySolver.CreateInitialStateConstraintCallbacks()
        self.assertEqual(2, len(constraints), msg="2 initial state constraints returned")
        self.assertEqual(0.0-state[0], constraints[0]['fun'](state), "x_0")
        self.assertEqual(0.0-state[4], constraints[1]['fun'](state), "v_0")
        constraints = scipySolver.CreateFinalStateConstraintCallbacks()
        self.assertEqual(2, len(constraints), msg="2 final state constraints returned")
        self.assertEqual(1.0-state[3], constraints[0]['fun'](state), "x_f")        
        self.assertEqual(0.0-state[7], constraints[1]['fun'](state), "v_f")

    def testCostFunctionInTermsOfZ(self) :
        oneDWorkProblem = OneDWorkProblem()
        scipySolver = ScipyDiscretizationMinimizeWrapper(oneDWorkProblem)

        cost = scipySolver.CostFunctionInTermsOfZ(oneDWorkProblem.CreateTimeRange(3), [1.0, 1.0, 1.0, 1.0,   0.1, 0.1, 0.1, 0.1,   0.25, 0.25, 1.0, 1.0 ])
        self.assertAlmostEqual(0.5052083333333334 , cost, delta=0.000001, msg="cost")

    def testCreateIndividualColocationConstraints(self) :
        #TODO: Incomplete
        problem = OneDWorkProblem()
        scipySolver = ScipyDiscretizationMinimizeWrapper(problem)
        t = problem.CreateTimeRange(4)

        constraints = scipySolver.CreateIndividualCollocationConstraints(t)
        self.assertEqual(8, len(constraints), "8 constraints back")
        z = self.OptimizerStateForExpectedAnswerFor4LengthConstraint()
        expectedAns = self.getExpectedAnswersFor4LengthConstraint()

        for i in range(0, len(constraints)) :
            self.assertEqual(expectedAns[i], constraints[i]['fun'](z), msg="index " + str(i))

    def testGetOptimizerStateAtIndex(self) :
        problem = OneDWorkProblem()
        scipySolver = ScipyDiscretizationMinimizeWrapper(problem)
        sampleOptimizerState = [0,1,2,3,4,10,11,12,13,14,20,21,22,23,24]
        returnedState = scipySolver.GetOptimizerStateAtIndex(sampleOptimizerState, 2)
        assert returnedState[0] == 2
        assert returnedState[1] == 12
        assert returnedState[2] == 22        

    @staticmethod 
    def OptimizerStateForExpectedAnswerFor4LengthConstraint() :
        return [0,1,3,6,4,  10,13,17,22,29, 33,43,55,67,80]
    @staticmethod
    def getExpectedAnswersFor4LengthConstraint() :
        # the first few were calculated by hand, the others are answers from an earlier run (making using this a regression test)
        return [1.875, 1.75, 1.875, 8.375, 6.5, 8.25, 10.25, 11.375]

    def testColocationConstraintIntegrationRule(self) :
        problem = OneDWorkProblem()
        scipySolver = ScipyDiscretizationMinimizeWrapper(problem)
        n=4
        t = problem.CreateTimeRange(n)
        z = self.OptimizerStateForExpectedAnswerFor4LengthConstraint()
        expectedAns = self.getExpectedAnswersFor4LengthConstraint()
        for i in range(0, 4) :
            self.assertEqual(expectedAns[i], scipySolver.CollocationConstraintIntegrationRule(t, z, i, 0), msg="X at index " + str(i))
            self.assertEqual(expectedAns[i+n], scipySolver.CollocationConstraintIntegrationRule(t, z, i, 1), msg="v at index " + str(i))

        # these could be better, look for the specific exception and error message, probably even best as a separate function...
        self.assertRaises(Exception, lambda : scipySolver.CollocationConstraintIntegrationRule(t, z, 4, 1), "can't go beyond known time")
        self.assertRaises(Exception, lambda : scipySolver.CollocationConstraintIntegrationRule(t, z, 2, 2), "only 2 state values to test")


