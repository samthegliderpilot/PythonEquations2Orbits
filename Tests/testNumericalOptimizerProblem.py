import unittest
from PythonOptimizationWithNlp.NumericalOptimizerProblem import NumericalOptimizerProblemBase
from PythonOptimizationWithNlp.Problems.OneDimensionalMinimalWorkProblem import OneDWorkProblem

class testNumericalOptimizerProblemBase(unittest.TestCase) :

    def testNumberOfStateVariables(self) :
        oneDWorkProblem = OneDWorkProblem()
        numOfStateVariables = oneDWorkProblem.NumberOfStateVariables
        assert numOfStateVariables == 2

    def testNumberOfControlVariables(self) :
        oneDWorkProblem = OneDWorkProblem()
        numberOfControlStates = oneDWorkProblem.NumberOfControlVariables
        assert numberOfControlStates == 1

    def testNumberOfOptimizerVariables(self) :
        oneDWorkProblem = OneDWorkProblem()
        numberOfOptimizerStates = oneDWorkProblem.NumberOfOptimizerStateVariables
        assert numberOfOptimizerStates == 3

    def testCreateTimeRange(self) :
        oneDWorkProblem = OneDWorkProblem()
        expectedT = [0.0, 0.25, 0.5, 0.75, 1.0]
        t = oneDWorkProblem.CreateTimeRange(len(expectedT)-1)
        self.assertEqual(len(t), len(expectedT), msg="t arrays need to be same size")
        for i in range(0, len(t)) :
            self.assertEqual(expectedT[i], t[i], msg="T at index " + str(i))
    


