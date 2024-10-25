import pytest
from pyeq2orb.Problems.OneDimensionalMinimalWorkProblem import OneDWorkProblem # type: ignore


def testNumberOfStateVariables() :
    oneDWorkProblem = OneDWorkProblem()
    numOfStateVariables = oneDWorkProblem.NumberOfStateVariables
    assert numOfStateVariables == 2

def testNumberOfControlVariables() :
    oneDWorkProblem = OneDWorkProblem()
    numberOfControlStates = oneDWorkProblem.NumberOfControlVariables
    assert numberOfControlStates == 1

def testNumberOfOptimizerVariables() :
    oneDWorkProblem = OneDWorkProblem()
    numberOfOptimizerStates = oneDWorkProblem.NumberOfOptimizerStateVariables
    assert numberOfOptimizerStates == 3

def testCreateTimeRange() :
    oneDWorkProblem = OneDWorkProblem()
    expectedT = [0.0, 0.25, 0.5, 0.75, 1.0]
    t = oneDWorkProblem.CreateTimeRange(len(expectedT)-1)
    assert len(t) == len(expectedT), "t arrays need to be same size"
    for i in range(0, len(t)) :
        assert expectedT[i] == t[i], "T at index " + str(i)



