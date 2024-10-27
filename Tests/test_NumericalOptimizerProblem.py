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

