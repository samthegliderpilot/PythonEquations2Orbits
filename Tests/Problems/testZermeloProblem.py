import unittest
from pyeq2orb.Problems.ZermeloProblem import ZermelosProblem
from scipy.integrate import solve_ivp # type: ignore
import numpy as np
import sympy as sy
from pyeq2orb.Numerical import ScipyCallbackCreators
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions
import pyeq2orb as pe2o
import importlib
from typing import List
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs
from pyeq2orb.ProblemBase import Problem, ProblemVariable

class testZermelosProblem(unittest.TestCase):
    def testBasics(self) :
        prob = ZermelosProblem()
        assert prob != None   

    def testOptimalControlProcess(self):
        prob = ZermelosProblem()
        lambdas = Problem.CreateCostateVariables(prob.StateVariables)
        prob.AddCostateVariables(lambdas)

        expectedHamiltonian = lambdas[0]*prob.StateVariableDynamics[0] + lambdas[1]*prob.StateVariableDynamics[1]
        actualHamiltonian = prob.CreateHamiltonian()
        assert 0 == (expectedHamiltonian-actualHamiltonian).simplify()

        costate1Dynamics = expectedHamiltonian.diff(prob.StateVariables[0])
        costate2Dynamics = expectedHamiltonian.diff(prob.StateVariables[0])

        # this might be too simple to be a real test
        actualCostateDynamics = prob.CreateLambdaDotCondition(actualHamiltonian)
        assert 0 == (costate1Dynamics - actualCostateDynamics[0]).simplify()
        assert 0 == (costate2Dynamics - actualCostateDynamics[1]).simplify()