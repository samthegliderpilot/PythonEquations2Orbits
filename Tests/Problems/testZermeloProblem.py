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
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelper, OdeLambdifyHelperWithBoundaryConditions

class testZermelosProblem(unittest.TestCase):

    @staticmethod
    def createZermelosProblem() :
        theProblem = Problem()
        t = sy.Symbol('t')
        t0 = sy.Symbol('t_0')
        tf = sy.Symbol('t_f')
        v = sy.Symbol('V')
        theta = sy.Function(r'\theta')(t)
        x = ProblemVariable(sy.Function('x')(t), v*sy.cos(theta))
        y = ProblemVariable(sy.Function('y')(t), v*sy.sin(theta))

        xfDesiredValue = sy.Symbol('x_f')
        yfDesiredValue = sy.Symbol('y_f')
        xfBc = x.Element.subs(t, tf)-xfDesiredValue
        yfBc = y.Element.subs(t, tf)-yfDesiredValue

        theProblem.AddStateVariable(x)
        theProblem.AddStateVariable(y)

        theProblem.BoundaryConditions.append(xfBc)
        theProblem.BoundaryConditions.append(yfBc)

        theProblem._terminalCost = tf
        theProblem._timeFinalSymbol = tf
        theProblem._timeInitialSymbol = t0
        theProblem._timeSymbol = t     
        theProblem.ControlSymbols.append(theta)

        return theProblem   

    @staticmethod
    def solveForTanThetaZeroV(problem: Problem, hamiltonian : sy.Expr) -> sy.Expr:
        theta = problem.ControlSymbols[0]
        h_u = hamiltonian.diff(theta)
        sinTheta = sy.solve(h_u, sy.sin(theta))
        tanTheta = sinTheta/sy.cos(theta)
        return tanTheta

    def testOptimalControlProcess(self):
        problem = testZermelosProblem.createZermelosProblem()
        lambdas = Problem.CreateCostateVariables(prob.StateSymbols)
        prob.AddCostateSymbols(lambdas)

        expectedHamiltonian = lambdas[0]*prob.StateVariableDynamics[0] + lambdas[1]*prob.StateVariableDynamics[1]
        actualHamiltonian = prob.CreateHamiltonian()
        assert 0 == (expectedHamiltonian-actualHamiltonian).simplify()

        costate1Dynamics = expectedHamiltonian.diff(prob.StateSymbols[0])
        costate2Dynamics = expectedHamiltonian.diff(prob.StateSymbols[0])

        # this might be too simple to be a real test
        actualCostateDynamics = prob.CreateLambdaDotCondition(actualHamiltonian)
        assert 0 == (costate1Dynamics - actualCostateDynamics[0]).simplify()
        assert 0 == (costate2Dynamics - actualCostateDynamics[1]).simplify()


        theta = problem.ControlSymbols[0]
        h_u = actualHamiltonian.diff(theta)
        sinTheta = sy.solve(h_u, sy.sin(theta))
        cosTheta = sy.solve(h_u, sy.cos(theta))

        #for eom in prob.StateSymbols

        expectedControlExpression = lambdas[1]/lambdas[0]
        actualControlExpression = prob.CreateControlExpressionsFromHamiltonian(actualHamiltonian, prob.ControlSymbols)
        assert 1 == len(actualControlExpression)
        assert (expectedControlExpression-sy.tan(actualControlExpression[0])).simplify() == 0



    def testNumericalEvaluation(self):
        prob = testZermelosProblem.createZermelosProblem()
        
