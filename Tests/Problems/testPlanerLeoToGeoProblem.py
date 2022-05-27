import unittest
from PythonOptimizationWithNlp.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
from scipy.integrate import odeint
import numpy as np
import sympy as sy
import math
from scipy.optimize import fsolve
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
from PythonOptimizationWithNlp.ScaledSymbolicProblem import ScaledSymbolicProblem
from PythonOptimizationWithNlp.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from PythonOptimizationWithNlp.Numerical import ScipyCallbackCreators


class testPlanerLeoToGeoProblem(unittest.TestCase) :
    def testInitialization(self) :
        problem = ContinuousThrustCircularOrbitTransferProblem()
        self.assertEqual(problem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol), problem.TerminalCost, msg="terminal cost")
        self.assertEqual(4, len(problem.StateVariables), msg="count of state variables")
        self.assertEqual(1, len(problem.ControlVariables), msg="count of control variables")
        self.assertEqual(0, problem.UnIntegratedPathCost, msg="unintegrated path cost")
        self.assertEqual("t" , problem.TimeSymbol.name, msg="time symbol")
        self.assertEqual("t_f" , problem.TimeFinalSymbol.name, msg="time final symbol")
        self.assertEqual("t_0" , problem.TimeInitialSymbol.name, msg="time initial symbol")
        # thorough testing of EOM's and Boundary Conditions will be covered with solver/regression tests
        self.assertEqual(4, len(problem.EquationsOfMotion), msg="number of EOM's")
        self.assertEqual(2, len(problem.BoundaryConditions), msg="number of BCs")

    def testDifferentialTransversalityCondition(self) :
        problem = ContinuousThrustCircularOrbitTransferProblem()
        lambdas = SymbolicProblem.CreateCoVector(problem.StateVariables, 'L', problem.TimeFinalSymbol)
        hamiltonian = problem.CreateHamiltonian(lambdas)
        xversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, lambdas, 0.0) # not allowing final time to vary

        zeroedOutCondition =(xversality[0]-(sy.sqrt(problem.Mu)*lambdas[2]/(2*problem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol)**(3/2)) - lambdas[0] + 1)).expand().simplify()
        self.assertTrue((zeroedOutCondition).is_zero, msg="first xvers cond")
        self.assertTrue((xversality[1]+lambdas[-1]).is_zero, msg="lmd theta condition")
          