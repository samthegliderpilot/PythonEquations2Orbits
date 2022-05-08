import unittest
from PythonOptimizationWithNlp.Problems.ContinuousThrustCircularOrbitTransferProblem import PlanerLeoToGeoProblem

class testPlanerLeoToGeoProblem(unittest.TestCase) :
    def testInitialization(self) :
        problem = PlanerLeoToGeoProblem()
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