import unittest
from PythonOptimizationWithNlp.Problems.ContinuousThrustCircularOrbitTransferProblem import PlanerLeoToGeoProblem

class testPlanerLeoToGeoProblem(unittest.TestCase) :
    def testInitialization(self) :
        problem = PlanerLeoToGeoProblem()
        self.assertEqual(-1.0*problem.StateVariables[0], problem.TerminalCost, msg="terminal cost")
        self.assertEqual(4, len(problem.StateVariables), msg="count of state variables")
        self.assertEqual(1, len(problem.ControlVariables), msg="count of control variables")
        self.assertEqual(0, problem.UnIntegratedPathCost, msg="unintegrated path cost")
        self.assertEqual("t" , problem.TimeSymbol.name, msg="time symbol")
        self.assertEqual("t_f" , problem.TimeFinalSymbol.name, msg="time final symbol")
        self.assertEqual("t_0" , problem.Time0Symbol.name, msg="time initial symbol")
        self.assertEqual(5, len(problem.ConstantSymbols), msg="number of constants")
        self.assertEqual(0, len(problem.PathConstraints), msg="number of path constraints")
        # thorough testing of EOM's and Boundary Conditions will be covered with solver/regression tests
        self.assertEqual(4, len(problem.EquationsOfMotion), msg="number of EOM's")
        self.assertEqual(2, len(problem.FinalBoundaryConditions), msg="number of BCs")