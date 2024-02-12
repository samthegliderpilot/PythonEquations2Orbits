# import unittest

# from sympy import numer
# from pyeq2orb.NumericalProblemFromSymbolic import NumericalProblemFromSymbolicProblem
# from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem

# #TODO: This unit tests need significant improvements

# class testNumericalProblemFromSymbolicProblem(unittest.TestCase) :
#     def testBasicOperation(self) :
#         symbolicProblem = ContinuousThrustCircularOrbitTransferProblem()
#         constantsSubsDict = symbolicProblem.SubstitutionDictionary
#         constantsSubsDict[symbolicProblem.Isp] = 1
#         constantsSubsDict[symbolicProblem.MassInitial] = 2
#         constantsSubsDict[symbolicProblem.Gravity] = 3
#         constantsSubsDict[symbolicProblem.Mu]= 4
#         constantsSubsDict[symbolicProblem.Thrust] = 5
#         numericalProblem = NumericalProblemFromSymbolicProblem(symbolicProblem, None)

#         self.assertEqual(2, len(numericalProblem.BoundaryConditionCallbacks), msg="2 boundary conditions")
#         self.assertIsNotNone(numericalProblem.SingleEquationOfMotionWithTInState([1, 1,2,3,4,5], 0), msg="first eoms with t")
#         self.assertIsNotNone(numericalProblem.SingleEquationOfMotionWithTInState([1, 1,2,3,4,5], 1), msg="second eoms with t")
#         self.assertIsNotNone(numericalProblem.SingleEquationOfMotionWithTInState([1, 1,2,3,4,5], 2), msg="third eoms with t")
#         self.assertIsNotNone(numericalProblem.SingleEquationOfMotionWithTInState([1, 1,2,3,4,5], 3), msg="fourth eoms with t")

#         self.assertIsNotNone(numericalProblem.SingleEquationOfMotion(1, [1,2,3,4,5], 0), msg="first eoms")
#         self.assertIsNotNone(numericalProblem.SingleEquationOfMotion(1, [1,2,3,4,5], 1), msg="second eoms")
#         self.assertIsNotNone(numericalProblem.SingleEquationOfMotion(1, [1,2,3,4,5], 2), msg="third eoms")
#         self.assertIsNotNone(numericalProblem.SingleEquationOfMotion(1, [1,2,3,4,5], 3), msg="fourth eoms")
        
#         self.assertEqual(4, len(numericalProblem.EquationOfMotion(1, [1,2,3,4,5])), msg="4 eoms")

#         self.assertIsNotNone(numericalProblem.TerminalCost(1.0, [1,2,3,4,5]), msg="terminal cost")
#         self.assertIsNotNone(numericalProblem.UnIntegratedPathCost(1, [1,2,3,4,5]), msg="path cost")

