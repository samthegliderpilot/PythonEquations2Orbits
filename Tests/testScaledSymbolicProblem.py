import unittest
import sympy as sy
from PythonOptimizationWithNlp.ScaledSymbolicProblem import ScaledSymbolicProblem
from PythonOptimizationWithNlp.Problems.OneDimensionalMinimalWorkProblem import OneDWorkSymbolicProblem
from PythonOptimizationWithNlp.Problems.ContinuousThrustCircularOrbitTransferProblem import PlanerLeoToGeoProblem

class testScaledSymbolicProblem(unittest.TestCase) :

    def testVariableScaling(self) :
        baseProblem = OneDWorkSymbolicProblem()
        newSvs = ScaledSymbolicProblem.CreateBarVariables(baseProblem.StateVariables, baseProblem.TimeSymbol)
        scalingDict = {}
        scalingDict[baseProblem.StateVariables[0]]=2
        scalingDict[baseProblem.StateVariables[1]]=3 
        outerProblem = ScaledSymbolicProblem(baseProblem, newSvs, scalingDict, False)
        firstEomValue = outerProblem.EquationsOfMotion[outerProblem.StateVariables[0]].subs({outerProblem.StateVariables[0]: 1.5, outerProblem.StateVariables[1]: 0.4})
        secondEomValue=outerProblem.EquationsOfMotion[outerProblem.StateVariables[1]].subs({outerProblem.ControlVariables[0]: 1.6})
        self.assertEqual(3.0*0.4/2.0, firstEomValue, msg="first eom evaluated")
        self.assertEqual(1.6/3.0, secondEomValue, msg="second eom evaluated")


    def testScalingComplicatedProblem(self) :
        baseProblem = PlanerLeoToGeoProblem()
        newSvs = ScaledSymbolicProblem.CreateBarVariables(baseProblem.StateVariables, baseProblem.TimeSymbol)
        r0=sy.Symbol('r_0')
        u0=sy.Symbol('u_0')
        v0=sy.Symbol('v_0')
        lon0= sy.Symbol('lon_0')
        scalingDict = {}
        scalingDict[baseProblem.StateVariables[0]]=r0
        scalingDict[baseProblem.StateVariables[1]]=u0
        scalingDict[baseProblem.StateVariables[2]]=v0
        scalingDict[baseProblem.StateVariables[3]]=lon0
        outerProblem = ScaledSymbolicProblem(baseProblem, newSvs, scalingDict, False)
        firstEomValue = outerProblem.EquationsOfMotion[outerProblem.StateVariables[0]].subs({outerProblem.StateVariables[1]: 1.5})
        secondEomValue=outerProblem.EquationsOfMotion[outerProblem.StateVariables[1]].subs({outerProblem.ControlVariables[0]: 1.6})
        self.assertEqual(u0*1.5/r0, firstEomValue, msg="first eom evaluated")
        

        #self.assertEqual(1.6/3.0, secondEomValue, msg="second eom evaluated")    


