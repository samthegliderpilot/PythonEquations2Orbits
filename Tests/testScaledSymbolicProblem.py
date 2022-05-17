import unittest
import sympy as sy
from PythonOptimizationWithNlp.ScaledSymbolicProblem import ScaledSymbolicProblem
from PythonOptimizationWithNlp.ScaledSymbolicProblem import SymbolicProblem
from PythonOptimizationWithNlp.Problems.OneDimensionalMinimalWorkProblem import OneDWorkSymbolicProblem
from PythonOptimizationWithNlp.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem

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
        baseProblem = ContinuousThrustCircularOrbitTransferProblem()
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


    def testCreatingDifferentialTransversalityCondition(self) :
        orgProblem = ContinuousThrustCircularOrbitTransferProblem()
        mu = orgProblem.Mu
        t = sy.Symbol('t')
        newSvs = [sy.Function('rs')(t), sy.Function('rs')(t), sy.Function('vs')(t), sy.Function('lons')(t)]
        subs = {orgProblem.StateVariables[0]: 4.0, orgProblem.StateVariables[1]: 3.0, orgProblem.StateVariables[2]: 5.0, orgProblem.StateVariables[3]: 7.0, }
        problem = ScaledSymbolicProblem(orgProblem, newSvs, subs, False)
        lambdas = SymbolicProblem.CreateCoVector(problem.StateVariables, 'L', problem.TimeFinalSymbol)
        r = problem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol)
        l_r = lambdas[0]
        l_v = lambdas[2]
        hamiltonian = problem.CreateHamiltonian(lambdas)
        xversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, lambdas, 0.0) # not allowing final time to vary

        zeroedOutCondition =(xversality[0]-(sy.sqrt(mu)*l_v/(2*(r*4.0)**(3/2)) - l_r + 1)).expand().simplify()
        self.assertTrue((zeroedOutCondition).is_zero, msg="first xvers cond")
        self.assertTrue((xversality[1]+lambdas[-1]).is_zero, msg="lmd theta condition")

    def testCreatingAugmentedTransversalityCondition(self) :
        orgProblem = ContinuousThrustCircularOrbitTransferProblem()
        t = sy.Symbol('t')
        newSvs = [sy.Function('rs')(t), sy.Function('rs')(t), sy.Function('vs')(t), sy.Function('lons')(t)]
        subs = {orgProblem.StateVariables[0]: 4.0, orgProblem.StateVariables[1]: 3.0, orgProblem.StateVariables[2]: 5.0, orgProblem.StateVariables[3]: 7.0, }
        problem = ScaledSymbolicProblem(orgProblem, newSvs, subs, False)
        lambdas = SymbolicProblem.CreateCoVector(problem.StateVariables, 'l', problem.TimeFinalSymbol)
        l_r = lambdas[0]
        l_u = lambdas[1]
        l_v = lambdas[2]
        l_theta = lambdas[3]
        mu = orgProblem.Mu
        r = problem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol)
        b1=sy.Symbol('b1')
        b2=sy.Symbol('b2')
        aug = [b1,b2 ]
        xversality = problem.TransversalityConditionsByAugmentation(lambdas, aug)
        print(xversality)

        firstZeroExpression = (xversality[0]-(-sy.sqrt(mu)*b2/(2*(r*4.0)**(3/2)) + l_r - 1)).expand().simplify()
        print(firstZeroExpression)
        secondsZeroExp = xversality[1]-(-b1 + l_u).expand().simplify()
        thirdZeroExp = xversality[2]-(-b2 + l_v).expand().simplify()
        fourthZeroExp = xversality[3]-(l_theta).expand().simplify()

        self.assertTrue(firstZeroExpression.is_zero, msg="first")
        self.assertTrue(secondsZeroExp.is_zero, msg="second")
        self.assertTrue(thirdZeroExp.is_zero, msg="third")
        self.assertTrue(fourthZeroExp.is_zero, msg="fourth")