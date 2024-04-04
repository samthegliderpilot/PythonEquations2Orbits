from inspect import istraceback
import unittest
import sympy as sy
from pyeq2orb.ProblemBase import ProblemVariable, Problem
from pyeq2orb.Problems.OneDimensionalMinimalWorkProblem import OneDWorkSymbolicProblem
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from pyeq2orb.Symbolics.Vectors import Vector # type: ignore
from pyeq2orb import SafeSubs
import math
class testSymbolicOptimizerProblem(unittest.TestCase) :

    def testCreateCoVectorFromList(self) :
        prob = OneDWorkSymbolicProblem()
        expectedCoVector = [sy.Function(r'\lambda_{x}', real=True)(prob.TimeSymbol), sy.Function(r'\lambda_{v}', real=True)(prob.TimeSymbol)]
        actualCostateVector = Problem.CreateCoVector(prob.StateVariables, r'\lambda', prob.TimeSymbol)
        self.assertEqual(expectedCoVector[0], actualCostateVector[0], "lmd x")
        self.assertEqual(expectedCoVector[1], actualCostateVector[1], "lmd v")

    def testCreateCoVectorFromSymbol(self) :
        prob = OneDWorkSymbolicProblem()
        expectedCoVector = sy.Function(r'\lambda_{x}', real=True)(prob.TimeSymbol)
        actualCostateVector = Problem.CreateCoVector(prob.StateVariables[0], r'\lambda', prob.TimeSymbol)
        self.assertEqual(expectedCoVector, actualCostateVector, "lmd x")

    def testCreateCoVectorFromVector(self) :
        prob = OneDWorkSymbolicProblem()
        expectedCoVector = [sy.Function(r'\lambda_{x}', real=True)(prob.TimeSymbol), sy.Function(r'\lambda_{v}', real=True)(prob.TimeSymbol)]
        actualCostateVector = Problem.CreateCoVector(Vector.fromArray(prob.StateVariables), r'\lambda', prob.TimeSymbol)
        self.assertEqual(expectedCoVector[0], actualCostateVector[0,0], "lmd x")
        self.assertEqual(expectedCoVector[1], actualCostateVector[1,0], "lmd v")

    def testCreateHamiltonian(self) :
        prob = OneDWorkSymbolicProblem()
        lambdas = Problem.CreateCoVector(prob.StateVariables, 'L', prob.TimeSymbol)
        expectedHamiltonian = prob.UnIntegratedPathCost + lambdas[0]*prob.StateVariableDynamics[0] + lambdas[1]*prob.StateVariableDynamics[1] 
        actualHamiltonian = prob.CreateHamiltonian(lambdas)
        self.assertTrue((expectedHamiltonian-actualHamiltonian).simplify().expand().simplify().is_zero)

    def testCreateControlConditions(self) :
        prob = OneDWorkSymbolicProblem()
        lambdas = Problem.CreateCoVector(prob.StateVariables, 'L', prob.TimeSymbol)
        hamiltonian = prob.CreateHamiltonian(lambdas)
        expectedControlCondition = lambdas[1] + 2.0*prob.ControlVariables[0]
        controlExp = prob.CreateHamiltonianControlExpressions(hamiltonian)
        self.assertTrue((expectedControlCondition - controlExp[0,0]).is_zero)

    def testCreatingLambdaDotConditions(self) :
        prob = OneDWorkSymbolicProblem()
        lambdas = Problem.CreateCoVector(prob.StateVariables, 'L', prob.TimeSymbol)
        hamiltonian = prob.CreateHamiltonian(lambdas)
        expectedLambdaXDot = 0
        expectedLambdaVDot = -1*lambdas[0]
        actualLambdaDots = prob.CreateLambdaDotCondition(hamiltonian)
        self.assertTrue((expectedLambdaXDot-actualLambdaDots[0,0]).is_zero, msg="lmdXDot")
        self.assertTrue((expectedLambdaVDot-actualLambdaDots[1,0]).is_zero, msg="lmdVDot")

    def testCreatingDifferentialTransversalityCondition(self) :
        problem = ContinuousThrustCircularOrbitTransferProblem()
        lambdas = Problem.CreateCoVector(problem.StateVariables, 'L', problem.TimeFinalSymbol)
        hamiltonian = problem.CreateHamiltonian(lambdas)
        xversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, 0.0, lambdas) # not allowing final time to vary

        zeroedOutCondition =(xversality[0]-(sy.sqrt(problem.Mu)*lambdas[2]/(2*problem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol)**(3/2)) - lambdas[0] + 1)).expand().simplify()
        self.assertTrue((zeroedOutCondition).is_zero, msg="first xvers cond")
        self.assertTrue((xversality[1]+lambdas[-1]).is_zero, msg="lmd theta condition")

    def testCreatingAugmentedTransversalityCondition(self) :
        problem = ContinuousThrustCircularOrbitTransferProblem()
        lambdas = Problem.CreateCoVector(problem.StateVariables, 'l', problem.TimeFinalSymbol)
        l_r = lambdas[0]
        l_u = lambdas[1]
        l_v = lambdas[2]
        l_theta = lambdas[3]
        mu = problem.Mu
        r = problem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol)
        b1=sy.Symbol('b1')
        b2=sy.Symbol('b2')
        aug = [b1,b2 ]
        xversality = problem.TransversalityConditionsByAugmentation(aug, lambdas)
        print(xversality)

        firstZeroExpression = (xversality[0]-(-sy.sqrt(mu)*b2/(2*r**(3/2)) + l_r - 1)).expand().simplify()
        print(firstZeroExpression)
        secondsZeroExp = xversality[1]-(-b1 + l_u).expand().simplify()
        thirdZeroExp = xversality[2]-(-b2 + l_v).expand().simplify()
        fourthZeroExp = xversality[3]-(l_theta).expand().simplify()

        self.assertTrue(firstZeroExpression.is_zero, msg="first")
        self.assertTrue(secondsZeroExp.is_zero, msg="second")
        self.assertTrue(thirdZeroExp.is_zero, msg="third")
        self.assertTrue(fourthZeroExp.is_zero, msg="fourth")


    def testCreateEquationOfMotionsAsEquations(self):
        prob = OneDWorkSymbolicProblem()
        eqsOfMotion = prob.CreateEquationOfMotionsAsEquations()
        self.assertEqual(2, len(eqsOfMotion), msg="2 equations returned back")
        self.assertEqual(prob.StateVariables[0].diff(prob.TimeSymbol), eqsOfMotion[0].lhs, msg="lhs of first eom")
        self.assertEqual(prob.StateVariableDynamics[0], eqsOfMotion[0].rhs, msg="rhs of first eom")

        self.assertEqual(prob.StateVariables[1].diff(prob.TimeSymbol), eqsOfMotion[1].lhs, msg="lhs of second eom")
        self.assertEqual(prob.StateVariableDynamics[1], eqsOfMotion[1].rhs, msg="rhs of second eom")

    def testCreateCostFunctionAsEquation(self) :
        prob = OneDWorkSymbolicProblem()
        costFunction = prob.CreateCostFunctionAsEquation()
        expectedRhs = sy.integrate(prob.UnIntegratedPathCost, (prob.TimeSymbol, prob.TimeInitialSymbol, prob.TimeFinalSymbol))
        self.assertEqual(sy.Symbol('J'), costFunction.lhs, msg="default lhs")
        self.assertEqual(expectedRhs, costFunction.rhs, msg="correct rhs")

        someOtherLhs = sy.Symbol("P")
        costFunction = prob.CreateCostFunctionAsEquation(someOtherLhs)
        self.assertEqual(someOtherLhs, costFunction.lhs, msg="custom lhs")
        self.assertEqual(expectedRhs, costFunction.rhs, msg="correct rhs 2")

    def testEquationsOfMotionInMatrixForm(self) :
        prob = OneDWorkSymbolicProblem()
        eqsOfMotion = prob.EquationsOfMotionInMatrixForm()
        self.assertEqual(prob.StateVariableDynamics[0], eqsOfMotion[0,0], msg="first eom")
        self.assertEqual(prob.StateVariableDynamics[1], eqsOfMotion[1,0], msg="second eom")

    def testStateVariablesInMatrixForm(self) :
        prob = OneDWorkSymbolicProblem()
        stateAsMatrix = prob.StateVariablesInMatrixForm()
        self.assertEqual(prob.StateVariables[0], stateAsMatrix[0,0], msg="first state variable")
        self.assertEqual(prob.StateVariables[1], stateAsMatrix[1,0], msg="second state variable")

    def testControlVariablesInMatrixForm(self) :
        prob = OneDWorkSymbolicProblem()
        stateAsMatrix = prob.ControlVariablesInMatrixForm()
        self.assertEqual(prob.ControlVariables[0], stateAsMatrix[0,0], msg="first control variable")

    def testAddInitialValuesToDictionary(self) :
        prob = OneDWorkSymbolicProblem()
        subsDict = {}
        initialState = [0.2, 0.3, 0.4, 0.5]
        prob.AddInitialValuesToDictionary(subsDict, initialState)
        self.assertEqual(2, len(subsDict), msg="only 2 items in subs dict")
        initialSvs = prob.CreateVariablesAtTime0(prob.StateVariables)
        self.assertEqual(initialState[0], subsDict[initialSvs[0]], msg="x was added when no lambdas")
        self.assertEqual(initialState[1], subsDict[initialSvs[1]], msg="vx was added when no lambdas")

        lambdas = prob.CreateCoVector(prob.StateVariables, "lmd", prob.TimeInitialSymbol)
        initialState = [1.2, 1.3, 1.4, 1.5]
        prob.AddInitialValuesToDictionary(subsDict, initialState, lambdas)
        self.assertEqual(4, len(subsDict), msg="4 items in subs dict")
        self.assertEqual(initialState[0], subsDict[initialSvs[0]], msg="x was added")
        self.assertEqual(initialState[1], subsDict[initialSvs[1]], msg="vx was added")        
        self.assertEqual(initialState[2], subsDict[lambdas[0]], msg="lmd x was added")
        self.assertEqual(initialState[3], subsDict[lambdas[1]], msg="lmd vx was added")        

    def testSafeSubs(self) :
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        b = sy.Symbol('c')
        expr = a+b
        expr2 = a*a
        self.assertEqual(2, SafeSubs(2, {a:b}), msg="int")
        self.assertEqual(2.0, SafeSubs(2.0, {a:b}), msg="float")
        self.assertEqual(b, SafeSubs(a, {a:b}), msg="just a symbol")
        self.assertEqual(2*b, SafeSubs(expr, {a:b}), msg="an expression")
        self.assertEqual([2*b, b**2], SafeSubs([expr, expr2], {a:b}), msg="list of expressions")

    def testEvaluateHamiltonianAndItsFirstTwoDerivatives(self) :
        problem = ContinuousThrustCircularOrbitTransferProblem()
        lambdas = Problem.CreateCoVector(problem.StateVariables, 'l', problem.TimeSymbol)
        problem._costateElements.extend([ProblemVariable(x, None) for x in lambdas])
        problem.StateVariableDynamics.extend([0.0,0.0,0.0,0.0])
        a = sy.Symbol('a')
        fakeHamiltonian = 3.0*sy.cos(problem.ControlVariables[0] * problem.StateVariables[0]*2.0*a)
        answer = {problem.StateVariables[0] : [0.0, math.pi/8.0], problem.StateVariables[1] : [0.0, 0.0], problem.StateVariables[2] : [0.0, 0.0], problem.StateVariables[3] : [0.0, 0.0] }
        answer[problem.CostateSymbols[0]] = [1.0, 1.0]
        answer[problem.CostateSymbols[1]] = [0.0, 0.0]
        answer[problem.CostateSymbols[2]] = [0.0, 0.0]
        answer[problem.CostateSymbols[3]] = [0.0, 0.0]
        [h, dh, ddh] = problem.EvaluateHamiltonianAndItsFirstTwoDerivatives(answer, [0.0, 1.0], fakeHamiltonian, {problem.ControlVariables[0]: (problem.CostateSymbols[0]+problem.CostateSymbols[1])}, {a: 2.0})
        self.assertAlmostEqual(3.0, h[0], places=10,msg="h0")
        self.assertAlmostEqual(0.0, h[1], places=10, msg="h1")
        self.assertAlmostEqual(0.0, dh[0], places=10,msg="dh0")        
        self.assertAlmostEqual(-4.71238898038469, dh[1], places=10, msg="dh1")
        self.assertAlmostEqual(0.0, ddh[0], places=10,msg="ddh0")
        self.assertAlmostEqual(0.0, ddh[1], places=10, msg="dh1")

    def testDefaultDescaleResults(self) :
        originalProblem = ContinuousThrustCircularOrbitTransferProblem()        
        barVars =  Problem.CreateBarVariables(originalProblem.StateVariables, originalProblem.TimeSymbol)
        scalingExpressions = {originalProblem.StateVariables[0]: barVars[0],originalProblem.StateVariables[1]: barVars[1],originalProblem.StateVariables[2]: barVars[2],originalProblem.StateVariables[3]: barVars[3]}
        problem = originalProblem.ScaleStateVariables(barVars, scalingExpressions)
        self.assertEqual(scalingExpressions, problem.DescaleResults(scalingExpressions))



    def testFullySymbolicProblem(self):
        problem = Problem()
        problem.TimeSymbol = sy.Symbol('t', real=True)
        problem.TimeInitialSymbol = sy.Symbol('t_0', real=True)
        problem.TimeFinalSymbol = sy.Symbol('t_f', real=True)
        x = sy.Function('x', real=True)(problem.TimeSymbol)
        u = sy.Function('u', real=True)(problem.TimeSymbol)

        dxdt = sy.Symbol('v_x', real=True) * u

        problem.AddStateVariable(ProblemVariable(x, dxdt))
        problem.ControlVariables.append(u)

        problem.AddStateVariable(ProblemVariable(sy.Function(r'\lambda{x}', real=True)(problem.TimeFinalSymbol), 1))

        bc = x.subs(problem.TimeSymbol, problem.TimeFinalSymbol) - 5
        problem.BoundaryConditions.append(bc)



