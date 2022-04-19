from inspect import istraceback
import unittest
import sympy as sy
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
from PythonOptimizationWithNlp.Problems.OneDimensionalMinimalWorkProblem import OneDWorkSymbolicProblem
from PythonOptimizationWithNlp.Symbolics.Vectors import Vector
class testSymbolicOptimizerProblem(unittest.TestCase) :

    def testCreateCoVectorFromList(self) :
        prob = OneDWorkSymbolicProblem()
        expectedCovector = [sy.Function(r'\lambda_{x}', real=True)(prob.TimeSymbol), sy.Function(r'\lambda_{v}', real=True)(prob.TimeSymbol)]
        actualCostateVector = SymbolicProblem.CreateCoVector(prob.StateVariables, r'\lambda', prob.TimeSymbol)
        self.assertEqual(expectedCovector[0], actualCostateVector[0], "lmd x")
        self.assertEqual(expectedCovector[1], actualCostateVector[1], "lmd v")

    def testCreateCoVectorFromSymbol(self) :
        prob = OneDWorkSymbolicProblem()
        expectedCovector = sy.Function(r'\lambda_{x}', real=True)(prob.TimeSymbol)
        actualCostateVector = SymbolicProblem.CreateCoVector(prob.StateVariables[0], r'\lambda', prob.TimeSymbol)
        self.assertEqual(expectedCovector, actualCostateVector, "lmd x")

    def testCreateCoVectorFromVector(self) :
        prob = OneDWorkSymbolicProblem()
        expectedCovector = [sy.Function(r'\lambda_{x}', real=True)(prob.TimeSymbol), sy.Function(r'\lambda_{v}', real=True)(prob.TimeSymbol)]
        actualCostateVector = SymbolicProblem.CreateCoVector(Vector.fromArray(prob.StateVariables), r'\lambda', prob.TimeSymbol)
        self.assertEqual(expectedCovector[0], actualCostateVector[0,0], "lmd x")
        self.assertEqual(expectedCovector[1], actualCostateVector[1,0], "lmd v")

    def testCreateHamiltonian(self) :
        prob = OneDWorkSymbolicProblem()
        lambdas = SymbolicProblem.CreateCoVector(prob.StateVariables, 'L', prob.TimeSymbol)
        expectedHamiltonian = prob.UnIntegratedPathCost + lambdas[0]*prob.EquationsOfMotion[prob.StateVariables[0]] + lambdas[1]*prob.EquationsOfMotion[prob.StateVariables[1]] 
        actualHamiltonian = prob.CreateHamiltonian(lambdas)
        self.assertTrue((expectedHamiltonian-actualHamiltonian).simplify().expand().simplify().is_zero)

    def testCreateControlConditions(self) :
        prob = OneDWorkSymbolicProblem()
        lambdas = SymbolicProblem.CreateCoVector(prob.StateVariables, 'L', prob.TimeSymbol)
        hamiltonian = prob.CreateHamiltonian(lambdas)
        expectedControlCondition = lambdas[1] + 2.0*prob.ControlVariables[0]
        controlExp = prob.CreateHamiltonianControlExpressions(hamiltonian)
        self.assertTrue((expectedControlCondition - controlExp[0,0]).is_zero)

    def testCreatingLambdaDotConditions(self) :
        prob = OneDWorkSymbolicProblem()
        lambdas = SymbolicProblem.CreateCoVector(prob.StateVariables, 'L', prob.TimeSymbol)
        hamiltonian = prob.CreateHamiltonian(lambdas)
        expectedLambdaXDot = 0
        expectedLambdaVDot = -1*lambdas[0]
        actualLambdaDots = prob.CreateLambdaDotCondition(hamiltonian)
        self.assertTrue((expectedLambdaXDot-actualLambdaDots[0,0]).is_zero, msg="lmdXDot")
        self.assertTrue((expectedLambdaVDot-actualLambdaDots[1,0]).is_zero, msg="lmdVDot")

    def testCreatingTransversalityCondition(self) :
        # this is a pretty sad test, once a more complicated example is available, use it
        prob = OneDWorkSymbolicProblem()
        lambdas = SymbolicProblem.CreateCoVector(prob.StateVariables, 'L', prob.TimeSymbol)
        hamiltonian = prob.CreateHamiltonian(lambdas)
        xversality = prob.CreateDifferentialTransversalityConditions(hamiltonian, lambdas, 0.0) # not allowing final time to vary
        print(xversality)

    def testCreateEquationOfMotionsAsEquations(self):
        prob = OneDWorkSymbolicProblem()
        eqsOfMotion = prob.CreateEquationOfMotionsAsEquations()
        self.assertEqual(2, len(eqsOfMotion), msg="2 equations returned back")
        self.assertEqual(prob.StateVariables[0].diff(prob.TimeSymbol), eqsOfMotion[0].lhs, msg="lhs of first eom")
        self.assertEqual(prob.EquationsOfMotion[prob.StateVariables[0]], eqsOfMotion[0].rhs, msg="rhs of first eom")

        self.assertEqual(prob.StateVariables[1].diff(prob.TimeSymbol), eqsOfMotion[1].lhs, msg="lhs of second eom")
        self.assertEqual(prob.EquationsOfMotion[prob.StateVariables[1]], eqsOfMotion[1].rhs, msg="rhs of second eom")

    def testCreateCostFunctionAsEquation(self) :
        prob = OneDWorkSymbolicProblem()
        costFunction = prob.CreateCostFunctionAsEquation()
        expectedRhs = sy.integrate(prob.UnIntegratedPathCost, (prob.TimeSymbol, prob.Time0Symbol, prob.TimeFinalSymbol))
        self.assertEqual(sy.Symbol('J'), costFunction.lhs, msg="default lhs")
        self.assertEqual(expectedRhs, costFunction.rhs, msg="correct rhs")

        someOtherLhs = sy.Symbol("P")
        costFunction = prob.CreateCostFunctionAsEquation(someOtherLhs)
        self.assertEqual(someOtherLhs, costFunction.lhs, msg="custom lhs")
        self.assertEqual(expectedRhs, costFunction.rhs, msg="correct rhs 2")

    def testEquationsOfMotionInMatrixForm(self) :
        prob = OneDWorkSymbolicProblem()
        eqsOfMotion = prob.EquationsOfMotionInMatrixForm()
        self.assertEqual(prob.EquationsOfMotion[prob.StateVariables[0]], eqsOfMotion[0,0], msg="first eom")
        self.assertEqual(prob.EquationsOfMotion[prob.StateVariables[1]], eqsOfMotion[1,0], msg="second eom")

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
        self.assertEqual(initialState[0], subsDict[prob.StateVariables[0]], msg="x was added when no lambdas")
        self.assertEqual(initialState[1], subsDict[prob.StateVariables[1]], msg="vx was added when no lambdas")

        lambdas = prob.CreateCoVector(prob.StateVariables, "lmd", prob.TimeSymbol)
        initialState = [1.2, 1.3, 1.4, 1.5]
        prob.AddInitialValuesToDictionary(subsDict, initialState, lambdas)
        self.assertEqual(4, len(subsDict), msg="4 items in subs dict")
        self.assertEqual(initialState[0], subsDict[prob.StateVariables[0]], msg="x was added")
        self.assertEqual(initialState[1], subsDict[prob.StateVariables[1]], msg="vx was added")        
        self.assertEqual(initialState[2], subsDict[lambdas[0]], msg="lmd x was added")
        self.assertEqual(initialState[3], subsDict[lambdas[1]], msg="lmd vx was added")        

