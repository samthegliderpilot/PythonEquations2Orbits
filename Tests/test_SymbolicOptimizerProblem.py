from inspect import istraceback
import pytest
import sympy as sy
from pyeq2orb.ProblemBase import ProblemVariable, Problem # type: ignore
from pyeq2orb.Problems.OneDimensionalMinimalWorkProblem import OneDWorkSymbolicProblem # type: ignore
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem # type: ignore
from pyeq2orb.Symbolics.Vectors import Vector # type: ignore
from pyeq2orb import SafeSubs # type: ignore
import math

def testCreateCoVectorFromList() :
    prob = OneDWorkSymbolicProblem()
    expectedCoVector = [sy.Function(r'\lambda_{x}', real=True)(prob.TimeSymbol), sy.Function(r'\lambda_{v}', real=True)(prob.TimeSymbol)]
    actualCostateVector = Problem.CreateCostateVariables(prob.StateSymbols, r'\lambda', prob.TimeSymbol)
    assert expectedCoVector[0]== actualCostateVector[0], "lmd x"
    assert expectedCoVector[1]== actualCostateVector[1], "lmd v"

def testCreateCoVectorFromSymbol() :
    prob = OneDWorkSymbolicProblem()
    expectedCoVector = sy.Function(r'\lambda_{x}', real=True)(prob.TimeSymbol)
    actualCostateVector = Problem.CreateCostateVariables(prob.StateSymbols[0], r'\lambda', prob.TimeSymbol)
    assert expectedCoVector== actualCostateVector, "lmd x"

def testCreateCoVectorFromVector() :
    prob = OneDWorkSymbolicProblem()
    expectedCoVector = [sy.Function(r'\lambda_{x}', real=True)(prob.TimeSymbol), sy.Function(r'\lambda_{v}', real=True)(prob.TimeSymbol)]
    actualCostateVector = Problem.CreateCostateVariables(Vector.fromArray(prob.StateSymbols), r'\lambda', prob.TimeSymbol)
    assert expectedCoVector[0]== actualCostateVector[0,0], "lmd x"
    assert expectedCoVector[1]== actualCostateVector[1,0], "lmd v"

def testCreateHamiltonian() :
    prob = OneDWorkSymbolicProblem()
    lambdas = Problem.CreateCostateVariables(prob.StateSymbols, 'L', prob.TimeSymbol)
    expectedHamiltonian = prob.UnIntegratedPathCost + lambdas[0]*prob.StateVariableDynamics[0] + lambdas[1]*prob.StateVariableDynamics[1] 
    actualHamiltonian = prob.CreateHamiltonian(lambdas)
    assert (expectedHamiltonian-actualHamiltonian).simplify().expand().simplify().is_zero

def testCreateControlConditions() :
    prob = OneDWorkSymbolicProblem()
    lambdas = Problem.CreateCostateVariables(prob.StateSymbols, 'L', prob.TimeSymbol)
    hamiltonian = prob.CreateHamiltonian(lambdas)
    expectedControlCondition = lambdas[1] + 2.0*prob.ControlSymbols[0]
    controlExp = prob.CreateHamiltonianControlExpressions(hamiltonian)
    assert (expectedControlCondition - controlExp[0,0]).is_zero

def testCreatingLambdaDotConditions() :
    prob = OneDWorkSymbolicProblem()
    lambdas = Problem.CreateCostateVariables(prob.StateSymbols, 'L', prob.TimeSymbol)
    hamiltonian = prob.CreateHamiltonian(lambdas)
    expectedLambdaXDot = 0
    expectedLambdaVDot = -1*lambdas[0]
    actualLambdaDots = prob.CreateLambdaDotCondition(hamiltonian)
    assert (expectedLambdaXDot-actualLambdaDots[0,0]).is_zero,"lmdXDot"
    assert (expectedLambdaVDot-actualLambdaDots[1,0]).is_zero,"lmdVDot"

def testCreatingDifferentialTransversalityCondition() :
    problem = ContinuousThrustCircularOrbitTransferProblem()
    lambdas = Problem.CreateCostateVariables(problem.StateSymbols, 'L', problem.TimeFinalSymbol)
    hamiltonian = problem.CreateHamiltonian(lambdas)
    transversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, 0.0, lambdas) # not allowing final time to vary

    zeroedOutCondition =(transversality[0]-(sy.sqrt(problem.Mu)*lambdas[2]/(2*problem.StateSymbols[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol)**(3/2)) - lambdas[0] + 1)).expand().simplify()
    assert (zeroedOutCondition).is_zero,"first xvers cond"
    assert (transversality[1]+lambdas[-1]).is_zero,"lmd theta condition"

def testCreatingAugmentedTransversalityCondition() :
    problem = ContinuousThrustCircularOrbitTransferProblem()
    lambdas = Problem.CreateCostateVariables(problem.StateSymbols, 'l', problem.TimeFinalSymbol)
    l_r = lambdas[0]
    l_u = lambdas[1]
    l_v = lambdas[2]
    l_theta = lambdas[3]
    mu = problem.Mu
    r = problem.StateSymbols[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol)
    b1=sy.Symbol('b1')
    b2=sy.Symbol('b2')
    aug = [b1,b2 ]
    transversality = problem.TransversalityConditionsByAugmentation(aug, lambdas)
    print(transversality)

    firstZeroExpression = (transversality[0]-(-sy.sqrt(mu)*b2/(2*r**(3/2)) + l_r - 1)).expand().simplify()
    print(firstZeroExpression)
    secondsZeroExp = transversality[1]-(-b1 + l_u).expand().simplify()
    thirdZeroExp = transversality[2]-(-b2 + l_v).expand().simplify()
    fourthZeroExp = transversality[3]-(l_theta).expand().simplify()

    assert firstZeroExpression.is_zero,"first"
    assert secondsZeroExp.is_zero,"second"
    assert thirdZeroExp.is_zero,"third"
    assert fourthZeroExp.is_zero,"fourth"


def testCreateEquationOfMotionsAsEquations():
    prob = OneDWorkSymbolicProblem()
    eqsOfMotion = prob.CreateEquationOfMotionsAsEquations()
    assert 2, len(eqsOfMotion)=="2 equations returned back"
    assert prob.StateSymbols[0].diff(prob.TimeSymbol)== eqsOfMotion[0].lhs,"lhs of first eom"
    assert prob.StateVariableDynamics[0]== eqsOfMotion[0].rhs,"rhs of first eom"

    assert prob.StateSymbols[1].diff(prob.TimeSymbol)== eqsOfMotion[1].lhs,"lhs of second eom"
    assert prob.StateVariableDynamics[1]== eqsOfMotion[1].rhs,"rhs of second eom"

def testCreateCostFunctionAsEquation() :
    prob = OneDWorkSymbolicProblem()
    costFunction = prob.CreateCostFunctionAsEquation()
    expectedRhs = sy.integrate(prob.UnIntegratedPathCost, (prob.TimeSymbol, prob.TimeInitialSymbol, prob.TimeFinalSymbol))
    assert sy.Symbol('J')== costFunction.lhs,"default lhs"
    assert expectedRhs== costFunction.rhs,"correct rhs"

    someOtherLhs = sy.Symbol("P")
    costFunction = prob.CreateCostFunctionAsEquation(someOtherLhs)
    assert someOtherLhs== costFunction.lhs,"custom lhs"
    assert expectedRhs== costFunction.rhs,"correct rhs 2"

def testEquationsOfMotionInMatrixForm() :
    prob = OneDWorkSymbolicProblem()
    eqsOfMotion = prob.EquationsOfMotionInMatrixForm()
    assert prob.StateVariableDynamics[0]== eqsOfMotion[0,0],"first eom"
    assert prob.StateVariableDynamics[1]== eqsOfMotion[1,0],"second eom"

def testStateVariablesInMatrixForm() :
    prob = OneDWorkSymbolicProblem()
    stateAsMatrix = prob.StateVariablesInMatrixForm()
    assert prob.StateSymbols[0]== stateAsMatrix[0,0],"first state variable"
    assert prob.StateSymbols[1]== stateAsMatrix[1,0],"second state variable"

def testControlVariablesInMatrixForm() :
    prob = OneDWorkSymbolicProblem()
    stateAsMatrix = prob.ControlVariablesInMatrixForm()
    assert prob.ControlSymbols[0]== stateAsMatrix[0,0],"first control variable"

def testAddInitialValuesToDictionary() :
    prob = OneDWorkSymbolicProblem()
    subsDict = {}
    initialState = [0.2, 0.3, 0.4, 0.5]
    prob.AddInitialValuesToDictionary(subsDict, initialState)
    assert 2== len(subsDict),"only 2 items in subs dict"
    initialSvs = prob.StateSymbolsInitial()
    assert initialState[0]== subsDict[initialSvs[0]],"x was added when no lambdas"
    assert initialState[1]== subsDict[initialSvs[1]],"vx was added when no lambdas"

    lambdas = prob.CreateCostateVariables(prob.StateSymbols, "lmd", prob.TimeInitialSymbol)
    initialState = [1.2, 1.3, 1.4, 1.5]
    prob.AddInitialValuesToDictionary(subsDict, initialState, lambdas)
    assert 4== len(subsDict),"4 items in subs dict"
    assert initialState[0]== subsDict[initialSvs[0]],"x was added"
    assert initialState[1]== subsDict[initialSvs[1]],"vx was added"       
    assert initialState[2]== subsDict[lambdas[0]],"lmd x was added"
    assert initialState[3]== subsDict[lambdas[1]],"lmd vx was added"      

def testSafeSubs() :
    a = sy.Symbol('a')
    b = sy.Symbol('b')
    b = sy.Symbol('c')
    expr = a+b
    expr2 = a*a
    assert 2== SafeSubs(2, {a:b}),"int"
    assert 2.0== SafeSubs(2.0, {a:b}),"float"
    assert b== SafeSubs(a, {a:b}),"just a symbol"
    assert 2*b== SafeSubs(expr, {a:b}),"an expression"
    assert [2*b, b**2]== SafeSubs([expr, expr2], {a:b}),"list of expressions"

def testEvaluateHamiltonianAndItsFirstTwoDerivatives() :
    problem = ContinuousThrustCircularOrbitTransferProblem()
    lambdas = Problem.CreateCostateVariables(problem.StateSymbols, 'l', problem.TimeSymbol)
    problem._costateElements.extend([ProblemVariable(x, None) for x in lambdas])
    problem.StateVariableDynamics.extend([0.0,0.0,0.0,0.0])
    a = sy.Symbol('a')
    fakeHamiltonian = 3.0*sy.cos(problem.ControlSymbols[0] * problem.StateSymbols[0]*2.0*a)
    answer = {problem.StateSymbols[0] : [0.0, math.pi/8.0], problem.StateSymbols[1] : [0.0, 0.0], problem.StateSymbols[2] : [0.0, 0.0], problem.StateSymbols[3] : [0.0, 0.0] }
    answer[problem.CostateSymbols[0]] = [1.0, 1.0]
    answer[problem.CostateSymbols[1]] = [0.0, 0.0]
    answer[problem.CostateSymbols[2]] = [0.0, 0.0]
    answer[problem.CostateSymbols[3]] = [0.0, 0.0]
    [h, dh, ddh] = problem.EvaluateHamiltonianAndItsFirstTwoDerivatives(answer, [0.0, 1.0], fakeHamiltonian, {problem.ControlSymbols[0]: (problem.CostateSymbols[0]+problem.CostateSymbols[1])}, {a: 2.0})
    assert pytest.approx(3.0, 1.0e-10) == h[0], "h0"
    assert pytest.approx(0.0, 1.0e-10) == h[1], "h1"
    assert pytest.approx(0.0, 1.0e-10) == dh[0], "dh0"
    assert pytest.approx(-4.71238898038469, 1.0e-10) == dh[1], "dh1"
    assert pytest.approx(0.0, 1.0e-10) == ddh[0], "ddh0"
    assert pytest.approx(0.0, 1.0e-10) == ddh[1], "ddh1"

def testDefaultDescaleResults() :
    originalProblem = ContinuousThrustCircularOrbitTransferProblem()        
    barVars =  Problem.CreateBarVariables(originalProblem.StateSymbols, originalProblem.TimeSymbol)
    scalingExpressions = {originalProblem.StateSymbols[0]: barVars[0],originalProblem.StateSymbols[1]: barVars[1],originalProblem.StateSymbols[2]: barVars[2],originalProblem.StateSymbols[3]: barVars[3]}
    problem = originalProblem.ScaleStateVariables(barVars, scalingExpressions)
    assert scalingExpressions== problem.DescaleResults(scalingExpressions)



def testFullySymbolicProblem():
    problem = Problem()
    problem.TimeSymbol = sy.Symbol('t', real=True)
    problem.TimeInitialSymbol = sy.Symbol('t_0', real=True)
    problem.TimeFinalSymbol = sy.Symbol('t_f', real=True)
    x = sy.Function('x', real=True)(problem.TimeSymbol)
    u = sy.Function('u', real=True)(problem.TimeSymbol)

    dxdt = sy.Symbol('v_x', real=True) * u

    problem.AddStateVariable(ProblemVariable(x, dxdt))
    problem.ControlSymbols.append(u)

    problem.AddStateVariable(ProblemVariable(sy.Function(r'\lambda{x}', real=True)(problem.TimeFinalSymbol), 1))

    bc = x.subs(problem.TimeSymbol, problem.TimeFinalSymbol) - 5
    problem.BoundaryConditions.append(bc)



