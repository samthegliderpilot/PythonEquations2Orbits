import unittest
import sympy as sy
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.ScaledSymbolicProblem import SymbolicProblem
from pyeq2orb.Problems.OneDimensionalMinimalWorkProblem import OneDWorkSymbolicProblem
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from scipy.integrate import solve_ivp # type: ignore
from pyeq2orb.Numerical import ScipyCallbackCreators

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
        transversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, 0.0, lambdas) # not allowing final time to vary

        zeroedOutCondition =(transversality[0]-(sy.sqrt(mu)*l_v/(2*(r*4.0)**(3/2)) - l_r + 1)).expand().simplify()
        self.assertTrue((zeroedOutCondition).is_zero, msg="first xvers cond")
        self.assertTrue((transversality[1]+lambdas[-1]).is_zero, msg="lmd theta condition")

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
        transversality = problem.TransversalityConditionsByAugmentation(aug, lambdas)

        firstZeroExpression = (transversality[0]-(-sy.sqrt(mu)*b2/(2*(r*4.0)**(3/2)) + l_r - 1)).expand().simplify()
        secondsZeroExp = transversality[1]-(-b1 + l_u).expand().simplify()
        thirdZeroExp = transversality[2]-(-b2 + l_v).expand().simplify()
        fourthZeroExp = transversality[3]-(l_theta).expand().simplify()

        self.assertTrue(firstZeroExpression.is_zero, msg="first")
        self.assertTrue(secondsZeroExp.is_zero, msg="second")
        self.assertTrue(thirdZeroExp.is_zero, msg="third")
        self.assertTrue(fourthZeroExp.is_zero, msg="fourth")


    # Regression tests for the scaled problem (for the circle to circle orbit transfer)
    # Ideally I would make more unit tests, but this will catch when thing break
    def testScaledStateRegression(self) :
        from Tests.Problems.testPlanerLeoToGeoProblem import testPlanerLeoToGeoProblem # including it here to avoid VS Code from finding TestPlanerLeoToGeo twice
        (odeSolveIvpCb, fSolveCb, tArray, z0, problem) = testPlanerLeoToGeoProblem.CreateEvaluatableCallbacks(True, False, True)
        knownAnswer = [14.95703946,  0.84256983, 15.60187053]
        answer = fSolveCb(knownAnswer)
        print(z0)
        i=0
        for val in answer :
            self.assertTrue(abs(val) < 0.2, msg=str(i)+"'th value in fsolve answer")
            i=i+1
        odeAns = solve_ivp(odeSolveIvpCb, [tArray[0], tArray[-1]], [*z0, *knownAnswer], args=tuple(), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
        finalState = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(odeAns)
        self.assertAlmostEqual(finalState[0], 6.31357956984563, 1, msg="radius check")
        self.assertAlmostEqual(finalState[1], 0.000, 2, msg="u check")
        self.assertAlmostEqual(finalState[2], 0.397980812304531, 1, msg="v check")

    def testScaldStateWithAdjoinedTransversalityRegression(self) :
        from Tests.Problems.testPlanerLeoToGeoProblem import testPlanerLeoToGeoProblem # including it here to avoid VS Code from finding TestPlanerLeoToGeo twice
        (odeSolveIvpCb, fSolveCb, tArray, z0, problem) = testPlanerLeoToGeoProblem.CreateEvaluatableCallbacks(True, False, False)
        knownAnswer = [14.95703446,  0.84256877, 15.60186291, -7.43265181, 13.6499807]
        answer = fSolveCb(knownAnswer)
        i=0
        for val in answer :
            self.assertTrue(abs(val) < 0.2, msg=str(i)+"'th value in fsolve answer")
            i=i+1
        odeAns = solve_ivp(odeSolveIvpCb, [tArray[0], tArray[-1]], [*z0, *knownAnswer[0:3]], args=tuple(knownAnswer[3:]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
        finalState = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(odeAns)
        self.assertAlmostEqual(finalState[0], 6.31357956984563, 1, msg="radius check")
        self.assertAlmostEqual(finalState[1], 0.000, 2, msg="u check")
        self.assertAlmostEqual(finalState[2], 0.397980812304531, 1, msg="v check")        

    def testScaledStateAndTimeRegression(self) :
        from Tests.Problems.testPlanerLeoToGeoProblem import testPlanerLeoToGeoProblem # including it here to avoid VS Code from finding TestPlanerLeoToGeo twice
        (odeSolveIvpCb, fSolveCb, tArray, z0, problem) = testPlanerLeoToGeoProblem.CreateEvaluatableCallbacks(True, True, True)
        knownAnswer = [1.49570410e+01, 8.42574567e-01, 1.56018729e+01, 3.43139328e+05]
        answer = fSolveCb(knownAnswer)
        print(z0)
        i=0
        for val in answer :
            self.assertTrue(abs(val) < 0.2, msg=str(i)+"'th value in fsolve answer")
            i=i+1
        odeAns = solve_ivp(odeSolveIvpCb, [tArray[0], tArray[-1]], [*z0, *knownAnswer[0:3]], args=tuple(knownAnswer[3:]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
        finalState = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(odeAns)
        self.assertAlmostEqual(finalState[0], 6.31357956984563, 1, msg="radius check")
        self.assertAlmostEqual(finalState[1], 0.000, 2, msg="u check")
        self.assertAlmostEqual(finalState[2], 0.397980812304531, 1, msg="v check")        

    def testScaledStateAndTimeAndAdjoinedTransversalityRegression(self) :
        from Tests.Problems.testPlanerLeoToGeoProblem import testPlanerLeoToGeoProblem # including it here to avoid VS Code from finding TestPlanerLeoToGeo twice
        (odeSolveIvpCb, fSolveCb, tArray, z0, problem) = testPlanerLeoToGeoProblem.CreateEvaluatableCallbacks(True, True, False)
        knownAnswer = [1.49570364e+01,  8.42572232e-01,  1.56018680e+01,  3.43139328e+05, -7.43267414e+00,  1.36499856e+01]
        answer = fSolveCb(knownAnswer)
        print(z0)
        i=0
        for val in answer :
            self.assertTrue(abs(val) < 0.2, msg=str(i)+"'th value in fsolve answer")
            i=i+1
        odeAns = solve_ivp(odeSolveIvpCb, [tArray[0], tArray[-1]], [*z0, *knownAnswer[0:3]], args=tuple(knownAnswer[3:]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
        finalState = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(odeAns)
        self.assertAlmostEqual(finalState[0], 6.31357956984563, 1, msg="radius check")
        self.assertAlmostEqual(finalState[1], 0.000, 2, msg="u check")
        self.assertAlmostEqual(finalState[2], 0.397980812304531, 1, msg="v check")

        values = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(problem.IntegrationSymbols,  odeAns)
        descaled = problem.DescaleResults(values)
        self.assertAlmostEqual(descaled[problem.WrappedProblem.StateVariables[0]][-1], 42162080.85814935, delta=50, msg="radius check descaled")
        self.assertAlmostEqual(descaled[problem.WrappedProblem.StateVariables[1]][-1], 0.000, 2, msg="u check descaled")
        self.assertAlmostEqual(descaled[problem.WrappedProblem.StateVariables[2]][-1], 3074.735, 1, msg="v check descaled")    
