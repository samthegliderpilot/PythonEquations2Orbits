import unittest
import sympy as sy
from collections import OrderedDict
from pyeq2orb.Problems.OneDimensionalMinimalWorkProblem import OneDWorkSymbolicProblem
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from scipy.integrate import solve_ivp # type: ignore
from pyeq2orb.Numerical import ScipyCallbackCreators
from pyeq2orb import SafeSubs
from pyeq2orb.ProblemBase import Problem, ProblemVariable


class testScaledSymbolicProblem(unittest.TestCase) :

    def testCreatingDifferentialTransversalityCondition(self) :
        orgProblem = ContinuousThrustCircularOrbitTransferProblem()
        mu = orgProblem.Mu
        t = orgProblem.TimeSymbol
        newSvs = [sy.Function('rs')(t), sy.Function('rs')(t), sy.Function('vs')(t), sy.Function('lons')(t)]
        subs = {orgProblem.StateVariables[0]: 4.0*newSvs[0], orgProblem.StateVariables[1]: 3.0*newSvs[1], orgProblem.StateVariables[2]: 5.0*newSvs[2], orgProblem.StateVariables[3]: 7.0*newSvs[3] }
        problem = orgProblem.ScaleStateVariables(newSvs, subs)
        lambdas = Problem.CreateCoVector(problem.StateVariables, 'L', problem.TimeFinalSymbol)
        r = problem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol)
        l_r = lambdas[0]
        l_v = lambdas[2]
        hamiltonian = problem.CreateHamiltonian(lambdas)
        transversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, 0.0, SafeSubs(lambdas, {problem.TimeSymbol:problem.TimeFinalSymbol})) # not allowing final time to vary

        zeroedOutCondition =(transversality[0]-(sy.sqrt(mu)*l_v/(2*(r*4.0)**(3/2)) - l_r + 1)).expand().simplify()
        self.assertTrue((zeroedOutCondition).is_zero, msg="first xvers cond")
        self.assertTrue((transversality[1]+lambdas[-1]).is_zero, msg="lmd theta condition")

    def testCreatingAugmentedTransversalityCondition(self) :
        orgProblem = ContinuousThrustCircularOrbitTransferProblem()
        t = sy.Symbol('t')
        newSvs = [sy.Function('rs')(t), sy.Function('rs')(t), sy.Function('vs')(t), sy.Function('lons')(t)]
        subs = {orgProblem.StateVariables[0]: 4.0*newSvs[0], orgProblem.StateVariables[1]: 3.0*newSvs[1], orgProblem.StateVariables[2]: 5.0*newSvs[2], orgProblem.StateVariables[3]: 7.0*newSvs[3] }
        problem = orgProblem.ScaleStateVariables(newSvs, subs)
        lambdas = Problem.CreateCoVector(problem.StateVariables, 'l', problem.TimeFinalSymbol)
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
        # print(z0)
        # i=0
        # for val in answer :
        #     self.assertTrue(abs(val) < 0.2, msg=str(i)+"'th value in fsolve answer")
        #     i=i+1
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
        # i=0
        # for val in answer :
        #     self.assertTrue(abs(val) < 0.2, msg=str(i)+"'th value in fsolve answer")
        #     i=i+1
        odeAns = solve_ivp(odeSolveIvpCb, [tArray[0], tArray[-1]], [*z0, *knownAnswer[0:3]], args=tuple(knownAnswer[3:]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
        # print("")
        # print([odeAns.y[4][0],odeAns.y[5][0],odeAns.y[6][0]])
        # print(knownAnswer)
        finalState = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(odeAns)
        self.assertAlmostEqual(finalState[0], 6.31357956984563, 3, msg="radius check")
        self.assertAlmostEqual(finalState[1], 0.000, 3, msg="u check")
        self.assertAlmostEqual(finalState[2], 0.397980812304531, 3, msg="v check")        

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


    def testScalingStateVariables(self):
        t = sy.Symbol('t', real=True)
        r = sy.Function('r')(t)
        u = sy.Function('u')(t)
        v = sy.Function('v')(t)
        l = sy.Function('l')(t)
        m = sy.Function('m')(t)
        
        r0 = sy.Symbol('r_0')
        u0 = sy.Symbol('u_0')
        v0 = sy.Symbol('v_0')
        l0 = sy.Symbol('l_0')

        m0 = sy.Symbol('m_0', real=True, positive=True)
        mDot =sy.Symbol('\dot{m}', real=True, negative=True)
        mu = sy.Symbol(r'\mu', real=True)
        thrust =sy.Symbol('T', real=True, positive=True)
        angle = sy.Function(r'\alpha')(t)

        rDot = u
        uDot = v*v/r - mu/(r*r) + thrust*sy.sin(angle)/(m0-mDot*t)
        vDot = -u*v/r + (thrust*sy.cos(angle))/(m0-mDot*t)
        lDot = v/r

        prob = Problem()
        prob.TimeSymbol =t 
        prob.Time0Symbol = sy.Symbol('t_0')
        prob.TimeFinalSymbol = sy.Symbol('t_f')
        prob.AddStateVariable(ProblemVariable(r, rDot))
        prob.AddStateVariable(ProblemVariable(u, uDot))
        prob.AddStateVariable(ProblemVariable(v, vDot))
        prob.AddStateVariable(ProblemVariable(l, lDot))

        prob.ControlVariables.append(angle)

        prob.TerminalCost = r.subs(t, prob.TimeFinalSymbol)

        tf = prob.TimeFinalSymbol

        bc1 = u.subs(t, tf)
        bc2 = v.subs(t, tf) - 3000
        prob.BoundaryConditions.append(bc1)
        prob.BoundaryConditions.append(bc2)

        # start scaling it
        rb = sy.Function(r'\bar{r}')(t)
        ub = sy.Function(r'\bar{u}')(t)
        vb = sy.Function(r'\bar{v}')(t)
        lb = sy.Function(r'\bar{l}')(t)
        newVariables = [rb, ub, vb, lb]
        scalingDict = {r:r0*rb, u:v0*ub, v:v0*vb, l:lb}

        #DO IT
        scaledProblem = prob.ScaleStateVariables(newVariables, scalingDict)

        # from the book (not scaling by time, so tf is 1 and t instead of \tau)
        eta = v0/r0
        simpleSubsDict = {r:rb*r0, u:ub*v0, v:vb*v0, l:lb*1}
        expectedRDot = SafeSubs(rDot/r0, simpleSubsDict)
        expectedUDot = SafeSubs(uDot/v0, simpleSubsDict)
        expectedVDot = SafeSubs(vDot/v0, simpleSubsDict)
        expectedLDot = SafeSubs(lDot/1, simpleSubsDict)

        self.assertTrue((expectedRDot - scaledProblem.StateVariableDynamics[0]).is_zero, msg="rDot")
        self.assertTrue((expectedUDot - scaledProblem.StateVariableDynamics[1]).is_zero, msg="uDot")
        self.assertTrue((expectedVDot - scaledProblem.StateVariableDynamics[2]).is_zero, msg="vDot")        
        self.assertTrue((expectedLDot - scaledProblem.StateVariableDynamics[3]).is_zero, msg="lDot")

        bc1Expected = ub.subs(t, tf)*v0
        bc2Expected = vb.subs(t, tf)*v0-3000
        self.assertTrue((bc1Expected - scaledProblem.BoundaryConditions[0]).is_zero, msg="u_f bc")
        self.assertTrue((bc2Expected - scaledProblem.BoundaryConditions[1]).is_zero, msg="v_f bc")

        self.assertTrue((rb.subs(t, tf)-scaledProblem.TerminalCost).is_zero, msg="cost")        

    def testScalingTime(self):
        t = sy.Symbol('t', real=True)
        r = sy.Function('r')(t)
        u = sy.Function('u')(t)
        v = sy.Function('v')(t)
        l = sy.Function('l')(t)
        m = sy.Function('m')(t)
        
        r0 = sy.Symbol('r_0')
        u0 = sy.Symbol('u_0')
        v0 = sy.Symbol('v_0')
        l0 = sy.Symbol('l_0')

        m0 = sy.Symbol('m_0', real=True, positive=True)
        mDot =sy.Symbol('\dot{m}', real=True, negative=True)
        mu = sy.Symbol(r'\mu', real=True)
        thrust =sy.Symbol('T', real=True, positive=True)
        angle = sy.Function(r'\alpha')(t)

        rDot = u
        uDot = v*v/r - mu/(r*r) + thrust*sy.sin(angle)/(m0-mDot*t)
        vDot = -u*v/r + (thrust*sy.cos(angle))/(m0-mDot*t)
        lDot = v/r

        prob = Problem()
        prob.TimeSymbol =t 
        prob.TimeInitialSymbol = sy.Symbol('t_0', real=True)
        prob.TimeFinalSymbol = sy.Symbol('t_f', real=True)
        prob.AddStateVariable(ProblemVariable(r, rDot))
        prob.AddStateVariable(ProblemVariable(u, uDot))
        prob.AddStateVariable(ProblemVariable(v, vDot))
        prob.AddStateVariable(ProblemVariable(l, lDot))

        prob.ControlVariables.append(angle)

        prob.TerminalCost = r.subs(t, prob.TimeFinalSymbol)

        tf = prob.TimeFinalSymbol

        bc1 = u.subs(t, tf)
        bc2 = v.subs(t, tf) - 3000
        prob.BoundaryConditions.append(bc1)
        prob.BoundaryConditions.append(bc2)

        tau = sy.Symbol('TT', real=True)
        tau0 = sy.Symbol('TT_0', real=True)
        tauF = sy.Symbol('TT_f', real=True)
        tauInTermsOfT = tau * tf

        rb = sy.Function(r'r')(tau)
        ub = sy.Function(r'u')(tau)
        vb = sy.Function(r'v')(tau)
        lb = sy.Function(r'l')(tau)
        newVariables = [rb, ub, vb, lb]
        scalingDict = {}

        #DO IT
        scaledProblem = prob.ScaleTime(tau, tau0, tauF, tauInTermsOfT)

        # from the book (not scaling by time)
        eta = v0/r0
        
        simpleSubsDict =OrderedDict()
        simpleSubsDict[r] = rb
        simpleSubsDict[u] = ub
        simpleSubsDict[v] = vb
        simpleSubsDict[l] = lb
        simpleSubsDict[t] = tau*tf
        #simpleSubsDict[angle.subs(t, tau*tf)] = angle.subs(t, tau)
        expectedRDot = SafeSubs(rDot*tf, simpleSubsDict)
        expectedUDot = SafeSubs(uDot*tf, simpleSubsDict).subs(angle.subs(t, tau*tf), angle.subs(t, tau))
        expectedVDot = SafeSubs(vDot*tf, simpleSubsDict).subs(angle.subs(t, tau*tf), angle.subs(t, tau))
        expectedLDot = SafeSubs(lDot*tf, simpleSubsDict)

        self.assertTrue((expectedRDot - scaledProblem.StateVariableDynamics[0]).is_zero, msg="rDot")
        self.assertTrue((expectedUDot - scaledProblem.StateVariableDynamics[1]).is_zero, msg="uDot")
        self.assertTrue((expectedVDot - scaledProblem.StateVariableDynamics[2]).is_zero, msg="vDot")        
        self.assertTrue((expectedLDot - scaledProblem.StateVariableDynamics[3]).is_zero, msg="lDot")

        bc1Expected = ub.subs(tau, tauF)
        bc2Expected = vb.subs(tau, tauF)-3000
        self.assertTrue((bc1Expected - scaledProblem.BoundaryConditions[0]).is_zero, msg="u_f bc")
        self.assertTrue((bc2Expected - scaledProblem.BoundaryConditions[1]).is_zero, msg="v_f bc")

        self.assertTrue((rb.subs(tau, tauF)-scaledProblem.TerminalCost).is_zero, msg="cost")           

    def testScalingStateAndTime(self):
        t = sy.Symbol('t', real=True)
        r = sy.Function('r')(t)
        u = sy.Function('u')(t)
        v = sy.Function('v')(t)
        l = sy.Function('l')(t)
        m = sy.Function('m')(t)
        
        r0 = sy.Symbol('r_0')
        u0 = sy.Symbol('u_0')
        v0 = sy.Symbol('v_0')
        l0 = sy.Symbol('l_0')

        m0 = sy.Symbol('m_0', real=True, positive=True)
        mDot =sy.Symbol('\dot{m}', real=True, negative=True)
        mu = sy.Symbol(r'\mu', real=True)
        thrust =sy.Symbol('T', real=True, positive=True)
        angle = sy.Function(r'\alpha')(t)

        rDot = u
        uDot = v*v/r - mu/(r*r) + thrust*sy.sin(angle)/(m0-mDot*t)
        vDot = -u*v/r + (thrust*sy.cos(angle))/(m0-mDot*t)
        lDot = v/r

        prob = Problem()
        prob.TimeSymbol =t 
        prob.TimeInitialSymbol = sy.Symbol('t_0', real=True)
        prob.TimeFinalSymbol = sy.Symbol('t_f', real=True)
        prob.AddStateVariable(ProblemVariable(r, rDot))
        prob.AddStateVariable(ProblemVariable(u, uDot))
        prob.AddStateVariable(ProblemVariable(v, vDot))
        prob.AddStateVariable(ProblemVariable(l, lDot))

        prob.ControlVariables.append(angle)
        
        tf = prob.TimeFinalSymbol

        prob.TerminalCost = r.subs(t, prob.TimeFinalSymbol)

        bc1 = u.subs(t, tf)
        bc2 = v.subs(t, tf) - 3000
        prob.BoundaryConditions.append(bc1)
        prob.BoundaryConditions.append(bc2)

        tau = sy.Symbol('TT', real=True)
        tau0 = sy.Symbol('TT_0', real=True)
        tauF = sy.Symbol('TT_f', real=True)
        tauInTermsOfT = tau * tf

        rb = sy.Function(r'\bar{r}')(tau)
        ub = sy.Function(r'\bar{u}')(tau)
        vb = sy.Function(r'\bar{v}')(tau)
        lb = sy.Function(r'\bar{l}')(tau)
        newVariables = [rb, ub, vb, lb]
        scalingDict = {r:r0*rb, u:v0*ub, v:v0*vb, l:lb}

        #DO IT
        scaledStateProblem = prob.ScaleStateVariables(newVariables, scalingDict)
        scaledProblem = scaledStateProblem.ScaleTime(tau, tau0, tauF, tauInTermsOfT)

        # from the book (not scaling by time)
        eta = v0/r0
        
        simpleSubsDict =OrderedDict()
        simpleSubsDict[r] = rb*r0
        simpleSubsDict[u] = ub*v0
        simpleSubsDict[v] = vb*v0
        simpleSubsDict[l] = lb*1
        simpleSubsDict[t] = tau*tf
        #simpleSubsDict[angle.subs(t, tau*tf)] = angle.subs(t, tau)
        expectedRDot = SafeSubs(rDot*tf/r0, simpleSubsDict)
        expectedUDot = SafeSubs(uDot*tf/v0, simpleSubsDict).subs(angle.subs(t, tau*tf), angle.subs(t, tau))
        expectedVDot = SafeSubs(vDot*tf/v0, simpleSubsDict).subs(angle.subs(t, tau*tf), angle.subs(t, tau))
        expectedLDot = SafeSubs(lDot*tf/1, simpleSubsDict)

        self.assertTrue((expectedRDot - scaledProblem.StateVariableDynamics[0]).is_zero, msg="rDot")
        self.assertTrue((expectedUDot - scaledProblem.StateVariableDynamics[1]).is_zero, msg="uDot")
        self.assertTrue((expectedVDot - scaledProblem.StateVariableDynamics[2]).is_zero, msg="vDot")        
        self.assertTrue((expectedLDot - scaledProblem.StateVariableDynamics[3]).is_zero, msg="lDot")

        bc1Expected = ub.subs(tau, tauF)*v0
        bc2Expected = vb.subs(tau, tauF)*v0-3000
        self.assertTrue((bc1Expected - scaledProblem.BoundaryConditions[0]).is_zero, msg="u_f bc")
        self.assertTrue((bc2Expected - scaledProblem.BoundaryConditions[1]).is_zero, msg="v_f bc")      

        self.assertTrue((rb.subs(t, tf)-scaledProblem.TerminalCost).is_zero, msg="cost")             
