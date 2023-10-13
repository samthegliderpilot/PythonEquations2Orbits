import unittest
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from scipy.integrate import solve_ivp # type: ignore
import numpy as np
import sympy as sy
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from pyeq2orb.Numerical import ScipyCallbackCreators
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions
import pyeq2orb as pe2o
import importlib
from typing import List

class testPlanerLeoToGeoProblem(unittest.TestCase) :
    @staticmethod
    def CreateEvaluatableCallbacks(scale : bool, scaleTime : bool, useDifferentialTransversality : bool) :
        # constants
        g = 9.80665
        mu = 3.986004418e14  
        thrust = 20.0
        isp = 6000.0
        m0 = 1500.0

        # initial values
        r0 = 6678000.0
        u0 = 0.0
        v0 = sy.sqrt(mu/r0) # circular
        lon0 = 0.0
        # I know from many previous runs that this is the time needed to go from LEO to GEO.
        # However, below works well wrapped in another fsolve to control the final time for a desired radius.
        tfVal  = 3600*3.97152*24 
        tfOrg = tfVal

        # these are options to switch to try different things        
        scaleTime = scale and scaleTime       
        if useDifferentialTransversality :
            nus = []
        else:
            nus = [sy.Symbol('B_{u_f}'), sy.Symbol('B_{v_f}')]
        baseProblem = ContinuousThrustCircularOrbitTransferProblem()
        initialStateValues = baseProblem.CreateVariablesAtTime0(baseProblem.StateVariables)
        problem = baseProblem #type: SymbolicProblem

        if scale :
            newSvs = ScaledSymbolicProblem.CreateBarVariables(problem.StateVariables, problem.TimeSymbol) 
            problem = ScaledSymbolicProblem(baseProblem, newSvs, {problem.StateVariables[0]: initialStateValues[0], 
                                                                problem.StateVariables[1]: initialStateValues[2], 
                                                                problem.StateVariables[2]: initialStateValues[2], 
                                                                problem.StateVariables[3]: 1.0} , scaleTime)

        # make the time array
        tArray = np.linspace(0.0, tfOrg, 1200)
        if scaleTime:
            tfVal = 1.0
            tArray = np.linspace(0.0, 1.0, 1200)
        # register constants
        constantsSubsDict = problem.SubstitutionDictionary
        constantsSubsDict[baseProblem.Isp] = isp
        constantsSubsDict[baseProblem.MassInitial] = m0
        constantsSubsDict[baseProblem.Gravity] = g
        constantsSubsDict[baseProblem.Mu]= mu
        constantsSubsDict[baseProblem.Thrust] = thrust

        # register initial state values
        constantsSubsDict.update(zip(initialStateValues, [r0, u0, v0, 1.0]))
        if scale :
            # and reset the real initial values using tau_0 instead of time
            initialValuesAtTau0 = SymbolicProblem.SafeSubs(initialStateValues, {baseProblem.TimeInitialSymbol: problem.TimeInitialSymbol})
            constantsSubsDict.update(zip(initialValuesAtTau0, [r0, u0, v0, 1.0]))

            r0= r0/r0
            u0=u0/v0
            v0=v0/v0
            lon0=lon0/1.0
            # add the scaled initial values (at tau_0).  We should NOT need to add these at t_0
            initialScaledStateValues = problem.CreateVariablesAtTime0(problem.StateVariables)
            constantsSubsDict.update(zip(initialScaledStateValues, [r0, u0, v0, 1.0])) 
            
        # this next block does most of the problem, pretty standard optimal control actions
        problem.Lambdas.extend(problem.CreateCoVector(problem.StateVariables, r'\lambda', problem.TimeSymbol))
        lambdas = problem.Lambdas
        hamiltonian = problem.CreateHamiltonian(lambdas)
        lambdaDotExpressions = problem.CreateLambdaDotCondition(hamiltonian)
        dHdu = problem.CreateHamiltonianControlExpressions(hamiltonian)[0]
        controlSolved = sy.solve(dHdu, problem.ControlVariables[0])[0] # something that may be different for other problems is when there are multiple control variables

        # you are in control of the order of integration variables and what EOM's get evaluated, start updating the problem
        # this line sets the lambdas in the equations of motion and integration state
        problem.EquationsOfMotion.update(zip(lambdas, lambdaDotExpressions))
        SymbolicProblem.SafeSubs(problem.EquationsOfMotion, {problem.ControlVariables[0]: controlSolved})
        # the trig simplification needs the deep=True for this problem to make the equations even cleaner
        for (key, value) in problem.EquationsOfMotion.items() :
            problem.EquationsOfMotion[key] = value.trigsimp(deep=True).simplify() # some simplification to make numerical code more stable later, and that is why this code forces us to do things somewhat manually.  There are often special things like this that we ought to do that you can't really automate.

        ## Start with the boundary conditions
        if scaleTime : # add BC if we are working with the final time (kind of silly for this example, but we need an equal number of in's and out's for fsolve later)
            problem.BoundaryConditions.append(baseProblem.TimeFinalSymbol-tfOrg)

        # make the transversality conditions
        if len(nus) != 0:
            transversalityCondition = problem.TransversalityConditionsByAugmentation(nus)
        else:
            transversalityCondition = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, sy.Symbol(r'dt_f'))
        # and add them to the problem
        problem.BoundaryConditions.extend(transversalityCondition)

        initialFSolveStateGuess = ContinuousThrustCircularOrbitTransferProblem.CreateInitialLambdaGuessForLeoToGeo(problem, controlSolved)

        # lambda_lon is always 0, so do that cleanup
        del problem.EquationsOfMotion[lambdas[3]]
        problem.BoundaryConditions.remove(transversalityCondition[-1])
        lmdTheta = lambdas.pop()
        constantsSubsDict[lmdTheta]=0
        constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeFinalSymbol)]=0
        constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeInitialSymbol)]=0

        # start the conversion to a numerical problem
        if scaleTime :
            initialFSolveStateGuess.append(tfOrg)

        otherArgs = []
        if scaleTime :
            otherArgs.append(baseProblem.TimeFinalSymbol)
        if len(nus) > 0 :
            otherArgs.extend(nus)
        
        lambdifyHelper = OdeLambdifyHelperWithBoundaryConditions(problem.TimeSymbol, problem.TimeInitialSymbol, problem.TimeFinalSymbol, problem.EquationsOfMotionAsEquations, problem.BoundaryConditions, otherArgs, problem.SubstitutionDictionary)
        
        odeIntEomCallback = lambdifyHelper.CreateSimpleCallbackForSolveIvp()
        # run a test solution to get a better guess for the final nu values, this is a good technique, but 
        # it is still a custom-to-this-problem piece of code because it is still initial-guess work
        
        if len(nus) > 0 :
            initialFSolveStateGuess.append(initialFSolveStateGuess[1])
            initialFSolveStateGuess.append(initialFSolveStateGuess[2])  
            argsForOde = [] #type: List[float]
            if scaleTime :
                argsForOde.append(tfOrg)
            argsForOde.append(initialFSolveStateGuess[1])
            argsForOde.append(initialFSolveStateGuess[2])  
            testSolution = solve_ivp(odeIntEomCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *initialFSolveStateGuess[0:3]], args=tuple(argsForOde), t_eval=tArray, dense_output=True, method="LSODA")  
            #testSolution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, *initialFSolveStateGuess[0:3]], tArray, args=tuple(argsForOde))
            finalValues = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(testSolution)
            initialFSolveStateGuess[-2] = finalValues[5]
            initialFSolveStateGuess[-1] = finalValues[6]

        fSolveCallback = ContinuousThrustCircularOrbitTransferProblem.createSolveIvpSingleShootingCallbackForFSolve(problem, problem.IntegrationSymbols, [r0, u0, v0, lon0], tArray, odeIntEomCallback, problem.BoundaryConditions, lambdas, otherArgs)
        return (odeIntEomCallback, fSolveCallback, tArray, [r0, u0, v0, lon0], problem)

    def testInitialization(self) :
        problem = ContinuousThrustCircularOrbitTransferProblem()
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

    def testDifferentialTransversalityCondition(self) :
        problem = ContinuousThrustCircularOrbitTransferProblem()
        lambdas = SymbolicProblem.CreateCoVector(problem.StateVariables, 'L', problem.TimeFinalSymbol)
        hamiltonian = problem.CreateHamiltonian(lambdas)
        xversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, 0.0, lambdas) # not allowing final time to vary

        zeroedOutCondition =(xversality[0]-(sy.sqrt(problem.Mu)*lambdas[2]/(2*problem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol)**(3/2)) - lambdas[0] + 1)).expand().simplify()
        self.assertTrue((zeroedOutCondition).is_zero, msg="first xvers cond")
        self.assertTrue((xversality[1]+lambdas[-1]).is_zero, msg="lmd theta condition")

    # Regression tests. Ideally I would make more unit tests, but this will catch when thing break
    def testRegressionWithDifferentialTransversality(self) :
        importlib.reload(pe2o)
        (odeSolveIvpCb, fSolveCb, tArray, z0, problem) = testPlanerLeoToGeoProblem.CreateEvaluatableCallbacks(False, False, True)
        knownAnswer = [26.22754527,  1277.08055436, 23647.7219169]
        answer = fSolveCb(knownAnswer)
        i=0
        for val in answer :
            self.assertTrue(abs(val) < 0.2, msg=str(i)+"'th value in fsolve answer" + str(val) + " too big")
            i=i+1
        odeAns = solve_ivp(odeSolveIvpCb, [tArray[0], tArray[-1]], [*z0, *knownAnswer], args=tuple([]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
        finalState = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(odeAns)
        self.assertAlmostEqual(finalState[0], 42162071.898083754, delta=20, msg="radius check")
        self.assertAlmostEqual(finalState[1], 0.000, 2, msg="u check")
        self.assertAlmostEqual(finalState[2], 3074.735, 1, msg="v check")

    def testRegressionWithAdjoinedTransversality(self) :
        importlib.reload(pe2o)
        (odeSolveIvpCb, fSolveCb, tArray, z0, problem) = testPlanerLeoToGeoProblem.CreateEvaluatableCallbacks(False, False, False)
        knownAnswer = [26.22755418,   1277.08146331,  23647.73092022, -11265.69782522, 20689.28488067]
        answer = fSolveCb(knownAnswer)
        i=0
        for val in answer :
            self.assertTrue(abs(val) < 0.3, msg=str(i)+"'th value in fsolve answer " + str(val) + " too big")
            i=i+1
        odeAns = solve_ivp(odeSolveIvpCb, [tArray[0], tArray[-1]], [*z0, *knownAnswer[0:3]], args=tuple(knownAnswer[3:]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
        finalState = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(odeAns)
        self.assertAlmostEqual(finalState[0], 42162141.30863323, delta=40, msg="radius check")
        self.assertAlmostEqual(finalState[1], 0.000, delta=0.01, msg="u check")
        self.assertAlmostEqual(finalState[2], 3074.735, delta=1, msg="v check")              
