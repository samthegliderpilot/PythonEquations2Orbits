import unittest
from PythonOptimizationWithNlp.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
from scipy.integrate import odeint
import numpy as np
import sympy as sy
import math
from scipy.optimize import fsolve
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
from PythonOptimizationWithNlp.ScaledSymbolicProblem import ScaledSymbolicProblem
from PythonOptimizationWithNlp.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem
from PythonOptimizationWithNlp.Numerical import ScipyCallbackCreators


class testPlanerLeoToGeoProblem(unittest.TestCase) :
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
        xversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, lambdas, 0.0) # not allowing final time to vary

        zeroedOutCondition =(xversality[0]-(sy.sqrt(problem.Mu)*lambdas[2]/(2*problem.StateVariables[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol)**(3/2)) - lambdas[0] + 1)).expand().simplify()
        self.assertTrue((zeroedOutCondition).is_zero, msg="first xvers cond")
        self.assertTrue((xversality[1]+lambdas[-1]).is_zero, msg="lmd theta condition")

    def testEndToEnd(self):
        #TODO: Still a work in progress
        g = 9.80665
        mu = 3.986004418e14  
        thrust = 20.0
        isp = 6000.0
        m0 = 1500.0

        r0 = 6678000.0
        u0 = 0.0
        v0 = sy.sqrt(mu/r0) # circular
        lon0 = 0.0
        # This is cheating, I know from many previous runs that this is the time needed to go from LEO to GEO.
        # However, below works well wrapped in another fsolve to control the final time for a desired radius.
        tfVal  = 3600*3.97152*24 

        tfOrg = tfVal
        scale = True
        scaleTime = scale and False
        # your choice of the nu vector here controls which transversality condition we use
        #nus = [sy.Symbol('B_{u_f}'), sy.Symbol('B_{v_f}')]
        nus = []

        baseProblem = ContinuousThrustCircularOrbitTransferProblem()
        problem = baseProblem
        if scale :
            newSvs = [sy.Function(r'\bar{r}')(baseProblem.TimeSymbol), sy.Function(r'\bar{u}')(baseProblem.TimeSymbol), sy.Function(r'\bar{v}')(baseProblem.TimeSymbol), sy.Function(r'\bar{lon}')(baseProblem.TimeSymbol)]
            problem = ScaledSymbolicProblem(baseProblem, newSvs, {baseProblem.StateVariables[0]: baseProblem.StateVariables[0].subs(baseProblem.TimeSymbol, baseProblem.TimeInitialSymbol), 
                                                                baseProblem.StateVariables[1]: baseProblem.StateVariables[2].subs(baseProblem.TimeSymbol, baseProblem.TimeInitialSymbol), 
                                                                baseProblem.StateVariables[2]: baseProblem.StateVariables[2].subs(baseProblem.TimeSymbol, baseProblem.TimeInitialSymbol), 
                                                                baseProblem.StateVariables[3]: 1.0} , scaleTime)


        baseProblem.RegisterConstantValue(baseProblem.Isp, isp)
        baseProblem.RegisterConstantValue(baseProblem.MassInitial, m0)
        baseProblem.RegisterConstantValue(baseProblem.Gravity, g)
        baseProblem.RegisterConstantValue(baseProblem.Mu, mu)
        baseProblem.RegisterConstantValue(baseProblem.Thrust, thrust)

        tArray = np.linspace(0.0, tfOrg, 1200)
        if scaleTime:
            tfVal = 1.0
            tArray = np.linspace(0.0, 1.0, 1200)
        constantsSubsDict = problem.SubstitutionDictionary
        initialStateValues = baseProblem.CreateVariablesAtTime0(baseProblem.StateVariables)
        initialScaledStateValues = problem.CreateVariablesAtTime0(problem.StateVariables)

        constantsSubsDict[initialStateValues[0]] = r0
        constantsSubsDict[initialStateValues[1]] = u0
        constantsSubsDict[initialStateValues[2]] = v0
        constantsSubsDict[initialStateValues[3]] = lon0

        if scale :
            # we need the initial unscaled values at tau_0 in the substitution dictionary (equations below will 
            # change t for tau, and we still need the unscaled values to evaluate correctly)
            constantsSubsDict[baseProblem.StateVariables[0].subs(baseProblem.TimeSymbol, problem.TimeInitialSymbol)] = r0
            constantsSubsDict[baseProblem.StateVariables[1].subs(baseProblem.TimeSymbol, problem.TimeInitialSymbol)] = u0
            constantsSubsDict[baseProblem.StateVariables[2].subs(baseProblem.TimeSymbol, problem.TimeInitialSymbol)] = v0
            constantsSubsDict[baseProblem.StateVariables[3].subs(baseProblem.TimeSymbol, problem.TimeInitialSymbol)] = lon0

            # and reset the real initial values
            r0= r0/r0
            u0=u0/v0
            v0=v0/v0
            lon0=lon0/1.0

            constantsSubsDict[initialScaledStateValues[0]] = r0
            constantsSubsDict[initialScaledStateValues[1]] = u0
            constantsSubsDict[initialScaledStateValues[2]] = v0
            constantsSubsDict[initialScaledStateValues[3]] = lon0    
            
        # this next block does most of the problem, pretty standard actually
        lambdas = problem.CreateCoVector(problem.StateVariables, r'\lambda', problem.TimeSymbol)
        hamiltonian = problem.CreateHamiltonian(lambdas)
        dHdu = problem.CreateHamiltonianControlExpressions(hamiltonian).doit()[0]
        controlSolved = sy.solve(dHdu, problem.ControlVariables[0])[0]

        finalEquationsOfMotion = {}
        for x in problem.StateVariables :
            finalEquationsOfMotion[x] = problem.EquationsOfMotion[x].subs(problem.ControlVariables[0], controlSolved).trigsimp(deep=True).simplify()

        lambdaDotExpressions = problem.CreateLambdaDotCondition(hamiltonian).doit()
        for i in range(0, len(lambdas)) :
            finalEquationsOfMotion[lambdas[i]] = lambdaDotExpressions[i].subs(problem.ControlVariables[0], controlSolved).simplify()

        lmdsF = problem.SafeSubs(lambdas, {problem.TimeSymbol: problem.TimeFinalSymbol})
        if len(nus) != 0:
            transversalityCondition = problem.TransversalityConditionsByAugmentation(lmdsF, nus)
        else:
            transversalityCondition = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, lmdsF, sy.Symbol(r'dt_f'))

        lmdsAtT0 = problem.CreateVariablesAtTime0(lambdas)    
        constantsSubsDict[lmdsAtT0[3]] = 0.0    

        # creating the initial values is unique to each problem, it is luck that 
        # my intuition pays off and we find a solution later
        # We want initial alpha to be 0 (or really close to it) per intuition
        # We can choose lmdv and solve for lmdu.  Start with lmdv to be 1
        # solve for lmdu with those assumptions

        constantsSubsDict[lmdsAtT0[2]] = 1.0 


        controlAtT0 = problem.CreateVariablesAtTime0(controlSolved)
        sinOfControlAtT0 = sy.sin(controlAtT0).trigsimp(deep=True).expand().simplify()
        alphEq = sinOfControlAtT0.subs(lmdsAtT0[2], constantsSubsDict[lmdsAtT0[2]])
        # doesn't like 0, so let's make it small
        ans1 = sy.solveset(sy.Eq(0.0001,alphEq), lmdsAtT0[1])

        for thing in ans1 :
            ansForLmdu = thing
        constantsSubsDict[lmdsAtT0[1]] = float(ansForLmdu)

        # if we assume that we always want to keep alpha small (0), we can solve dlmd_u/dt=0 for lmdr_0
        lmdUDotAtT0 = problem.CreateVariablesAtTime0(finalEquationsOfMotion[lambdas[1]])
        lmdUDotAtT0 = lmdUDotAtT0.subs(constantsSubsDict)
        inter=sy.solve(sy.Eq(lmdUDotAtT0, 0), lmdsAtT0[0])
        lambdaR0Value = inter[0].subs(constantsSubsDict) # we know there is just 1
        constantsSubsDict[lmdsAtT0[0]] = float(lambdaR0Value) # later on, arrays will care that this MUST be a float


        # lambda_lon is always 0, so do that cleanup
        del finalEquationsOfMotion[lambdas[3]]
        lmdsAtT0.pop()
        transversalityCondition.pop()
        lmdTheta = lambdas.pop()
        lmdsF.pop()
        constantsSubsDict[lmdTheta]=0
        constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeFinalSymbol)]=0
        constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeInitialSymbol)]=0

        # start the conversion to a numerical answer
        initialFSolveStateGuess = []
        for lmdAtT0 in lmdsAtT0 :
            initialFSolveStateGuess.append(constantsSubsDict[lmdAtT0])
            del constantsSubsDict[lmdAtT0]
        if scaleTime :
            initialFSolveStateGuess.append(tfOrg)
        integrationStateVariableArray = []
        integrationStateVariableArray.extend(problem.StateVariables)
        integrationStateVariableArray.extend(lambdas)
        otherArgs = []
        if scaleTime :
            otherArgs.append(baseProblem.TimeFinalSymbol)
        if len(nus) > 0 :
            otherArgs.extend(nus)
            
        odeIntEomCallback = ScipyCallbackCreators.CreateSimpleCallbackForOdeint(problem.TimeSymbol, integrationStateVariableArray, finalEquationsOfMotion, constantsSubsDict, otherArgs)

        allBcAndTransConditions = []
        allBcAndTransConditions.extend(transversalityCondition)
        allBcAndTransConditions.extend(problem.BoundaryConditions)
        if scaleTime :
            allBcAndTransConditions.append(baseProblem.TimeFinalSymbol-tfOrg)

        # run a test solution to get a better guess for the final nu values, this is a good technique, but 
        # it is still a custom-to-this-problem piece of code because it is still initial-guess work

        if len(nus) > 0 :
            initialFSolveStateGuess.append(initialFSolveStateGuess[1])
            initialFSolveStateGuess.append(initialFSolveStateGuess[2])  
            argsForOde = []
            if scaleTime :
                argsForOde.append(tfOrg)
            argsForOde.append(initialFSolveStateGuess[1])
            argsForOde.append(initialFSolveStateGuess[2])  
            testSolution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, *initialFSolveStateGuess[0:3]], tArray, args=tuple(argsForOde))
            initialFSolveStateGuess[-2] = testSolution[:,5][-1]
            initialFSolveStateGuess[-1] = testSolution[:,6][-1]



        fSolveCallback = ContinuousThrustCircularOrbitTransferProblem.ContinuousThrustCircularOrbitTransferProblem.createOdeIntSingleShootingCallbackForFSolve(problem, integrationStateVariableArray, [r0, u0, v0, lon0], tArray, odeIntEomCallback, allBcAndTransConditions, lambdas, otherArgs)
        fSolveSol = fsolve(fSolveCallback, initialFSolveStateGuess, epsfcn=0.00001, full_output=True) # just to speed things up and see how the initial one works
        print(fSolveSol)
        # final run with answer
        solution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, *fSolveSol[0][0:3]], tArray, args=tuple(fSolveSol[0][3:len(fSolveSol[0])]))
        #solution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, 26.0, 1.0, 27.0], tArray, args=(tfOrg,))
        asDict = ScipyCallbackCreators.ConvertOdeIntResultsToDictionary(integrationStateVariableArray, solution)


          