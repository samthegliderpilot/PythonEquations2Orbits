import pytest
from pyeq2orb.Problems.ContinuousThrustCircularOrbitTransfer import ContinuousThrustCircularOrbitTransferProblem #type: ignore
from scipy.integrate import solve_ivp # type: ignore
import numpy as np
import sympy as sy
from pyeq2orb.Numerical import ScipyCallbackCreators #type: ignore
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions #type: ignore
import pyeq2orb as pe2o #type: ignore
import importlib
from typing import List
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs #type: ignore
from pyeq2orb.ProblemBase import Problem, ProblemVariable #type: ignore
from pyeq2orb.Utilities.utilitiesForTest import assertAlmostEquals, assertAlmostEqualsDelta #type: ignore

class differentialEquations:
    def __init__(self, r0, u0, v0, lon0, g, thrust, m0, mDot, lmdLon):
        self.r0 = r0
        self.u0 = u0
        self.v0 = v0
        self.lon0 = lon0
        self.g = g
        self.thrust = thrust
        self.m0 = m0
        self.mDot = mDot
        self.lmdLon = lmdLon


    def scaledDifferentalEquationCallbac(self, t, y, *args):
        
        r = y[0]
        u = y[1]
        v = y[2]
        l = y[3]
        lmdR = y[4]
        lmdU = y[5]
        lmdV = y[6]
        if len(y) == 8:
            lmdLon = y[7]
        else:
            ldmLon = self.lmdLon

        lmdUV = math.sqrt(lmdu**2+lmdV**2)
        thrust = self.thrust
        m0 = self.m0
        mDot = self.mDot
        tf = args[0]

        eta = 1

        drdt = u*eta
        dudt = ((v**2/r - 1/(r**2))*eta)+(thrust/(m0 - mDot*t*tf))*(lmdU/(lmdUV))
        dvdt = -1*u*v*eta/r + (thrust/(m0 - mDot*t*tf))*(lmdv/(lmdUV))
        dlondt = v*eta/r
        dlmdRdt = lmdU*((v**2)/(r**2) - (2/(r**3))*eta - lmdV*u*v*eta/(r**2) + lmdLon*eta*(v/(r**2)))
        dlmdUdt = -1*lmdR*eta + lmdV*v/r
        dlmdVdt = -1*lmdU*2*v*eta/r + lmdV*u*eta/r - lmdLon*eta/r

        dydt = [drdt, dudt, dvdt, dlondt, dlmdRdt, dlmdUdt, dlmdVdt]

        if len(y) == 8:
            dlmdLon = 0
            dydt.append(dlmdLon)
        
        return dydt


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
    initialStateValues = baseProblem.StateSymbolsInitial()
    problem = baseProblem #type: Problem
    lambdas = Problem.CreateCostateVariables(problem.StateSymbols, r'\lambda', problem.TimeSymbol)
    baseProblem.CostateSymbols.extend(lambdas)
    if scale :
        newSvs = Problem.CreateBarVariables(problem.StateSymbols, problem.TimeSymbol) 
        scaleTimeFactor = None
        if scaleTime :
            scaleTimeFactor = problem.TimeFinalSymbol
        problem = baseProblem.ScaleStateVariables(newSvs, {problem.StateSymbols[0]: newSvs[0]*initialStateValues[0], 
                                                            problem.StateSymbols[1]: newSvs[1]*initialStateValues[2], 
                                                            problem.StateSymbols[2]: newSvs[2]* initialStateValues[2], 
                                                            problem.StateSymbols[3]: newSvs[3]})    
        if scaleTime :
            tau = sy.Symbol('\tau', real=True)
            problem = problem.ScaleTime(tau, sy.Symbol('\tau_0', real=True), sy.Symbol('\tau_f', real=True), tau*problem.TimeFinalSymbol)   

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
        initialValuesAtTau0 = SafeSubs(initialStateValues, {baseProblem.TimeInitialSymbol: problem.TimeInitialSymbol})
        constantsSubsDict.update(zip(initialValuesAtTau0, [r0, u0, v0, 1.0]))

        r0= r0/r0
        u0=u0/v0
        v0=v0/v0
        lon0=lon0/1.0
        # add the scaled initial values (at tau_0).  We should NOT need to add these at t_0
        initialScaledStateValues = problem.StateSymbolsInitial()
        constantsSubsDict.update(zip(initialScaledStateValues, [r0, u0, v0, 1.0])) 
        
    # this next block does most of the problem, pretty standard optimal control actions
    #problem.Lambdas = 
    
    hamiltonian = Problem.CreateHamiltonianStatic(problem.TimeSymbol, sy.Matrix([problem.StateVariableDynamics]).transpose(), 0, lambdas)
    lambdaDotExpressions = Problem.CreateLambdaDotConditionStatic(hamiltonian, sy.Matrix([problem.StateSymbols]).transpose())
    dHdu = Problem.CreateHamiltonianControlExpressionsStatic(hamiltonian, sy.Matrix([problem.ControlSymbols]).transpose())[0]
    controlSolved = sy.solve(dHdu, problem.ControlSymbols[0])[0] # something that may be different for other problems is when there are multiple control variables

    # you are in control of the order of integration variables and what EOM's get evaluated, start updating the problem
    # this line sets the lambdas in the equations of motion and integration state
    #problem.StateVariableDynamics.update(zip(lambdas, lambdaDotExpressions))
    for i in range(0, len(lambdas)) :
        problem.AddCostateVariable(ProblemVariable(lambdas[i], lambdaDotExpressions[i]))
    problem.SubstitutionDictionary[problem.ControlSymbols[0]]  =controlSolved
    SafeSubs(problem.StateVariableDynamics, {problem.ControlSymbols[0]: controlSolved})
    # the trig simplification needs the deep=True for this problem to make the equations even cleaner
    for i in range(0, len(problem.StateVariableDynamics)) :
        problem._stateElements[i].FirstOrderDynamics = problem._stateElements[i].FirstOrderDynamics.trigsimp(deep=True).simplify() # some simplification to make numerical code more stable later, and that is why this code forces us to do things somewhat manually.  There are often special things like this that we ought to do that you can't really automate.

    ## Start with the boundary conditions
    #if scaleTime : # add BC if we are working with the final time (kind of silly for this example, but we need an equal number of in's and out's for fsolve later)
    #    problem.BoundaryConditions.append(baseProblem.TimeFinalSymbol-tfOrg)

    tToTfSubsDict = {problem.TimeSymbol: problem.TimeFinalSymbol}
    # make the transversality conditions
    lambdasFinal = SafeSubs(lambdas,tToTfSubsDict)
    if len(nus) != 0:
        transversalityCondition = problem.TransversalityConditionsByAugmentation(nus, lambdasFinal)
    else:
        transversalityCondition = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, sy.Symbol(r'dt_f'), lambdasFinal)
    # and add them to the problem
    problem.BoundaryConditions.extend(transversalityCondition)

    initialFSolveStateGuess = ContinuousThrustCircularOrbitTransferProblem.CreateInitialLambdaGuessForLeoToGeo(problem, controlSolved, lambdas)

    # lambda_lon is always 0, so do that cleanup
    del problem.StateVariableDynamics[-1]
    del problem.StateSymbols[-1]
    problem.BoundaryConditions.remove(transversalityCondition[-1])
    problem._costateElements.pop()
    lmdTheta = lambdas.pop()
    constantsSubsDict[lmdTheta]=0
    constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeFinalSymbol)]=0
    constantsSubsDict[lmdTheta.subs(problem.TimeSymbol, problem.TimeInitialSymbol)]=0

    # start the conversion to a numerical problem
    if scaleTime :
        initialFSolveStateGuess.append(tfOrg)

    otherArgs = problem.OtherArguments
    if scaleTime :
        pass
        #otherArgs.append(baseProblem.TimeFinalSymbol)
    if len(nus) > 0 :
        otherArgs.extend(nus)
    
    allSvs = problem.StateSymbols
    allSvs.extend(problem.CostateSymbols)

    allDynamics = problem.StateVariableDynamics
    allDynamics.extend(problem.CostateDynamicsEquations)
    lambdifyHelper = OdeLambdifyHelperWithBoundaryConditions.CreateFromProblem(problem) #.TimeSymbol, problem.TimeInitialSymbol, problem.TimeFinalSymbol, allSvs, allDynamics, problem.BoundaryConditions, otherArgs, problem.SubstitutionDictionary)
    
    odeIntEomCallback = lambdifyHelper.CreateSimpleCallbackForSolveIvp()
    # run a test solution to get a better guess for the final nu values, this is a good technique, but 
    # it is still a custom-to-this-problem piece of code because it is still initial-guess work
    
    if len(nus) > 0 :
        initialFSolveStateGuess.append(initialFSolveStateGuess[1])
        initialFSolveStateGuess.append(initialFSolveStateGuess[2])  
        argsForOde = [] #type: List[float]
        if scaleTime :
            argsForOde.append(tfOrg)
        argsForOde.append(initialFSolveStateGuess[-2])
        argsForOde.append(initialFSolveStateGuess[-1])  
        testSolution = solve_ivp(odeIntEomCallback, [tArray[0], tArray[-1]], [r0, u0, v0, lon0, *initialFSolveStateGuess[0:3]], args=tuple(argsForOde), t_eval=tArray, dense_output=True, method="LSODA")  
        #testSolution = odeint(odeIntEomCallback, [r0, u0, v0, lon0, *initialFSolveStateGuess[0:3]], tArray, args=tuple(argsForOde))
        finalValues = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(testSolution)
        initialFSolveStateGuess[-2] = finalValues[5]
        initialFSolveStateGuess[-1] = finalValues[6]


    fSolveCallback = ContinuousThrustCircularOrbitTransferProblem.createSolveIvpSingleShootingCallbackForFSolve(problem, allSvs, [r0, u0, v0, lon0], tArray, odeIntEomCallback, problem.BoundaryConditions, lambdas, otherArgs)
    return (odeIntEomCallback, fSolveCallback, tArray, [r0, u0, v0, lon0], problem)

def testInitialization() :
    problem = ContinuousThrustCircularOrbitTransferProblem()
    assert problem.StateVariables[0].Element.subs(problem.TimeSymbol, problem.TimeFinalSymbol)== problem.TerminalCost, "terminal cost"
    assert 4== len(problem.StateSymbols), "count of state variables"
    assert 1== len(problem.ControlSymbols), "count of control variables"
    assert 0== problem.UnIntegratedPathCost, "unintegrated path cost"
    assert "t" == problem.TimeSymbol.name, "time symbol"
    assert "t_f" == problem.TimeFinalSymbol.name, "time final symbol"
    assert "t_0" == problem.TimeInitialSymbol.name, "time initial symbol"
    # thorough testing of EOM's and Boundary Conditions will be covered with solver/regression tests
    assert 4== len(problem.StateVariableDynamics), "number of EOM's"
    assert 2== len(problem.BoundaryConditions), "number of BCs"

def testDifferentialTransversalityCondition() :
    problem = ContinuousThrustCircularOrbitTransferProblem()
    lambdas = Problem.CreateCostateVariables(problem.StateSymbols, 'L', problem.TimeFinalSymbol)
    hamiltonian = problem.CreateHamiltonian(lambdas)
    transversality = problem.TransversalityConditionInTheDifferentialForm(hamiltonian, 0.0, lambdas) # not allowing final time to vary

    zeroedOutCondition =(transversality[0]-(sy.sqrt(problem.Mu)*lambdas[2]/(2*problem.StateSymbols[0].subs(problem.TimeSymbol, problem.TimeFinalSymbol)**(3/2)) - lambdas[0] + 1)).expand().simplify()
    assert (zeroedOutCondition).is_zero, "first xvers cond"
    assert (transversality[1]+lambdas[-1]).is_zero, "lmd theta condition"

# Regression tests. Ideally I would make more unit tests, but this will catch when thing break
def testRegressionWithDifferentialTransversality() :
    importlib.reload(pe2o)
    (odeSolveIvpCb, fSolveCb, tArray, z0, problem) = CreateEvaluatableCallbacks(False, False, True)
    knownAnswer = [26.22754527,  1277.08055436, 23647.7219169]
    answer = fSolveCb(knownAnswer)
    i=0
    for val in answer :
        assert abs(val) < 0.2, str(i)+"'th value in fsolve answer" + str(val) + " too big"
        i=i+1
    odeAns = solve_ivp(odeSolveIvpCb, [tArray[0], tArray[-1]], [*z0, *knownAnswer], args=tuple([]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
    finalState = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(odeAns)
    assertAlmostEqualsDelta(finalState[0], 42162079.2463495, 100, "radius check")
    assertAlmostEqualsDelta(finalState[1], 0.000, 0.1, "u check")
    assertAlmostEqualsDelta(finalState[2], 3074.735, 1.0, "v check")

def testRegressionWithAdjoinedTransversality() :
    importlib.reload(pe2o)
    (odeSolveIvpCb, fSolveCb, tArray, z0, problem) = CreateEvaluatableCallbacks(False, False, False)
    knownAnswer = [26.22755418,   1277.08146331,  23647.73092022, -11265.69782522, 20689.28488067]
    answer = fSolveCb(knownAnswer)
    i=0
    for val in answer :
        assert abs(val) < 0.3, str(i)+"'th value in fsolve answer " + str(val) + " too big"
        i=i+1
    odeAns = solve_ivp(odeSolveIvpCb, [tArray[0], tArray[-1]], [*z0, *knownAnswer[1:4]], args=tuple(knownAnswer[3:]), t_eval=tArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)  
    finalState = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(odeAns)
    assertAlmostEqualsDelta(finalState[0], 42162141.30863323, 100, "radius check")
    assertAlmostEqualsDelta(finalState[1], 0.000, .1, "u check")
    assertAlmostEqualsDelta(finalState[2], 3074.735, 1, "v check")              
