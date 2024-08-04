#%%
import sympy as sy
import os
import sys
sys.path.append('../')
sys.path.append('../../')
import math
from collections import OrderedDict
sy.init_printing()
from typing import Union, List, Optional, Sequence, cast, Dict, Iterator
import pyeq2orb
from pyeq2orb.ForceModels.TwoBodyForce import CreateTwoBodyMotionMatrix, CreateTwoBodyListForModifiedEquinoctialElements
from pyeq2orb.Coordinates.CartesianModule import Cartesian, MotionCartesian
from pyeq2orb.Coordinates.KeplerianModule import KeplerianElements
import pyeq2orb.Coordinates.KeplerianModule as KepModule
from pyeq2orb.ProblemBase import Problem, ProblemVariable
import pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule as mee
from IPython.display import display
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelperWithBoundaryConditions, LambdifyHelper
import scipyPaperPrinter as jh #type: ignore
from scipy.integrate import solve_ivp #type: ignore
import numpy as np
from pyeq2orb.Graphics.Plotly2DModule import plot2DLines
import pyeq2orb.Graphics.Primitives as prim
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb.Graphics.PlotlyUtilities import PlotAndAnimatePlanetsWithPlotly
from pyeq2orb import SafeSubs, MakeMatrixOfSymbols
from scipy.optimize import fsolve  #type: ignore
from pyeq2orb.Numerical import ScipyCallbackCreators #type: ignore

from pyeq2orb.Utilities.LambdifiedExpressionCache import CacheKey, ExpressionCache
import pygmo as pg #type: ignore
from abc import ABC, abstractmethod

expressionCache = ExpressionCache("SepspotRecreationExpressions.pickle")
expressionCache.ReloadFile()
def CreateAndCacheExpression(cache : ExpressionCache, id:str, sympyExpression, lmdfyCallback):
    key = CacheKey(id, str(sympyExpression))    
    item = cache.GetOrCreateAndCacheObject(key, lmdfyCallback)
    expressionCache.SaveCache()
    return item

jh.printMarkdown("# SEPSPOT Recreation")
#jh.printMarkdown("In working my way up through low-thrust modeling for satellite maneuvers, it is inevitable to run into Dr. Edelbaum's work.  Newer work such as Jean Albert Kechichian's practically requires understanding SEPSPOT as a prerequesit.  This writeup will go through the basics of SEPSPOT's algorithsm as described in the references below.")
muVal = 3.986004418e5  
def doItAll(tArray, includeJ2) ->blackBoxSingleShootingFunctions:
    #jh.printMarkdown("In other work in this python library, I have already created many helper types such as Equinoctial elements, their equations of motion, rotation matrices, and more. To start, we will define out set of equinoctial elements.  Unlike the orignial paper, I will be using the modified elements.  This replaces the semi-major axis with the parameter and reorders/renames some of the other elements.")
    t=sy.Symbol('t', real=True)
    J2 = sy.Symbol('J_2', real=True)
    rEarth = sy.Symbol('R', real=True, positive=True)
    mu = sy.Symbol(r'\mu', real=True, positive=True)
    accelSy = sy.Symbol('a', real=True, positive=True)
    muVal = 3.986004418e5  
    J2Val = 1.08263e-3 
    rEarthVal = 6378.137
    accelVal = 9.8e-5

    fullSubsDictionary = OrderedDict() #type: dict[sy.Expr, SymbolOrNumber]
    fullSubsDictionary[J2]=J2Val
    fullSubsDictionary[rEarth] = rEarthVal
    fullSubsDictionary[accelSy] = accelVal
    fullSubsDictionary[mu]=muVal

    initialKepElements = KeplerianElements(7000, 0.000, 28.5*math.pi/180.0, 0.0, 0.0, -2.299, muVal)
    finalKepElements = KeplerianElements(42000, 10**(-3), 1.0*math.pi/180.0, 0.0, 0.0, 0.0, muVal)

    simpleBoringEquiElements = mee.EquinoctialElementsHalfITrueLongitude.CreateSymbolicElements(t, mu)
    a = cast(sy.Expr, simpleBoringEquiElements.SemiMajorAxis)
    h = cast(sy.Expr, simpleBoringEquiElements.EccentricitySinTermH)
    k = cast(sy.Expr, simpleBoringEquiElements.EccentricityCosTermK)
    p = cast(sy.Expr, simpleBoringEquiElements.InclinationSinTermP)
    q = cast(sy.Expr, simpleBoringEquiElements.InclinationCosTermQ)
    L = cast(sy.Expr, simpleBoringEquiElements.Longitude)
    #n = sy.sqrt(mu/(a**3))

    initialEquiElements = mee.EquinoctialElementsHalfITrueLongitude.FromKeplerian(initialKepElements)
    a0V = float(initialEquiElements.SemiMajorAxis)
    h0V = float(initialEquiElements.EccentricitySinTermH)
    k0V = float(initialEquiElements.EccentricityCosTermK)
    p0V = float(initialEquiElements.InclinationSinTermP)
    q0V = float(initialEquiElements.InclinationCosTermQ)
    lon0= float(initialEquiElements.Longitude)
    
    finalEquiElements = mee.EquinoctialElementsHalfITrueLongitude.FromKeplerian(finalKepElements)
    aFV = float(finalEquiElements.SemiMajorAxis)
    hFV = float(finalEquiElements.EccentricitySinTermH)
    kFV = float(finalEquiElements.EccentricityCosTermK)
    pFV = float(finalEquiElements.InclinationSinTermP)
    qFV = float(finalEquiElements.InclinationCosTermQ)
    lonF= float(finalEquiElements.Longitude) 

    x = sy.Matrix([[simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermK, simpleBoringEquiElements.InclinationSinTermP, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.Longitude]]).transpose()
    z = [simpleBoringEquiElements.SemiMajorAxis, simpleBoringEquiElements.EccentricitySinTermH, simpleBoringEquiElements.EccentricityCosTermK, simpleBoringEquiElements.InclinationSinTermP, simpleBoringEquiElements.InclinationCosTermQ, simpleBoringEquiElements.Longitude]
    aSy = sy.Function('A', commutative=True)(x, t)
    u1 = sy.Symbol("u_1", real=True)
    u2 = sy.Symbol("u_2", real=True)
    u3 = sy.Symbol("u_3", real=True)
    uSy = sy.Matrix([[u1, u2, u3]]).transpose()
    
    B = simpleBoringEquiElements.CreatePerturbationMatrix(t, fullSubsDictionary)
    lonDot = sy.Matrix([[0],[0],[0],[0],[0],[1]])*simpleBoringEquiElements.UnperturbedLongitudeTimeDerivative(fullSubsDictionary)

    r = simpleBoringEquiElements.ROverA * simpleBoringEquiElements.SemiMajorAxis

    jh.showEquation("r", r)
    problem = Problem()
    for (k,v) in fullSubsDictionary.items():
        problem.SubstitutionDictionary[k] = v
    problem.TimeSymbol = t
    problem.TimeInitialSymbol = sy.Symbol('t_0', real=True)
    problem.TimeFinalSymbol = sy.Symbol('t_f', real=True)
    z0 = SafeSubs(z, {t: problem.TimeInitialSymbol})
    zF = SafeSubs(z, {t: problem.TimeFinalSymbol})

    qMult = 1
    if includeJ2:
        qMult = 1000

    problem.BoundaryConditions.append(zF[0]-aFV)
    problem.BoundaryConditions.append(zF[1]-hFV)
    problem.BoundaryConditions.append(zF[2]-kFV)
    problem.BoundaryConditions.append(zF[3]-pFV)
    problem.BoundaryConditions.append((zF[4]-qFV)*qMult)
    problem.BoundaryConditions.append(zF[5])

    
    sl = sy.sin(simpleBoringEquiElements.Longitude)
    cl = sy.cos(simpleBoringEquiElements.Longitude)
    j2Pert_r  =  -1*(3*mu*J2*(rEarth**2)/(2*r**4))*(1-(12*(q*sl-p*cl)**2)/(1+p*p+q*q)**2)
    j2Pert_th = -1*(12*mu*J2*(rEarth**2)/(r**4))*((q*sl-p*cl)*(q*cl+p*sl)/(1+p*p+q*q)**2)
    j2Pert_h  =  -1*(6*mu*J2*(rEarth**2)/(r**4))*(q*sl-p*cl)*(1-p*p-q*q)/((1+p*p+q*q)**2)
    jh.showEquation("J_{2_{r}}", j2Pert_r)
    jh.showEquation("J_{2_{t}}", j2Pert_th)
    jh.showEquation("J_{2_{h}}", j2Pert_h)
    j2GaussianVec = Cartesian(j2Pert_r, j2Pert_th, j2Pert_h)
    j2Equi = sy.Matrix([[cl, -1*sl, 0],[sl, cl, 0], [0,0,1]])*j2GaussianVec

    j2Equi = Cartesian(cl*j2Pert_r-sl*j2Pert_th, sl*j2Pert_r+cl*j2Pert_th, j2Pert_h)
    zDot = B*uSy*accelSy + lonDot
    if includeJ2:
        zDot = B*(uSy*accelSy +j2Equi) + lonDot

    for i in range(0, 6):
        for j in range(0, 3):
            jh.showEquation("B_{" + str(i+1) +"," +str(j+1) + "}", B[i,j])
    # zDot = M*uSy*accelSy + sy.Matrix([[0,0,0,0,0,taDifeq]]).transpose()



    for i in range(0, len(x)):
        problem.AddStateVariable(ProblemVariable(x[i], zDot[i]))

    tau = sy.Symbol('tt')
    originalProblem = problem
    problem = problem.ScaleTime(tau, sy.Symbol('tt_0'), sy.Symbol('tt_f'), tau*problem.TimeFinalSymbol)

    zDot = problem.EquationsOfMotionInMatrixForm()
    x = sy.Matrix(problem.StateSymbols)
    B = SafeSubs(B, {t: tau})
    def recurseArgs(someFunction, argsICareAbout, existingArgs) : 
        recursed = False
        if someFunction in argsICareAbout and not someFunction in existingArgs:
            existingArgs.append(someFunction)
            return existingArgs
        if( hasattr(someFunction, "args")) :
            for arg in someFunction.args:
                if hasattr(arg, "args"):
                    recurseArgs(arg, argsICareAbout, existingArgs)
                elif not arg in existingArgs:
                    existingArgs.append(arg)
        elif not someFunction in existingArgs:
            existingArgs.append(someFunction) #TODO: Not sure about this
        return existingArgs

    def createMatrixOfFunctionsFromDenseMatrix(someMatrix, argsICareAbout,stringName):
        mat = sy.Matrix.zeros(*someMatrix.shape)
        for r in range(0, someMatrix.rows) :
            for c in range(0, someMatrix.cols):
                thisElementName = stringName + "_{" + str(r) + "," + str(c)+"}"            
                mat[r,c] =sy.Function(thisElementName)(*recurseArgs(someMatrix[r,c], argsICareAbout, []))
        return mat

    mFullSymbol = createMatrixOfFunctionsFromDenseMatrix(B, x, "B")


    n = len(x)
    jh.printMarkdown("Staring with our x:")
    xSy = sy.MatrixSymbol('x', n, 1)
    jh.showEquation(xSy, x)
    jh.printMarkdown(r'We write our $\underline{\dot{x}}$ with the assumed optimal control vector $\underline{\hat{u}}$ as:')
    g1Sy = MakeMatrixOfSymbols(r'g_{1}', n, 1, [t])

    display(mFullSymbol)
    #xDotSy = SymbolicProblem.CreateCostateVariables(x, r'\dot{x}', t)
    #xDot = g1Sy+ aSy*mFullSymbol*uSy
    #jh.printMarkdown("Filling in our Hamiltonian, we get the following expression for our optimal thrust direction:")
    #lambdas = sy.Matrix([[r'\lambda_{1}',r'\lambda_{2}',r'\lambda_{3}',r'\lambda_{4}',r'\lambda_{5}']]).transpose()
    lambdas = Problem.CreateCostateVariables(x, r'\lambda', problem.TimeSymbol)
    #lambdasSymbol = sy.Symbol(r'\lambda^T', commutative=False)
    hamiltonian = lambdas.transpose()*zDot
    # print(hamiltonian)
    # display(hamiltonian)
    # jh.showEquation("H", hamiltonian)
    # stationaryCondition = sy.diff(hamiltonian, uSy)
    # print(stationaryCondition)
    # optU = sy.solve(stationaryCondition, uSy)
    # jh.showEquation(uSy, optU)
    #jh.printMarkdown(r'Sympy is having some trouble doing the derivative with MatrixSymbol\'s, so I\'ll explain instead.  The stationary condition will give us an expression for the optimal control, $\underline}{\hat{u}}$ by taking the partial derivative of H with respect to the control and setting it equal to zero. Then, solve for the control.  If we do that, noting that the control only appears with the $G_2$ term, and remembering that we want the normalized direction of the control vector, we get:')
    jh.printMarkdown(r'Although normally we would take the partial derivative of the Hamiltonian with respect to the control, since the Hamiltonian is linear in the control, we need to take a more intuitive approch.  We want to maximize the $G_2$ term.  It ends up being $\lambda^T a G_2 u$.  Remembering that a is a magnitude scalar and u will be a normalized direction, we can drop it.  U is a 3 by 1 matrix, and $\lambda^T G_2$ will be a 1 by 3 matrix.  Clearly to maximize this term, the optimal u needs to be in the same direction as $\lambda^T G_2$, giving us our optimal u of')

    optUOrg = lambdas.transpose()*B
    optU =optUOrg/optUOrg.norm()

    #controlSolved = sy.solve(sy.Eq(0, dHdu), uSy)
    fullSubsDictionary[uSy[0]]= optU[0]
    fullSubsDictionary[uSy[1]]= optU[1]
    fullSubsDictionary[uSy[2]]= optU[2]




    lmdDotArray = []
    print("starting 0")
    for i in range(0, n) :
        fullIntegralOfThisEom = -1*hamiltonian.diff(x[i])[0]
        lmdDotArray.append(fullIntegralOfThisEom)
        print("finished " + str(i))

    i=0
    for expr in lmdDotArray:
        jh.showEquation(lambdas[i].diff(t), expr)
        i=i+1
        break

    # now we try to integrate



    eoms = []

    for i in range(0, len(x)):
        theEq = sy.Eq(x[i].diff(t), zDot[i])
        eoms.append(theEq)
        #jh.showEquation(theEq)
    for i in range(0, len(lambdas)):
        eoms.append(sy.Eq(lambdas[i].diff(t), lmdDotArray[i]))

    eom1 = eoms[0]
    #jh.showEquation(eom1)

    actualSubsDic = {}
    for k,v in fullSubsDictionary.items() :
        actualSubsDic[k] = SafeSubs(v, fullSubsDictionary)
    fullSubsDictionary = actualSubsDic



    for i in range(0, len(lambdas)):
        problem.AddCostateElement(ProblemVariable(lambdas[i], lmdDotArray[i]))

    for (k,v) in fullSubsDictionary.items():
        problem.SubstitutionDictionary[k] =v

    scaleDict = {} #type: Dict[sy.Symbol, SymbolOrNumber]
    for sv in problem.StateSymbols :
        scaleDict[sv] = 1.0

    
    lmdHelper = OdeLambdifyHelperWithBoundaryConditions.CreateFromProblem(problem)

    lmdHelper.SubstitutionDictionary[originalProblem.TimeFinalSymbol] = originalProblem.TimeFinalSymbol
    lmdHelper.SubstitutionDictionary[lambdas[-1]] = 0
    #lmdHelper = OdeLambdifyHelperWithBoundaryConditions(t, sy.Symbol('t_0', real=True), sy.Symbol('t_f', real=True), list(x), list(zDot), [], [], fullSubsDictionary)

    
    print(problem.BoundaryConditions)
    lmdGuess = [4.675229762, 5.413413947e2, -9.202702084e3, 1.778011878e1, -2.268455855e4]#-2.2747428]
    if includeJ2 :
        lmdGuess = [4.800100306, 8.060772261e2,-9.150040837e3,3.281827358e1,-2.254928992e4]
    #lmdGuess = [4.475229762, 5.913413947e2, -9.702702084e3, 1.278011878e1, -2.968455855e4, -2.974742851]#-2.2747428] # bad values
    fullInitialState = [a0V, h0V, k0V, p0V, q0V, lon0]
    fullInitialState.extend(lmdGuess)
    print("read to lambdify")

    initialStateValues = [a0V, h0V,k0V, p0V, q0V, lon0 ]
    initialStateValues.extend(lmdGuess)
    tfV = 58110.9005
    #tfV = 60000.0
    initialStateValues.append(tfV)
    fSolveInitialState = [*lmdGuess[0:5]]
    fSolveInitialState.append(tfV)

    display("Starting ipv callback lambdification")
    name = "SepSpotRecreationIpvCallback_" + type(initialEquiElements).__name__
    if includeJ2:
        name= name+"_With J2"
    ipvCallback =CreateAndCacheExpression(expressionCache, name, sy.ImmutableMatrix(lmdHelper.ExpressionsToLambdify), lmdHelper.CreateSimpleCallbackForSolveIvp)
    #ipvCallback = lmdHelper.CreateSimpleCallbackForSolveIvp()
    display("Finished ipv callback lambdification")


    bcCallbacks = []
    fSolveState = [] #type: List[SymbolOrNumber]


    lambdaStarts= SafeSubs(lambdas, {problem.TimeSymbol: problem.TimeInitialSymbol})
    lambdaEnds = SafeSubs(lambdas, {problem.TimeSymbol: problem.TimeFinalSymbol})
    initialState = SafeSubs(problem.StateSymbols, {problem.TimeSymbol: problem.TimeInitialSymbol})
    finalState = SafeSubs(problem.StateSymbols, {problem.TimeSymbol: problem.TimeFinalSymbol})

    #all at t_0
    #[lon, lmdA, lmdF, lmdG, lmdP, lmdQ, lmdLmd, tf]
    fSolveState.append(initialState[5])
    fSolveState.extend(lambdaStarts)
    fSolveState.append(problem.TimeScaleFactor)

    bcCallbacks.extend(problem.BoundaryConditions)
    bcCallbacks = SafeSubs(bcCallbacks, problem.SubstitutionDictionary)

    fullBoundaryConditionState = [] #type: List[SymbolOrNumber]
    fullBoundaryConditionState.extend(initialState)
    fullBoundaryConditionState.extend(lambdaStarts)
    fullBoundaryConditionState.extend(finalState)
    fullBoundaryConditionState.extend(lambdaEnds)
    fullBoundaryConditionState.append(problem.TimeFinalSymbol)

    boundaryConditionsLambdified = sy.lambdify(fullBoundaryConditionState, bcCallbacks)
    

    hamlfLambdifyHelper = LambdifyHelper(lmdHelper.LambdifyArguments, hamiltonian, fullSubsDictionary)
    jh.showEquation("H", hamiltonian[0,0])
    hamltSubs1 = SafeSubs(hamiltonian[0,0], lmdHelper.SubstitutionDictionary).doit(deep=True)
    #hamltSubsFinal = SafeSubs(hamltSubs1, lmdHelper.SubstitutionDictionary).subs(t, tau*actualTfSec).subs(originalProblem.TimeFinalSymbol, 1.0).doit(deep=True)
    hamltSubsFinal = SafeSubs(hamltSubs1, lmdHelper.SubstitutionDictionary).subs(originalProblem.TimeFinalSymbol, 1.0).doit(deep=True)
    hamltEvalArgs = []
    hamltEvalArgs.extend(lmdHelper.LambdifyArguments)
    hamltEvalArgs.append(problem.TimeFinalSymbol)
    hamlfEvala = sy.lambdify(hamltEvalArgs, hamltSubsFinal, modules="numpy", cse=True)

    def realIpvCallback(tArray, ivpInitialState, ipvCallbackInner, tf = None) :
        #print(initialStateInCb)
        odeArgs = ()
        if tf != None :
            odeArgs = (tf,)
        tArray = np.linspace(0.0, 1.0, 400)            
        solution = solve_ivp(ipvCallbackInner, [tArray[0], tArray[-1]], ivpInitialState, args=odeArgs, t_eval=tArray, dense_output=True, method="RK45", rtol=1.49012e-8, atol=1.49012e-11)
        #solutionDictionary = ScipyCallbackCreators.ConvertEitherIntegratorResultsToDictionary(lmdHelper.NonTimeLambdifyArguments, solution)
        return solution


    def boundaryConditionEvaluationCallback(bcInitialState, bcFinalState, bcTimeValue, boundaryConditionCallback):
        stateNow = []
        stateNow.extend(bcInitialState)
        stateNow.extend(bcFinalState)
        stateNow.append(bcTimeValue)

        return boundaryConditionCallback(*stateNow)

    def fSolveCallback(justFSolveState):
        localIvpState = []
        localIvpState.extend(initialStateValues[0:5])
        localIvpState.extend(justFSolveState[0:-1])
        localIvpState.append(0)
        tArray = np.linspace(0.0, 1.0, 400)   
        ivpSol = realIpvCallback(tArray, localIvpState, ipvCallback, justFSolveState[-1])
        bcFinalState = ScipyCallbackCreators.GetFinalStateFromIntegratorResults(ivpSol)
        boundaryConditionSolution = boundaryConditionEvaluationCallback(localIvpState, bcFinalState, justFSolveState[-1], boundaryConditionsLambdified)[0:-1]
        print(boundaryConditionSolution)
        finalLongitude = bcFinalState[5]
        boundaryConditionSolution[-1] = 0#math.sin(finalLongitude)
        boundaryConditionSolution.append(0)#math.cos(finalLongitude)-1)
        boundaryConditionSolution.append(hamlfEvala(tArray, ivpSol.y, tArray[-1])[-1]-1.0)

        return boundaryConditionSolution

    fSolveGuess = []
    fSolveGuess.append(lon0)
    fSolveGuess.extend(lmdGuess)
    fSolveGuess.append(tfV)
    allTheCallbacks = blackBoxSingleShootingFunctions(ipvCallback, boundaryConditionsLambdified, fSolveCallback, initialStateValues[0:5], fSolveGuess, hamlfEvala)
    return allTheCallbacks


tArray = np.linspace(0.0, 1.0, 400)
#%%
twoBodyCallbacks = doItAll(tArray, False)

#%%
fSolveSol = fsolve(twoBodyCallbacks.fSolveCallback, twoBodyCallbacks.InitialFSolveGuess, full_output=True, factor=0.2, epsfcn=0.001, maxfev=0)

#%%

finalInitialState = [*twoBodyCallbacks.InitialStateForDifferentailEquations]
finalInitialState.extend(fSolveSol[0][0:-1])
finalInitialState.append(0)
actualTfSec = fSolveSol[0][-1]
solution = twoBodyCallbacks.realIpvCallback(tArray, finalInitialState, twoBodyCallbacks._difeqCallback, actualTfSec)
hamltSolutionTwoBody = twoBodyCallbacks.hamiltonianCallback(tArray, solution.y, actualTfSec)

j2Callbacks = doItAll(tArray, True)
fSolveSolJ2 = fsolve(j2Callbacks.fSolveCallback, j2Callbacks.InitialFSolveGuess, full_output=True, factor=0.2, epsfcn=0.001, maxfev=0)
actualTfSecJ2 = fSolveSolJ2[0][-1]
finalInitialJ2State = [*j2Callbacks.InitialStateForDifferentailEquations]
finalInitialJ2State.extend(fSolveSolJ2[0][0:-1])
finalInitialJ2State.append(0)

solutionJ2 = j2Callbacks.realIpvCallback(tArray, finalInitialJ2State, j2Callbacks._difeqCallback, actualTfSecJ2)
hamltSolutionJ2 = j2Callbacks.hamiltonianCallback(tArray, solutionJ2.y, actualTfSecJ2)


#%%
#solve with pygmo
class pygmoProblem :
    def __init__(self, callbacks : singleShootingFunctions):
        self.Callbacks = callbacks

    def fitness(self, x):
        return [0, *self.Callbacks.fSolveCallback(x)]

    def get_bounds(self):
        return ([-3.0, 4,  100.0, -11000.0, 10.0, -25000.0, 58050.0], 
                [-2.0, 5, 1000.0,  -8000.0, 40.0, -20000.0, 58200.0])

    def get_nic(self):
        return 0

    def get_nec(self):
        return 7

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)  

prob = pygmoProblem(j2Callbacks)
algo = pg.algorithm(uda = pg.mbh(pg.nlopt("cobyla"), seed=1))#, stop = 40, perturb = .01))
#algo.extract(pg.nlopt).local_optimizer = pg.nlopt('var2')
algo.set_verbosity(1)
pop = pg.population(prob = prob, size = 2000)
pop.problem.c_tol = [1.0, 1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-4]
pop = algo.evolve(pop)

bestX = pop.get_x()[pop.best_idx()]
finalInitialState = [*twoBodyCallbacks.InitialStateForDifferentailEquations]
finalInitialState.extend(bestX[:-1])
finalInitialState.append(0)
actualTfSec = fSolveSol[0][-1]
solution = twoBodyCallbacks.realIpvCallback(tArray, finalInitialState, twoBodyCallbacks._difeqCallback, actualTfSec)
hamltSolutionTwoBody = twoBodyCallbacks.hamiltonianCallback(tArray, solution.y, actualTfSec)

#fSolveSolJ2, solutionJ2, fullBoundaryConditionStateJ2 = doItAll(tArray, True)

#fSolveSol=fSolveSolJ2
#solution=solutionJ2
#fullBoundaryConditionState=fullBoundaryConditionStateJ2
# fSolveSolJ2 = fSolveSol
# solutionJ2 = solution
# fullBoundaryConditionStateJ2 = fullBoundaryConditionState
actualTfSecJ2 = fSolveSolJ2[0][-1]

#azimuthPlotDataSim = prim.XAndYPlottableLineData(time, np.array(simOtherValues[stateSymbols[7]])*180.0/math.pi, "azimuth_sim", '#ff00ff', 2, 0)
#elevationPlotDataSim = prim.XAndYPlottableLineData(time, np.array(simOtherValues[stateSymbols[8]])*180.0/math.pi, "elevation_sim", '#ffff00', 2, 0)

graphTArray = cast(Iterator[float], tArray*actualTfSec/3600)
graphTArrayJ2 = cast(Iterator[float], tArray*actualTfSecJ2/3600)


def plotAThing(title, label1, t1, dataset1, label2, t2, dataset2):
    plot2DLines([prim.XAndYPlottableLineData(t1, dataset1, label1, '#0000ff', 2, 0), prim.XAndYPlottableLineData(t2, dataset2, label2, '#ff00ff', 2, 0)], title)

titles = ["Semimajor Axis", "H", "K", "P", "Q", "Longitude (rad)", r'\lambda_{sma}', r'\lambda_h', r'\lambda_k', r'\lambda_p', r'\lambda_q', r'\lambda_{Longitude}']
i=0
for title in titles:
    plotAThing(titles[i], titles[i], graphTArray, solution.y[i], titles[i]+ " J_2", graphTArrayJ2, solutionJ2.y[i])
    i=i+1
i=0


plotAThing("Hamiltonian", "H", graphTArray, hamltSolutionTwoBody, "H_{J_2}", graphTArrayJ2, hamltSolutionJ2)

equiElements = [] #type: List[mee.EquinoctialElementsHalfI]
for i in range(0, len(tArray)):    
    temp = mee.EquinoctialElementsHalfITrueLongitude(solution.y[0][i], solution.y[1][i], solution.y[2][i],solution.y[3][i],solution.y[4][i],solution.y[5][i], muVal)
    
    #realEqui = scaleEquinoctialElements(temp, 1.0, 1.0)
    equiElements.append(temp)


finalKepElements = equiElements[-1].ToKeplerianConvertingAnomaly()
initialKepElements = equiElements[0].ToKeplerianConvertingAnomaly()
motions = mee.EquinoctialElementsHalfI.CreateEphemeris(equiElements)
satEphemeris = prim.EphemerisArrays()
satEphemeris.InitFromMotions(tArray, motions)
satPath = prim.PathPrimitive(satEphemeris)
satPath.color = "#ff00ff"

jh.showEquation("e", float(finalKepElements.Eccentricity))
jh.showEquation("i", float(finalKepElements.Inclination*180/math.pi))
jh.showEquation(r'\Omega', float(finalKepElements.RightAscensionOfAscendingNode*180/math.pi))
jh.showEquation(r'\omega', float(finalKepElements.ArgumentOfPeriapsis*180/math.pi))
jh.showEquation(r'M', float(finalKepElements.TrueAnomaly*180/math.pi))

# for i in range(0, 12):
#     if i == 6:
#         jh.showEquation(fullBoundaryConditionState[i+12], (solution.y[i][0]*180/math.pi)%360)
#     else:
#         jh.showEquation(fullBoundaryConditionState[i+12], solution.y[i][0])

# for i in range(0, 12):
#     if i == 6:
#         jh.showEquation(fullBoundaryConditionState[i+12], (solution.y[i][-1]*180/math.pi)%360)
#     else:
#         jh.showEquation(fullBoundaryConditionState[i+12], solution.y[i][-1])


import plotly.graph_objects as go #type: ignore
def ms(x, y, z, radius, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

earth = ms(0,0,0, 6378.137, 30)
sphere = go.Surface()
fig = PlotAndAnimatePlanetsWithPlotly("Orbiting Earth", [satPath], tArray, None)
fig.add_surface(x=earth[0], y=earth[1], z=earth[2], opacity=1.0, autocolorscale=False, showlegend=False)
fig.update_xaxes(visible=False, showticklabels=False)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
fig['layout']['sliders'][0]['pad']=dict(r= 0, t= 0, b=0, l=0)
fig['layout']['updatemenus'][0]['pad']=dict(r= 0, t= 0, b=0, l=0)
fig.show()  


import pandas as pd #type:ignore
import numpy as np #type:ignore
#"Initial Values": ["SMA (km)", "Ecc", "Inc (deg)", "RAAN (deg)", "AoP (deg)", "TA (deg)"],
df = pd.DataFrame({
    "Elements" : initialKepElements.NamesToArray,
    "Initial Elements" : initialKepElements.ToArray(True),    
    "Final Elements" : [float(x) for x in finalKepElements.ToArray(True)],
})
df.style \
  .format(precision=3, thousands=",", decimal=".") \
  .format_index(str.upper, axis=1) \
  .relabel_index(initialKepElements.NamesToArray(), axis=0)


print(actualTfSec)
print(actualTfSecJ2)
