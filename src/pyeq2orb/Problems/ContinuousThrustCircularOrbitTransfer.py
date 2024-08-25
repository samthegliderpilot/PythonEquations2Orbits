from matplotlib.figure import Figure # type: ignore
import sympy as sy
from typing import List, Dict, Any
from pyeq2orb.Numerical import ScipyCallbackCreators
from pyeq2orb.ProblemBase import ProblemVariable, Problem
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper
import math
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from scipy.integrate import solve_ivp # type: ignore
from pyeq2orb.Utilities.inherit import inherit_docstrings
import numpy.typing as npt
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs

@inherit_docstrings
class ContinuousThrustCircularOrbitTransferProblem(Problem) :
    def __init__(self) :
        """Initializes a new instance.  The equations of motion will be set in the order [r, u, v, longitude].
        """
        super().__init__()

        self._constantSymbols = []
        mu = sy.Symbol(r'\mu', real=True, positive=True)
        self.Mu = mu

        thrust = sy.Symbol('T', real=True, positive=True)
        self.Thrust = thrust

        m0 = sy.Symbol('m_0', real=True, positive=True)
        self.MassInitial = m0

        g = sy.Symbol('g', real=True, positive=True)
        self.Gravity = g

        isp = sy.Symbol('I_{sp}', real=True, positive=True)
        self.Isp = isp

        self._timeSymbol = sy.Symbol('t', real=True)
        self._timeInitialSymbol = sy.Symbol('t_0', real=True)
        self._timeFinalSymbol = sy.Symbol('t_f', real=True, positive=True)

        rs = sy.Function('r', real=True, positive=True)(self._timeSymbol)
        us = sy.Function('u', real=True, nonnegative=True)(self._timeSymbol)
        vs = sy.Function('v', real=True, nonnegative=True)(self._timeSymbol)
        longS = sy.Function('\\theta', real=True, nonnegative=True)(self._timeSymbol)
        control = sy.Function('\\alpha', real=True)(self._timeSymbol)

        self.MassFlowRate = -1*thrust/(isp*g)
        self.MassEquation = m0+self._timeSymbol*self.MassFlowRate

        stateVariableDynamics = []
        stateVariableDynamics.append(us)
        stateVariableDynamics.append(vs*vs/rs - mu/(rs*rs) + thrust*sy.sin(control)/self.MassEquation)
        stateVariableDynamics.append(-vs*us/rs + thrust*sy.cos(control)/self.MassEquation)
        stateVariableDynamics.append(vs/rs)

        self._stateElements.extend([
            ProblemVariable(rs, stateVariableDynamics[0]),
            ProblemVariable(us, stateVariableDynamics[1]),
            ProblemVariable(vs, stateVariableDynamics[2]),
            ProblemVariable(longS, stateVariableDynamics[3])])

        self._controlVariables.extend([control])

        finalSymbols = self.StateSymbolsFinal()
        self._boundaryConditions.extend([
                finalSymbols[1],
                finalSymbols[2]-sy.sqrt(mu/finalSymbols[0])
        ])

        self._terminalCost = finalSymbols[0] # maximization problem


    @staticmethod
    def createSolveIvpSingleShootingCallbackForFSolve(problem : Problem, integrationStateVariableArray, nonLambdaEomStateInitialValues, timeArray, solveIvpCallback, boundaryConditionExpressions, fSolveParametersToAppendToEom, fSolveOnlyParameters) :
        """A function showing a potential way to solve the boundary conditions for this problem in a shooting method with fsolve.  

        Args:
            problem (SymbolicProblem): The problem we are solving.  This can be a ContinuousThrustCircularOrbitTransferProblem or it wrapped in a ScaledSymbolicProblem
            integrationStateVariableArray (_type_): The integration state variables.
            nonLambdaEomStateInitialValues (_type_): The non-costate values that fsolve will be solving for.
            timeArray (_type_): The time array that the solution will be over.
            solveIvpCallback (_type_): The equation of motion callback that solve_ivp will solve.
            boundaryConditionExpressions (_type_): The boundary conditions (including transversality conditions) that the single shooting method will solve for.
            fSolveParametersToAppendToEom (_type_): Additional parameters that appear in the boundaryConditionExpressions that should also be passed to the equations of motion.
            fSolveOnlyParameters (_type_): Additional parameters that are in the boundaryConditionExpressions that should not be passed to the equations of motion.

        Returns:
            _type_: A callback to feed into scipy's fsolve for a single shooting method.
        """
        stateForBoundaryConditions = []
        stateForBoundaryConditions.extend(SafeSubs(integrationStateVariableArray, {problem.TimeSymbol: problem.TimeInitialSymbol}))
        stateForBoundaryConditions.extend(SafeSubs(integrationStateVariableArray, {problem.TimeSymbol: problem.TimeFinalSymbol}))
        #stateForBoundaryConditions.extend(fSolveParametersToAppendToEom)
        stateForBoundaryConditions.extend(fSolveOnlyParameters)
        

        boundaryConditionEvaluationCallbacks = LambdifyHelper.CreateLambdifiedExpressions(stateForBoundaryConditions, boundaryConditionExpressions, problem.SubstitutionDictionary)
        numberOfLambdasToPassToOdeInt = len(fSolveParametersToAppendToEom)
        def callbackForFsolve(costateAndCostateVariableGuesses) :
            z0 = []
            z0.extend(nonLambdaEomStateInitialValues)
            z0.extend(costateAndCostateVariableGuesses[0:numberOfLambdasToPassToOdeInt])
            args = costateAndCostateVariableGuesses[numberOfLambdasToPassToOdeInt:len(costateAndCostateVariableGuesses)]
            #ans = odeint(odeIntEomCallback, z0, tArray, args=tuple(args))
            ans = solve_ivp(solveIvpCallback, [timeArray[0], timeArray[-1]], z0, args=tuple(args), t_eval=timeArray, dense_output=True, method="LSODA", rtol=1.49012e-8, atol=1.49012e-11)
            finalState = []
            finalState.extend(ScipyCallbackCreators.GetInitialStateFromIntegratorResults(ans))
            finalState.extend(ScipyCallbackCreators.GetFinalStateFromIntegratorResults(ans))
            # add values in fSolve state after what is already there
            finalState.extend(costateAndCostateVariableGuesses[numberOfLambdasToPassToOdeInt:])
            finalAnswers = []
            finalAnswers.extend(boundaryConditionEvaluationCallbacks(*finalState))
        
            return finalAnswers    
        return callbackForFsolve

    def PlotSolution(self, tUnscaled : npt.NDArray, orderedDictOfUnScaledValues : Dict[sy.Symbol, List[float]], label : str) :        
        solsLists = [] #type: List[List[float]]
        for key, val in orderedDictOfUnScaledValues.items() :
            solsLists.append(val)
        plt.title("Longitude (rad)")
        longitudeDegArray = [lonRad / (2*math.pi) for lonRad in solsLists[3]]
        plt.plot(tUnscaled/86400, longitudeDegArray, label="longitude (rad)")

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.show()        
        plt.title("Radius")
        plt.plot(tUnscaled/86400, solsLists[0], label="radius (km)")

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.show()
        if len(solsLists) <= 4 :
            return
        plt.title('Lambdas')
        plt.plot(tUnscaled/86400, solsLists[4], label=r'$\lambda_r$ = ' + str(solsLists[4][0]))
        plt.plot(tUnscaled/86400, solsLists[5], label=r'$\lambda_u$ = ' + str(solsLists[5][0]))
        plt.plot(tUnscaled/86400, solsLists[6], label=r'$\lambda_v$ = ' + str(solsLists[6][0]))
        if len(solsLists) > 7 :
            plt.plot(tUnscaled/86400, solsLists[7], label=r'$\lambda_{\theta}}$ = ' + str(solsLists[7][0]))

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.show()

        ax = plt.subplot(111, projection='polar')
        thetas = []
        rads = []
        for ang, rad in zip(solsLists[3], solsLists[0]):
            thetas.append(ang)
            rads.append(rad)
        plt.title("Position Polar")    
        ax.plot(thetas, rads, label="Position (km)")

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.show()

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.title("Thrust angle (degrees vs days)")

        plt.plot(tUnscaled/86400, np.arctan2(solsLists[5], solsLists[6])*180.0/math.pi, label="Thrust angle (deg)")
        plt.show()

    @staticmethod
    def CreateInitialLambdaGuessForLeoToGeo(problem : Problem, controlSolved : sy.Expr, lambdas : List[sy.Symbol]) :
        # creating the initial values is unique to each problem, it is luck that 
        # my intuition payed off and we find a solution later
        # We want initial alpha to be 0 (or really close to it) per intuition
        # We can choose lmdv and solve for lmdu.  Start with lmdv to be 1
        # solve for lmdu with those assumptions      
        lambdasAtT0 = problem.CostateSymbolsInitial()
        constantsForLmdGuesses = problem.SubstitutionDictionary.copy()
        constantsForLmdGuesses[lambdasAtT0[2]] = 1.0 

        controlAtT0 = controlSolved.subs(problem.TimeSymbol, problem.TimeInitialSymbol)
        sinOfControlAtT0 = sy.sin(controlAtT0).trigsimp(deep=True).expand().simplify()
        alphaEq = sinOfControlAtT0.subs(lambdasAtT0[2], constantsForLmdGuesses[lambdasAtT0[2]])
        ans1 = sy.solveset(sy.Eq(0.00,alphaEq), lambdasAtT0[1])
        # doesn't like 0, so let's make it small
        ans1 = sy.solveset(sy.Eq(0.0001,alphaEq), lambdasAtT0[1])

        for thing in ans1 :
            ansForLambdaU = thing
        constantsForLmdGuesses[lambdasAtT0[1]] = float(ansForLambdaU)

        # if we assume that we always want to keep alpha small (0), we can solve dlmd_u/dt=0 for lmdr_0
        lmdUDotAtT0 = problem.CostateDynamicsEquations[1].subs(problem.TimeSymbol, problem.TimeInitialSymbol)
        lmdUDotAtT0 = lmdUDotAtT0.subs(constantsForLmdGuesses)
        inter=sy.solve(sy.Eq(lmdUDotAtT0, 0), lambdasAtT0[0].subs(constantsForLmdGuesses))
        lambdaR0Value = float(inter[0].subs(constantsForLmdGuesses)) # we know there is just 1
        constantsForLmdGuesses[lambdasAtT0[0]] = lambdaR0Value # later on, arrays will care that this MUST be a float
        initialFSolveStateGuess = [lambdaR0Value, float(ansForLambdaU), 1.0]
        return initialFSolveStateGuess        
    