from matplotlib.figure import Figure
import sympy as sy
from typing import List, Dict
from pyeq2orb.Numerical import ScipyCallbackCreators
from pyeq2orb.Numerical.LambdifyModule import LambdifyHelper
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from pyeq2orb.Utilities.inherit import inherit_docstrings

@inherit_docstrings
class ContinuousThrustCircularOrbitTransferProblem(SymbolicProblem) :
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

        self._stateVariables.extend([
            sy.Function('r', real=True, positive=True)(self._timeSymbol),
            sy.Function('u', real=True, nonnegative=True)(self._timeSymbol),
            sy.Function('v', real=True, nonnegative=True)(self._timeSymbol),
            sy.Function('\\theta', real=True, nonnegative=True)(self._timeSymbol)])

        self._controlVariables.extend([sy.Function('\\alpha', real=True)(self._timeSymbol)])

        self._boundaryConditions.extend([
                self._stateVariables[1].subs(self._timeSymbol, self._timeFinalSymbol),
                self._stateVariables[2].subs(self._timeSymbol, self._timeFinalSymbol)-sy.sqrt(mu/self._stateVariables[0].subs(self._timeSymbol, self._timeFinalSymbol))
        ])

        self._terminalCost = self._stateVariables[0].subs(self._timeSymbol, self._timeFinalSymbol) # maximization problem

        rs = self._stateVariables[0]
        us = self._stateVariables[1]
        vs = self._stateVariables[2]
        longS = self._stateVariables[3]
        control = self._controlVariables[0]

        self.MassFlowRate = -1*thrust/(isp*g)
        self.MassEquation = m0+self._timeSymbol*self.MassFlowRate

        self._equationsOfMotion[rs] = us
        self._equationsOfMotion[us] = vs*vs/rs - mu/(rs*rs) + thrust*sy.sin(control)/self.MassEquation
        self._equationsOfMotion[vs] = -vs*us/rs + thrust*sy.cos(control)/self.MassEquation
        self._equationsOfMotion[longS] = vs/rs
   
    def AppendConstantsToSubsDict(self, existingDict : dict, muVal : float, gVal : float, thrustVal : float, m0Val : float, ispVal : float) :
        """Helper function to make the substitution dictionary that is often needed when lambdifying 
        the symbolic equations.

        Args:
            existingDict (dict): The dictionary to add the values to.
            muVal (float): The gravitational parameter for the CB
            gVal (float): The value of gravity for fuel calculations (so use 1 Earth G)
            thrustVal (float): The value of the thrust
            m0Val (float): The initial mass of the spacecraft
            ispVal (float): The isp of the engines
        """
        existingDict[self.ConstantSymbols[0]] = muVal
        existingDict[self.ConstantSymbols[1]] = thrustVal
        existingDict[self.ConstantSymbols[2]] = m0Val
        existingDict[self.ConstantSymbols[3]] = gVal
        existingDict[self.ConstantSymbols[4]] = ispVal

    @staticmethod
    def createSolveIvpSingleShootingCallbackForFSolve(problem : SymbolicProblem, integrationStateVariableArray, nonLambdaEomStateInitialValues, timeArray, solveIvpCallback, boundaryConditionExpressions, fSolveParametersToAppendToEom, fSolveOnlyParameters) :
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
        stateForBoundaryConditions.extend(SymbolicProblem.SafeSubs(integrationStateVariableArray, {problem.TimeSymbol: problem.TimeInitialSymbol}))
        stateForBoundaryConditions.extend(SymbolicProblem.SafeSubs(integrationStateVariableArray, {problem.TimeSymbol: problem.TimeFinalSymbol}))
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

    def PlotSolution(self, tUnscaled : List[float], orderedDictOfUnScaledValues : Dict[object, List[float]], label : str) :
        
        solsLists = []
        for key, val in orderedDictOfUnScaledValues.items() :
            solsLists.append(val)
        plt.title("Longitude (rad)")
        plt.plot(tUnscaled/86400, solsLists[3]%(2*math.pi))

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.show()        
        plt.title("Radius")
        plt.plot(tUnscaled/86400, solsLists[0])

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
        ax.plot(thetas, rads)

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.show()

        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.legend(framealpha=1, shadow=True)
        plt.title("Thrust angle (degrees vs days)")

        plt.plot(tUnscaled/86400, np.arctan2(solsLists[5], solsLists[6])*180.0/math.pi)
        plt.show()

    @staticmethod
    def CreateInitialLambdaGuessForLeoToGeo(problem : SymbolicProblem, controlSolved : sy.Expr) :
        # creating the initial values is unique to each problem, it is luck that 
        # my intuition pays off and we find a solution later
        # We want initial alpha to be 0 (or really close to it) per intuition
        # We can choose lmdv and solve for lmdu.  Start with lmdv to be 1
        # solve for lmdu with those assumptions        
        lmdsAtT0 = problem.CreateVariablesAtTime0(problem.CostateSymbols)    
        constantsForLmdGuesses = problem.SubstitutionDictionary.copy()
        constantsForLmdGuesses[lmdsAtT0[2]] = 1.0 

        controlAtT0 = problem.CreateVariablesAtTime0(controlSolved)
        sinOfControlAtT0 = sy.sin(controlAtT0).trigsimp(deep=True).expand().simplify()
        alphEq = sinOfControlAtT0.subs(lmdsAtT0[2], constantsForLmdGuesses[lmdsAtT0[2]])
        ans1 = sy.solveset(sy.Eq(0.00,alphEq), lmdsAtT0[1])
        # doesn't like 0, so let's make it small
        ans1 = sy.solveset(sy.Eq(0.0001,alphEq), lmdsAtT0[1])

        for thing in ans1 :
            ansForLmdu = thing
        constantsForLmdGuesses[lmdsAtT0[1]] = float(ansForLmdu)

        # if we assume that we always want to keep alpha small (0), we can solve dlmd_u/dt=0 for lmdr_0
        lmdUDotAtT0 = problem.CreateVariablesAtTime0(problem.EquationsOfMotion[problem.CostateSymbols[1]])
        lmdUDotAtT0 = lmdUDotAtT0.subs(constantsForLmdGuesses)
        inter=sy.solve(sy.Eq(lmdUDotAtT0, 0), lmdsAtT0[0])
        lambdaR0Value = float(inter[0].subs(constantsForLmdGuesses)) # we know there is just 1
        constantsForLmdGuesses[lmdsAtT0[0]] = lambdaR0Value # later on, arrays will care that this MUST be a float
        initialFSolveStateGuess = [lambdaR0Value, float(ansForLmdu), 1.0]
        return initialFSolveStateGuess        
    
    def AddStandardResultsToFigure(self, figure : Figure, t : List[float], dictionaryOfValueArraysKeyedOffState : Dict[object, List[float]], label : str) -> None:
        """Adds the contents of dictionaryOfValueArraysKeyedOffState to the plot.

        Args:
            figure (matplotlib.figure.Figure): The figure the data is getting added to.
            t (List[float]): The time corresponding to the data in dictionaryOfValueArraysKeyedOffState.
            dictionaryOfValueArraysKeyedOffState (Dict[object, List[float]]): The data to get added.  The keys must match the values in self.State and self.Control.
            label (str): A label for the data to use in the plot legend.
        """
        pass        