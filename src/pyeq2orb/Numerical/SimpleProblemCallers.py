from pyeq2orb.ProblemBase import Problem, ProblemVariable
from pyeq2orb import SafeSubs
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb.NumericalOptimizerProblem import NumericalOptimizerProblemBase
import sympy as sy
from IPython.display import display
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Callable, cast, Iterable
import pyeq2orb.Numerical.ScipyCallbackCreators as ScipyCallbackCreators
import numpy as np
from scipy.integrate import solve_ivp #type: ignore
from scipy.optimize import fsolve  #type: ignore

"""
This module is proving wrappers that help stitch together the various stages of solving problems.
I realized at one point that I was trying to have 1 function go from the symbolic problem definition 
to a numerical solver.  But once I realized that I didn't have a layer that could just propagate 
and evaluate boundary conditions independent of the solver... that thought lead to these structures.

They really just provide from structure to solving problems.  They also implicitly define the states 
that get passed into callbacks.  There is some questionable coupling between these and the lambdify 
helpers that I might come to regret.  
"""

class IIntegrationAnswer(ABC):
    """The base interface for answers from a propagation problem with potential boundary conditions."""
    def __init__(self):
        pass

    @property
    @abstractmethod
    def TimeHistory(self) -> List[float]:
        """The time history of the problem.

        Returns:
            Iterable[float]: The time history as propagated. If scaling was performed, 
            this list should NOT be descaled (that is higher level responsibility).
        """
        pass

    @property
    @abstractmethod    
    def StateHistory(self) -> Dict[sy.Symbol, List[float]]:
        """Gets the state history in a dictionary keyed off of the original symbols
        (of time) getting propagated.  Again, if the underlying problem was scaled, 
        the values here should NOT be descaled.

        Returns:
            Dict[sy.Symbol, List[float]]: The dictionary of state histories of the problem.
        """
        pass

    @property
    @abstractmethod    
    def RawIntegratorOutput(self) -> Optional[Any]:
        """Get the raw integrator output.  This can be anything and is determined by 
        whatever algorithm you choose to use.  This can be None.

        Returns:
            Optional[Any]: Whatever the raw integrator output was
        """
        pass

    @property
    @abstractmethod
    def OriginalProblem(self) -> "EverythingProblem":
        pass

    def BuildBoundaryConditionStateFromIntegrationAnswer(self) -> List[float]:
        """Builds an flat array of the initial time, initial state, final time, and final state, which is
        what the lambdified Boundary Condition callback expects.

        Returns:
            List[float]: The initial time, initial state, final time and final state all in one simple list.
        """
        time = self.TimeHistory
        solutionDict = self.StateHistory
        bcState = []
        bcState.append(time[0])
        for k, v in solutionDict.items():
            bcState.append(v[0])
        bcState.append(time[-1])
        for k, v in solutionDict.items():
            bcState.append(v[-1])         
        # don't need args since they are passed in at evaluation time of the BC's   
        return bcState

    def StateVariableHistoryByIndex(self, i:int) -> List[float]:
        return self.StateHistory[self.OriginalProblem.StateSymbols[i]]

class IEverythingAnswer(IIntegrationAnswer):
    """The base interface for answers from a propagation problem with potential boundary conditions."""
    def __init__(self):
        pass
    
    @property
    @abstractmethod
    def BoundaryConditionValues(self) -> List[float]:
        """Gets the boundary conditions values of the problem. They are in the order that the 
        underlying problem had them in.

        Returns:
            List[float]: The boundary condition values.
        """
        pass



class SimpleIntegrationAnswer(IIntegrationAnswer):
    """An IIntegrationAnswer that is just wrapping underlying lists and dictionaries of the 
    values.

    Args:
        IIntegrationAnswer: This is an IIntegrationAnswer.
    """
    def __init__(self, originalProblem : "EverythingProblem", timeHistory: List[float], StateHistory : Dict[sy.Symbol, List[float]], rawIntegratorOutput : Optional[Any] = None):
        """Initializes a new instance.

        Args:
            originalProblem (EverythingProblem): The problem that evaluated these results.
            timeHistory (List[float]): The time history.
            StateHistory (Dict[sy.Symbol, List[float]]): The states separated into lists of floats keyed off of the symbols.
            rawIntegratorOutput (Optional[Any], optional): The raw integrator output. Defaults to None.
        """
        self._timeHistory = timeHistory
        self._stateHistory = StateHistory
        self._rawIntegratorOutput= rawIntegratorOutput
        self._originalProblem = originalProblem

    @property
    def TimeHistory(self) -> List[float]:
        """The time history of the problem.

        Returns:
            Iterable[float]: The time history as propagated. If scaling was performed, 
            this list should NOT be descaled.
        """
        return self._timeHistory

    @property
    def StateHistory(self) -> Dict[sy.Symbol, List[float]]:
        """Gets the state history in a dictionary keyed off of the original symbols
        (of time) getting propagated.  Again, if the underlying problem was scaled, 
        the values here should NOT be descaled.

        Returns:
            Dict[sy.Symbol, List[float]]: The dictionary of state histories of the problem.
        """
        return self._stateHistory

    @property
    def RawIntegratorOutput(self)->Optional[Any]:
        """Get the raw integrator output.  This can be anything and is determined by 
        whatever algorithm you choose to use.  This can be None.

        Returns:
            Optional[Any]: Whatever the raw integrator output was
        """
        return self._rawIntegratorOutput

    @property
    def OriginalProblem(self) -> "EverythingProblem":
        return self._originalProblem

class SimpleEverythingAnswer(SimpleIntegrationAnswer, IEverythingAnswer):
    """An IEverythingAnswer that is just wrapping underlying lists and dictionaries of the 
    values.

    Args:
        IEverythingAnswer: This is an IEverythingAnswer and a SimpleIntegrationAnswer.
    """
    def __init__(self, originalProblem :"EverythingProblem", timeHistory: List[float], stateHistory : Dict[sy.Symbol, List[float]], bcAnswer : List[float], rawIntegratorOutput = Optional[Any]):
        """"Initializes a new instance.

        Args:
            timeHistory (List[float]): The time history.
            StateHistory (Dict[sy.Symbol, List[float]]): The states separated into lists of floats keyed off of the symbols.
            bcAnswer (List[float]): The values of the boundary conditions.
            rawIntegratorOutput (Optional[Any], optional): The raw integrator output. Defaults to None.
        """
        SimpleIntegrationAnswer.__init__(self, originalProblem, timeHistory, stateHistory, rawIntegratorOutput)
        #self._timeHistory = timeHistory # I should look into calling IIntegrationAnswer specifically, but eh
        #self._stateHistory = stateHistory
        self._boundaryConditionValues = bcAnswer
        #self._rawIntegratorOutput = rawIntegratorOutput 
        #self._originalProblem = originalProblem

    @property
    def BoundaryConditionValues(self) -> List[float]:
        """Gets the boundary conditions values of the problem. They are in the order that the 
        underlying problem had them in.

        Returns:
            List[float]: The boundary condition values.
        """
        return self._boundaryConditionValues





class EverythingProblem(ABC):
    """A "problem" that propagates a trajectory in time and has optional boundary conditions.

    Args:
        ABC: This type has abstract methods that must be filled in to work correctly.
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    def StateSymbols(self) ->List[sy.Symbol]:
        """A list of the state variable symbols (normal state variables, co-states, path constraints... doesn't matter, all 
        variables getting propagated through time).

        Returns:
            List[sy.Symbol]: The symbols getting propagated.
        """
        pass

    @property
    @abstractmethod
    def OtherArgumentSymbols(self) -> List[sy.Symbol]:
        """A list of the other arguments that will be passed into both the integrator function and boundary conditions.

        Returns:
            List[sy.Symbol]: The other arguments/parameters that will get passed to the functions.
        """
        pass

    @property
    @abstractmethod
    def BoundaryConditionExpressions(self) -> List[sy.Expr]: # this is a sympy expression, it must equal 0
        """Returns a list of the symbolic expressions for the boundary conditions.  These expressions 
        should be designed that, for the desired solution, they will equal 0.

        Returns:
            List[sy.Expr]: The symbolic expressions for the boundary conditions.
        """
        pass

    @abstractmethod
    def EvaluateProblem(self, time : List[float], initialState : List[float], args : List[float]) ->IEverythingAnswer:
        """Evaluates this problem returning an IEverythingAnswer.

        Args:
            time (Iterable[float]): The points in time to evaluate.
            initialState (List[float]): The initial state to propagate from.
            parameters (List[float]): Any additional arguments to be passed to both the propagation 
            of the trajectory AND the boundary conditions.

        Returns:
            IEverythingAnswer: The evaluated trajectory and boundary conditions.
        """
        pass




class SingleShootingFunctions(EverythingProblem, ABC):
    def __init__(self):
        pass

    @abstractmethod
    def IntegrateDifferentialEquations(self, time : Iterable[float], y0 : List[float], args  : List[float]) -> IIntegrationAnswer:
        """Performs the full integration of the problem.  This function will wrap your call to solve_ivp or ode_solve or 
        whatever integration scheme you want.

        Args:
            time (List[float]): The times to evaluate the problem at.
            y0 (List[float]): _description_
            args (List[float]): _description_

        Returns:
            IIntegrationAnswer: _description_
        """
        pass

    @abstractmethod
    def BoundaryConditionEvaluation(self, integrationAnswer: IIntegrationAnswer, args : List[float]) -> List[float]:
        pass
    
    def EvaluateProblem(self, time : Iterable[float], initialState : List[float], args : List[float]) -> IEverythingAnswer:
        ivpAns = self.IntegrateDifferentialEquations(time, initialState, args)
        bcAns = self.BoundaryConditionEvaluation(ivpAns, args)
        return SimpleEverythingAnswer(self, ivpAns.TimeHistory, ivpAns.StateHistory, bcAns, ivpAns.RawIntegratorOutput)
            
    @staticmethod
    def CreateBoundaryConditionCallbackFromLambdifiedCallback(lambdifiedCallback) ->Callable[[IIntegrationAnswer, Optional[List[float]]], List[float]]:
        def callback(integrationAnswer : IIntegrationAnswer, *args : Tuple[float, ...]) -> List[float]:
            bcStateForLambdafiedCallback = integrationAnswer.BuildBoundaryConditionStateFromIntegrationAnswer()
            bcSolved = lambdifiedCallback(*bcStateForLambdafiedCallback, *args)
            return bcSolved
        return callback #type: ignore

    @staticmethod
    def CreateIntegrationCallbackFromLambdifiedCallback(lambdifiedCallback : Callable[[List[float], List[float], List[float]], Tuple[Dict[sy.Symbol, List[float]], Optional[Any]]], problem) -> Callable[[List[float], List[float], List[float]], IIntegrationAnswer]:
        def fullCallback(t : List[float], x : List[float], args : List[float]) -> IIntegrationAnswer:
            (dictAns, integratorOutput) = lambdifiedCallback(t, x, *args)
            integrationAnswer = SimpleIntegrationAnswer(problem, t, dictAns, integratorOutput)
            return integrationAnswer
        return fullCallback


class BlackBoxSingleShootingFunctions(SingleShootingFunctions):
    def __init__(self, integrationCallback, boundaryConditionCallback,  stateVariables : List[sy.Symbol], boundaryConditionExpressions : List[sy.Expr], otherArgs : Optional[List[sy.Symbol]] = None):
        self._integrationCallback = integrationCallback
        self._boundaryConditionCallback = boundaryConditionCallback
        self._stateVariables = stateVariables
        self._boundaryConditionExpressions = boundaryConditionExpressions
        if otherArgs == None:
            otherArgs = []            
        self._otherArguments = cast(List[sy.Symbol], otherArgs)

    @property
    def StateSymbols(self) ->List[sy.Symbol]:
        return self._stateVariables

    @property
    def BoundaryConditionExpressions(self) -> List[sy.Expr]:
        return self._boundaryConditionExpressions

    @property
    def OtherArgumentSymbols(self) ->List[sy.Symbol]:
        return self._otherArguments

    def IntegrateDifferentialEquations(self, t, y, args) -> IIntegrationAnswer:
        return self._integrationCallback(t, y, args)

    def BoundaryConditionEvaluation(self, integrationAnswer : IIntegrationAnswer, args  : List[float]) -> List[float]:     
        return self._boundaryConditionCallback(integrationAnswer, *args)

class BlackBoxSingleShootingFunctionsFromLambdifiedFunctions(BlackBoxSingleShootingFunctions):
    def __init__(self, integrationCallback, boundaryConditionCallback,  stateVariables : List[sy.Symbol], boundaryConditionExpressions : List[sy.Expr], otherArgs : Optional[List[sy.Symbol]] = None):
        realIvpCallback = BlackBoxSingleShootingFunctions.CreateIntegrationCallbackFromLambdifiedCallback(integrationCallback, self)
        realBcCallback = BlackBoxSingleShootingFunctions.CreateBoundaryConditionCallbackFromLambdifiedCallback(boundaryConditionCallback)
        super().__init__(realIvpCallback, realBcCallback, stateVariables, boundaryConditionExpressions, otherArgs)

class solverAnswer:
    def __init__(self, evaluatedAnswer : IEverythingAnswer, solvedControls : List[float], constraintValues: List[float], solverResult: Any = None):
        self.EvaluatedAnswer =evaluatedAnswer
        self.SolvedControls = solvedControls
        self.ConstraintValues = constraintValues
        self.SolverResult = solverResult




class singleShootingSolver(ABC):
    def __init__(self, evaluatableProblem:EverythingProblem, controlsForSolver, constraintsForSolver ):
        self.EvaluatableProblem = evaluatableProblem

        self._indicesOfInitialStateToUpdate : Dict[int, int] = {}
        self._indicesOfProblemParametersToUpdate : Dict[int, int] = {}
        overallBcStateValuesToSample :List[int] = []

        i = -1
        for control in controlsForSolver:
            i = i+1
            if control in evaluatableProblem.StateSymbols:
                thisControlIndex = evaluatableProblem.StateSymbols.index(control)
                self._indicesOfInitialStateToUpdate[i] = thisControlIndex
                continue
            thisControlIndex = evaluatableProblem.OtherArgumentSymbols.index(control)
            self._indicesOfProblemParametersToUpdate[i] = thisControlIndex

        self._indicesForConstraintsFromBoundaryConditions :List[int]= []
        for constraint in constraintsForSolver:
            thisConstraintIndex = evaluatableProblem.BoundaryConditionExpressions.index(constraint)
            self._indicesForConstraintsFromBoundaryConditions.append(thisConstraintIndex)

    @abstractmethod
    def solve(self, initialGuessOfSolverControls : List[float], time: Iterable[float], initialState : List[float], parameters : List[float], **solverKwargs) -> solverAnswer:
        pass

    def updateInitialStateWithSolverValues(self, solverGuess : List[float], mutableInitialState : List[float]):
        for k,v in self._indicesOfInitialStateToUpdate.items():
            mutableInitialState[v] = solverGuess[k]

    def updateInitialParametersWithSolverValues(self, solverGuess : List[float], mutableArgs : List[float]) ->List[float]:
        for k,v in self._indicesOfProblemParametersToUpdate.items():
            mutableArgs[v] = solverGuess[k]
        return mutableArgs

    def createArgsForIntegrator(self, solverGuess:List[float], mutableIntegratorArgs : List[float]):
        for k,v in self._indicesOfProblemParametersToUpdate.items():
            mutableIntegratorArgs[v] = solverGuess[k]

    def createArgsForSolver(self, solverGuess:List[float], mutableIntegratorArgs: List[float]):
        pass #TODO
    
            

    def getSolverValuesFromEverythingAns(self, answer: IEverythingAnswer) -> List[float]:
        solverAns :List[float] = []
        for i in self._indicesForConstraintsFromBoundaryConditions:
            solverAns.append(answer.BoundaryConditionValues[i])
        return solverAns




class fSolveSingleShootingSolver(singleShootingSolver):
    def __init__(self, evaluatableProblem:EverythingProblem, controlsForSolver, constraintsForSolver ):
        super().__init__(evaluatableProblem, controlsForSolver, constraintsForSolver)

    def solve(self, initialGuessOfSolverControls : List[float], time : Iterable[float], initialState : List[float], parameters: List[float], **kwargs) ->solverAnswer:
        # fsolve(bcCallback, fSolveInitialGuess, full_output=True,  factor=0.2,epsfcn=0.001 )

        def evaluateAnswer(fSolveValues, time, initialState) -> IEverythingAnswer:
            copiedInitialState = initialState.copy()
            self.updateInitialStateWithSolverValues(fSolveValues, copiedInitialState)

            
            if not(parameters == None or len(parameters) == 0):
                argsCopy = list(parameters)
                finalArgs : Tuple[float, ...]= tuple(self.updateInitialParametersWithSolverValues(fSolveValues, argsCopy))
            else:
                finalArgs =tuple()
            

            everythingAns = self.EvaluatableProblem.EvaluateProblem(time, copiedInitialState, finalArgs)
            return everythingAns   

        def fSolveCallback(fSolveGuess, *parametersForFSolve):
            # fsolve wraps the guess in an array, so pull it out
            everythingAns = evaluateAnswer(fSolveGuess, time, initialState)
            solvedForValues = self.getSolverValuesFromEverythingAns(everythingAns)
            while len(solvedForValues) < len(fSolveGuess):
                solvedForValues.append(0.0)
            print(solvedForValues)
            return solvedForValues

        fsolveAns = self.fsolveRun(fSolveCallback, initialGuessOfSolverControls, **kwargs)
        fsolveX = fsolveAns[0]
        if isinstance(fsolveX , float):
            fsolveX = fsolveAns
        finalRun = evaluateAnswer(fsolveX, time, initialState)
        return solverAnswer(finalRun, fsolveX, self.getSolverValuesFromEverythingAns(finalRun), fsolveAns)

    def fsolveRun(self, solverFunc, solverState, **kwargs):
        return fsolve(solverFunc, solverState, **kwargs)
