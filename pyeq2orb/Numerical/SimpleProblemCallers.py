from pyeq2orb.ProblemBase import Problem, ProblemVariable
from pyeq2orb import SafeSubs
from pyeq2orb.NumericalOptimizerProblem import NumericalOptimizerProblemBase
import sympy as sy
from IPython.display import display
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
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
    def __init__(self, timeHistory: List[float], StateHistory : Dict[sy.Symbol, List[float]], rawIntegratorOutput : Optional[Any] = None):
        """Initializes a new instance.

        Args:
            timeHistory (List[float]): The time history.
            StateHistory (Dict[sy.Symbol, List[float]]): The states separated into lists of floats keyed off of the symbols.
            rawIntegratorOutput (Optional[Any], optional): The raw integrator output. Defaults to None.
        """
        self._timeHistory = timeHistory
        self._stateHistory = StateHistory
        self._rawIntegratorOutput= rawIntegratorOutput

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



class SimpleEverythingAnswer(SimpleIntegrationAnswer, IEverythingAnswer):
    """An IEverythingAnswer that is just wrapping underlying lists and dictionaries of the 
    values.

    Args:
        IEverythingAnswer: This is an IEverythingAnswer and a SimpleIntegrationAnswer.
    """
    def __init__(self, timeHistory: List[float], stateHistory : Dict[sy.Symbol, List[float]], bcAnswer : List[float], rawIntegratorOutput = Optional[Any]):
        """"Initializes a new instance.

        Args:
            timeHistory (List[float]): The time history.
            StateHistory (Dict[sy.Symbol, List[float]]): The states separated into lists of floats keyed off of the symbols.
            bcAnswer (List[float]): The values of the boundary conditions.
            rawIntegratorOutput (Optional[Any], optional): The raw integrator output. Defaults to None.
        """
        self._timeHistory = timeHistory # I should look into calling IIntegrationAnswer specifically, but eh
        self._stateHistory = stateHistory
        self._boundaryConditionValues = bcAnswer
        self._rawIntegratorOutput = rawIntegratorOutput 

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
    def StateVariables(self) ->List[sy.Symbol]:
        """A list of the state variable symbols (normal state variables, costates, path constraines... doesn't matter, all 
        variables getting propagated through time).

        Returns:
            List[sy.Symbol]: The symbols getting propagated.
        """
        pass

    @property
    @abstractmethod
    def BoundaryConditionExpressions(self) -> List[sy.Expr]:
        """Returns a list of the symbolic expressions for the boundary conditions.

        Returns:
            List[sy.Expr]: The symbolic expressions for the boundary conditions.
        """
        pass

    @abstractmethod
    def EvaluateProblem(self, time : List[float], initialState : List[float], parameters : Tuple[float]) ->IEverythingAnswer:
        """Evaluates this problem returning an IEverythingAnswer.

        Args:
            time (Iterable[float]): The points in time to evaluate.
            initialState (List[float]): The initial state to propagate from.
            parameters (Tuple[float]): Any additional arguments to be passed to both the propagation 
            of the trajectory AND the boundary conditions.

        Returns:
            IEverythingAnswer: The evaluated trajectory and boundary conditions.
        """
        pass




class SingleShootingFunctions(EverythingProblem, ABC):
    def __init__(self):
        pass

    @abstractmethod
    def IntegrateDifferentialEquations(self, time : List[float], y0 : List[float], args : Tuple[float]) -> IIntegrationAnswer:
        """Performs the full integration of the problem.  This function will wrap your call to solve_ivp or ode_solve or 
        whatever integration scheme you want.

        Args:
            time (List[float]): The times to evaluate the problem at.
            y0 (List[float]): _description_
            args (Tuple[float]): _description_

        Returns:
            IIntegrationAnswer: _description_
        """
        pass

    def BuildBoundaryConditionStateFromIntegrationAnswer(self, integrationAnswer: IIntegrationAnswer) -> List[float]:
        """Builds a complete enough state to be able to evaluate boundary conditions, assuming that 
        they depend only on the initial and end states of propagation (since they are, you know, 
        boundary conditions).

        Args:
            integrationAnswer (IIntegrationAnswer): The integration results.

        Returns:
            List[float]: The initial time, initial state, final time and final state all in one simple list.
        """
        time = integrationAnswer.TimeHistory
        solutionDict = integrationAnswer.StateHistory
        bcState = []
        bcState.append(time[0])
        for k, v in solutionDict.items():
            bcState.append(v[0])
        bcState.append(time[-1])
        for k, v in solutionDict.items():
            bcState.append(v[-1])         
        # don't need args since they are passed in at evaluation time of the BC's   
        return bcState

    @abstractmethod
    def BoundaryConditionEvaluation(self, fullBcState, args) -> List[float]:
        """Evaluates the boundary conditions.  Given a 'full boundary condition state' 

        Args:
            fullBcState (_type_): _description_
            args (_type_): _description_

        Returns:
            List[float]: _description_
        """
        pass
    
    def EvaluateProblem(self, time : List[float], initialState : List[float], parameters : Tuple[float]) -> IEverythingAnswer:
        ivpAns = self.IntegrateDifferentialEquations(time, initialState, parameters)
        bcState = self.BuildBoundaryConditionStateFromIntegrationAnswer(ivpAns)
        bcAns = self.BoundaryConditionEvaluation(bcState, parameters)
        return SimpleEverythingAnswer(ivpAns.TimeHistory, ivpAns.StateHistory, bcAns, ivpAns.RawIntegratorOutput)


class BlackBoxSingleShootingFunctions(SingleShootingFunctions):
    def __init__(self, integrationCallback, boundaryConditionCallback,  stateVariables, boundaryConditionExpressions):
        self._integrationCallback = integrationCallback
        self._boundaryConditionCallback = boundaryConditionCallback
        self._stateVariables = stateVariables
        self._boundaryConditionExpressions = boundaryConditionExpressions

    @property
    def StateVariables(self) ->List[sy.Symbol]:
        return self._stateVariables

    @property
    def BoundaryConditionExpressions(self) -> List[sy.Expr]:
        return self._boundaryConditionExpressions

    def IntegrateDifferentialEquations(self, t, y, args):
        return self._integrationCallback(t, y, args)

    def BoundaryConditionEvaluation(self, fullBcState, args):     
        # if args == None:
        #     args = ((),)
        return self._boundaryConditionCallback(fullBcState, args)




class solverAnswer:
    def __init__(self, evaluatedAnswer : IEverythingAnswer, solvedControls : List[float], constraintValues: List[float], solverResult: Optional[Any] = None):
        self.EvaluatedAnswer =evaluatedAnswer
        self.SolvedControls = solvedControls
        self.ConstraintValues = constraintValues
        self.SolverResult = solverResult




class singleShootingSolver(ABC):
    def __init__(self, originalProblem : Problem, evaluatableProblem:EverythingProblem, controlsForSolver, constraintsForSolver ):
        self._indicesOfInitialStateToUpdate : List[int] = []
        self._indicesOfProblemParametersToUpdate : List[int] = []
        overallBcStateValuesToSample :List[int] = []

        for control in controlsForSolver:
            if control in evaluatableProblem.StateVariables:
                thisControlIndex = evaluatableProblem.StateVariables.index(control)
                self._indicesOfInitialStateToUpdate.append(thisControlIndex)
                continue
            thisControlIndex = originalProblem.OtherArguments.index(control)
            self._indicesOfProblemParametersToUpdate.append(thisControlIndex)

        self.EvaluatableProblem = evaluatableProblem

        self._indicesForConstraintsFromBoundaryConditions :List[int]= []
        for constraint in constraintsForSolver:
            thisConstraintIndex = originalProblem.BoundaryConditions.index(constraint)
            self._indicesForConstraintsFromBoundaryConditions.append(thisConstraintIndex)

    @abstractmethod
    def solve(self, initialGuessOfSolverControls : List[float], time, initialState : List[float], parameters : Tuple[float], *solverArgs, **solverKwargs) -> solverAnswer:
        pass

    def putInitialSolverGuessIntoEverythingProblemState(self, initialGuess, time : List[float], initialState : List[float], initialParameters : Tuple[float]) -> Tuple[List[float], List[float], Tuple[float]]:
        editedState = [*initialState]
        countOfValuesToUpdateInTheState = len(self._indicesOfInitialStateToUpdate)
        for i in range(0, countOfValuesToUpdateInTheState):
            thisIndex = self._indicesOfInitialStateToUpdate[i]
            editedState[thisIndex] = initialGuess[i]

        editedParameters = []
        if isinstance(initialParameters, float):
            editedParameters = [initialParameters]
        elif not initialParameters is None:
            editedParameters = [*initialParameters]
        for i in range(0, len(self._indicesOfProblemParametersToUpdate)):
            thisIndex = self._indicesOfProblemParametersToUpdate[i]
            editedParameters[thisIndex] = initialGuess[countOfValuesToUpdateInTheState+i-1]
        
        return (time, editedState, tuple(editedParameters)) #type: ignore


    def getSolverValuesFromEverythingAns(self, answer: IEverythingAnswer) -> List[float]:
        solverAns :List[float] = []
        for i in self._indicesForConstraintsFromBoundaryConditions:
            solverAns.append(answer.BoundaryConditionValues[i])
        return solverAns




class fSolveSingleShootingSolver(singleShootingSolver):
    def __init__(self, originalProblem : Problem, evaluatableProblem:EverythingProblem, controlsForSolver, constraintsForSolver ):
        super().__init__(originalProblem, evaluatableProblem, controlsForSolver, constraintsForSolver)

    def solve(self, initialGuessOfSolverControls : List[float], time, initialState : List[float], parameters : Tuple[float], *args, **kwargs) ->solverAnswer:
        # fsolve(bcCallback, fSolveInitialGuess, full_output=True,  factor=0.2,epsfcn=0.001 )

        def evaluateAnswer(fSolveValues, time, initialState, parameters) -> IEverythingAnswer:
            [realTime, realInitalState,realParameters] = self.putInitialSolverGuessIntoEverythingProblemState(fSolveValues, time, initialState, parameters)
            everythingAns = self.EvaluatableProblem.EvaluateProblem(realTime, realInitalState,realParameters)     
            return everythingAns   

        def fSolveCallback(fSolveGuess, parameters):
            # fsolve wraps the guess in an array, so pull it out
            everythingAns = evaluateAnswer(fSolveGuess, time, initialState, parameters)
            solvedForValues = self.getSolverValuesFromEverythingAns(everythingAns)
            return solvedForValues

        if parameters != None and len(parameters) > 0 and not "args" in kwargs:
            kwargs["args"] = parameters
        # for fsolve, there are no real args other than the required ones
        fsolveAns = self.fsolveRun(fSolveCallback, initialGuessOfSolverControls, parameters, **kwargs)# fsolve(fSolveCallback, initialGuessOfSolverControls, *args, **kwargs)
        fsolveX = fsolveAns[0]
        if isinstance(fsolveX , float):
            fsolveX = fsolveAns
        finalRun = evaluateAnswer(fsolveX, time, initialState, parameters)
        return solverAnswer(finalRun, fsolveX, self.getSolverValuesFromEverythingAns(finalRun), fsolveAns)

    def fsolveRun(self, solverFunc, solverState, parameters, **kwargs):
        return fsolve(solverFunc, solverState, parameters, **kwargs)
