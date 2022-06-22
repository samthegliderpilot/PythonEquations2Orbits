from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
from matplotlib.figure import Figure
import PythonOptimizationWithNlp.Utilities.SolutionDictionaryFunctions as DictionaryHelper
class NumericalOptimizerProblemBase(ABC) :
    """ A base type for the kinds of numerical optimization problems I hope to solve. 

    Args:
        ABC (abc.ABC): This class a abstract base class.  Several functions are intended to be implemented by derived types.
    """
    def __init__(self, t : object) :
        """Initializes a new instance

        Args:
            t (object): Some identifier for the independent variable (often time).  This is generally a sympy Symbol or a string.
            n (int): The count of segments that the trajectory will be broken up by.
        """
        self.State = []
        self._boundaryConditionCallbacks = []
        self.Time = t
        self.T0 = 0
        self.Control = []        
        self._knownFinalConditions = {}
        self._knownInitialConditions = {}
    
    @property
    def NumberOfStateVariables(self) ->int :
        """Returns the number of state variables.  
        
        Returns:
            int: The number of state variables.
        """
        return len(self.State)

    @property
    def NumberOfControlVariables(self) ->int :
        """Returns the number of control variables.  
        
        Returns:
            int: The number of control variables.
        """
        return len(self.Control)

    @property
    def NumberOfOptimizerStateVariables(self) ->int :
        """Returns the number of variables that are fed to the optimizer (the count of state + control variables).  This is the count BEFORE transcription.
        
        Returns:
            int: The number of variables that are fed to the optimizer (the count of state + control variables).
        """        
        return self.NumberOfControlVariables+self.NumberOfStateVariables

    def CreateTimeRange(self, n) ->List[float]:
        """Creates a default evenly spaced array of time values between self.T0 and self.Tf.

        Returns:
            List[float]: An array of the time values.
        """
        # this is a function that is likely to get overridden in more complicated problems        
        t = []
        step = (self.Tf-self.T0)/n
        for i in range(0, n) :
            t.append(self.T0+i*step)
        t.append(self.Tf)
        return t
    
    def InitialTrajectoryGuess(self,n :int, t0:float, stateAndControlAtT0 : List[float], tf : float, stateAndControlAtTf : List[float]) -> Dict[object, List[float]] :
        """Provides an initial guess for the overall trajectory.  By default this does a linear interpolation from the initial 
        to final conditions, but you are encouraged to override this method with something more refined (maybe integrate the 
        equations of motion making some reasonable guess to the control variables). 

        Args:
            n (int): Number of steps to take
            t0 (float): Initial time
            stateAndControlAtT0 (List[float]): Initial state and control
            tf (float): Final time
            stateAndControlAtTf (List[float]: Final state and control

        Returns:
            Dict[object, List[float]]: A guess for the initial trajectory that better be good enough for a solver of some sort. This will 
            include the time and control histories as well.
        """
        solution = {}
        
        for i in range(0, self.NumberOfStateVariables) :
            solution[self.State[i]]= [stateAndControlAtT0[i] + x*(stateAndControlAtTf[i]-stateAndControlAtT0[i])/n for x in range(n+1)]
        for i in range(self.NumberOfStateVariables, self.NumberOfOptimizerStateVariables) :            
            solution[self.Control[i-self.NumberOfStateVariables]] = [stateAndControlAtT0[i] + x*(stateAndControlAtTf[i]-stateAndControlAtT0[i])/n for x in range(n+1)]
        solution[self.Time] = list(np.linspace(t0, tf, n))
        return solution

    def InitialGuessCallback(self, t : float) -> List[float] :
        """A function to produce an initial state at t, 

        Args:
            t (float): the time to get the state and control at.

        Returns:
            List[float]: A list the values in the state followed by the values of the controls at t.
        """
        pass

    @abstractmethod
    def EquationOfMotion(self, t : float, stateAndControlAtT : List[float]) -> List[float] :
        """The equations of motion.  

        Args:
            t (float): The time.  
            stateAndControlAtT (List[float]): The current state and control at t to evaluate the equations of motion.

        Returns:
            List[float]: The derivative of the state variables 
        """  
        pass

    def SingleEquationOfMotion(self, t : float, stateAndControlAtT : List[float], indexOfEom : int) -> float :
        """Some solvers may require that 

        Args:
            t (float): _description_
            stateAndControlAtT (List[float]): _description_
            indexOfEom (int): _description_

        Returns:
            float: _description_
        """
        return self.EquationOfMotion(t, stateAndControlAtT)[indexOfEom]

    def ListOfEquationsOfMotionCallbacks(self) -> List :
        callbacks = []
        for i in len(self.State) :
            b =i*2 # pretty sure this trick to get the loop variable captured properly isn't going to work
            callback = lambda t, stateAndControlAtT : self.SingleEquationOfMotion(t, stateAndControlAtT, int(b/2))
            callbacks.append(callback)
        return callbacks

    def CostFunction(self, t : List[float], stateAndControl : Dict[object, List[float]]) -> float:
        """The cost of the problem.  

        Args:
            t (: List[float]): The time array being evaluated.
            state (: List[float]): The state as seen by the optimizer (the discretized x's and v's)
            control (: List[float]): The control as controlled by the optimizer (the discretized u).

        Returns:
            float: The cost.
        """
        finalState = DictionaryHelper.GetFinalStateDictionary(stateAndControl)
        return self.TerminalCost(t[-1], finalState) + self.IntegratePathCost(t, stateAndControl)

    @abstractmethod 
    def UnIntegratedPathCost(self, t : float, stateAndControl : List[float]) -> float :
        """Evaluates the path cost at t.

        Args:
            t (float): The current time.
            stateAndControl (List[float]): The state and control at the time as an array.

        Returns:
            float: The un-integrated path cost at the time.
        """
        pass
    
    def IntegratePathCost(self, tArray, stateAndControlDict) -> float :
        """It is highly encouraged to override this function.
        
        Integrates the UnIntegratedPathCost.  Note that if you are using a single shooting method, 
        it is recommended to instead set up another "equation of motion" of the UnIntegratedPathCost 
        with an initial value of and make the cost function pull out the final value of this.

        By default this does a default scipy.integrate.simp on an array of values created by 
        evaluating the UnIntegratedPathCost over the time array and states and controls.

        Args:
            tArray (_type_): The time array that have been evaluated already.
            stateAndControlDict (_type_): The time history of the state and control.

        Returns:
            float: The integrated value of the path cost.
        """
        from scipy.integrate import simps
        unintegratedValues = []
        for i in range(0, len(tArray)) :
            stateNow = DictionaryHelper.GetValueFromStateDictionaryAtIndex(stateAndControlDict, i)
            z = tuple(self.ConvertStateAndControlDictionaryToArray(stateNow))
            unintegratedValues.append(self.UnIntegratedPathCost(tArray[i], z))
        value = simps(unintegratedValues, x=tArray)
        
        return value

    def ConvertStateAndControlDictionaryToArray(self, stateAndControlDict : Dict[object, object]) -> List[object] :
        """Provides a way to convert a dictionary to an array that other types may
        prefer (such as integrators where the order of values getting integrated need to align with 
        the order of the equations of motion).  This is a function that should often be overridden 
        if there are additional items getting integrated (such as a path cost or an auxiliary variable).

        Args:
            stateAndControlDict (Dict[object, object]): The dictionary mapping state and controls (and 
            potentially other things) that are related to function evaluations.

        Returns:
            List[object]: The values from the dictionary in a standard order.
        """
        z = []
        for sv in self.State :            
            z.append(stateAndControlDict[sv])
        for sv in self.Control :
            z.append(stateAndControlDict[sv])            
        return z

    @abstractmethod 
    def TerminalCost(self, tf : float, finalStateAndControl : Dict[object, float]) -> float:
        """Evaluates the terminal cost from the final values of the function.

        Args:
            tf (float): The final time
            finalStateAndControl (Dict[object, float]): The final state.

        Returns:
            float: The terminal cost.
        """
        pass # technically, we don't need to pass the final control into the terminal cost function, but it is more trouble to not include it and on the off chance it helps in some specific problem

    @abstractmethod    
    def AddResultsToFigure(self, figure : Figure, t : List[float], dictionaryOfValueArraysKeyedOffState : Dict[object, List[float]], label : str) -> None:
        """Adds the contents of dictionaryOfValueArraysKeyedOffState to the plot.

        Args:
            figure (matplotlib.figure.Figure): The figure the data is getting added to.
            t (List[float]): The time corresponding to the data in dictionaryOfValueArraysKeyedOffState.
            dictionaryOfValueArraysKeyedOffState (Dict[object, List[float]]): The data to get added.  The keys must match the values in self.State and self.Control.
            label (str): A label for the data to use in the plot legend.
        """
        pass

    @property
    def KnownInitialConditions(self) ->Dict[object, float] :
        """A dictionary of the known initial conditions. Solvers need to take these into account (either 
        by making boundary conditions) or some other way when solving problems.  This may not be a complete 
        set of conditions, however it is assumed it is enough for the problem to be solvable.

        Returns:
            Dict[object, float]: Known initial conditions, the keys of which are the objects in the State.
        """
        return self._knownInitialConditions

    @property
    def KnownFinalConditions(self) ->Dict[object, float] :
        """A dictionary of the known final conditions. Solvers need to take these into account (either 
        by making boundary conditions) or some other way when solving problems.  This may not be a complete 
        set of conditions, however it is assumed it is enough for the problem to be solvable.

        Returns:
            Dict[object, float]: Known final conditions, the keys of which are the objects in the State.
        """        
        return self._knownFinalConditions

    @property
    def BoundaryConditionCallbacks(self) -> List[Callable[[float, List[float], float, List[float], float]]] :
        """A list of callbacks for additional boundary conditions.  These callbacks take the initial time, 
        initial state, final time and final state in that order.  You can set up initial and final state values 
        here if desired, but if they are simple constants, it is recommended to use the KnownInitialConditions
        and KnownFinalConditions.

        Returns:
            List[Callable[[float, List[float], float, List[float], float]]]: Additional boundary constraints.
        """
        return self._boundaryConditionCallbacks