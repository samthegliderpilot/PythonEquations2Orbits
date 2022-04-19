from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from matplotlib.figure import Figure

class NumericalOptimizerProblemBase(ABC) :
    """ A base type for the kinds of numerical optimization problems I hope to solve. 

    Args:
        ABC (abc.ABC): This class a abstract base class.  Several functions are intended to be implimented by derived types.
    """
    def __init__(self, t : object) :
        """Initializes a new instance

        Args:
            t (object): Some identifier for the independent variable (often time).  This is generally a sympy Symbol or a string.
            n (int): The count of segments that the trajectory will be broken up by.
        """
        self.State = []
        self.InitialBoundaryConditions = {}
        self.FinalBoundaryConditions = {}
        self.Time = t
        self.T0 = 0
        self.Control = []        
    
    @property
    def NumberOfStateVariables(self) ->int :
        """Returns the number of state variables.  This is the count BEFORE transcription.
        
        Returns:
            int: The number of state variables.
        """
        return len(self.State)

    @property
    def NumberOfControlVariables(self) ->int :
        """Returns the number of control variables.  This is the count BEFORE transcription.
        
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
    
    @abstractmethod
    def InitialGuessCallback(self, t : float) -> List[float] :
        """A function to produce an initial state at t, 

        Args:
            t (float): _description_

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

    @abstractmethod 
    def CostFunction(self, t : List[float], stateAndControl : Dict[object, List[float]]) -> float:
        """The cost of the problem.  

        Args:
            t (: List[float]): The time array being evaluated.
            state (: List[float]): The state as seen by the optimizer (the discretized x's and v's)
            control (: List[float]): The control as controlled by the optimizer (the discretized u).

        Returns:
            float: The cost.
        """
        pass

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

