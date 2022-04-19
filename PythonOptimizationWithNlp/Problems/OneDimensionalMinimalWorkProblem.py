from typing import List, Dict
import sympy as sy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from collections import OrderedDict
from PythonOptimizationWithNlp.NumericalOptimizerProblem import NumericalOptimizerProblemBase
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
from PythonOptimizationWithNlp.Utilities.inherit import inherit_docstrings

@inherit_docstrings
class OneDWorkSymbolicProblem(SymbolicProblem) :
    def __init__(self) :
        """Initializes a new instance.
        """
        super().__init__()
        self.Ts = sy.Symbol('t')
        self._t0 = sy.Symbol('t_0')
        self._tf = sy.Symbol('t_f')
        self._stateVariables = [sy.Function('x')(self.Ts), sy.Function('v')(self.Ts)]
        self._controlVariables = [sy.Function('u')(self.Ts)]
        self._equationsOfMotion = {self._stateVariables[0]:self._stateVariables[1], self._stateVariables[1] : self._controlVariables[0]}
        self._pathConstraints = {}
        self._constantSymbols = []
        self._unIntegratedPathCost = self._controlVariables[0]**2
        self._finalBoundaryConditions = [sy.Function('x')(self._tf) -1, sy.Function('v')(self._tf)]

    @property
    def StateVariables(self) -> List[sy.Symbol]:
        return self._stateVariables

    @property
    def ControlVariables(self) -> List[sy.Symbol]:
        return self._controlVariables

    @property
    def EquationsOfMotion(self) -> Dict[sy.Symbol, sy.Expr]:
        return self._equationsOfMotion

    @property
    def FinalBoundaryConditions(self) ->List[sy.Expr] : #TODO: Rename to just boundary conditions
        return self._finalBoundaryConditions

    @property
    def PathConstraints(self) -> Dict[sy.Symbol, sy.Expr] :
        return self._pathConstraints
    
    @property
    def ConstantSymbols(self) -> List[sy.Symbol] :
        return self._constantSymbols

    @property
    def TimeSymbol(self) -> sy.Expr :
        return self.Ts

    @property 
    def Time0Symbol(self) -> sy.Expr :
        return self._t0

    @property 
    def TimeFinalSymbol(self) -> sy.Expr :     
        return self._tf

    @property
    def TerminalCost(self) -> sy.Expr :     
        return 0.0

    @property
    def UnIntegratedPathCost(self) -> sy.Expr :  
        return self._unIntegratedPathCost

class OneDWorkProblem(NumericalOptimizerProblemBase) :
    """A very simple and straight forward optimal control problem.  Although there is an analytical solution 
    to be had, this is great for testing plumbing and connections between types.  As such, this type is 
    heavily used in unit tests.

    Args:
        NumericalOptimizerProblemBase (NumericalOptimizerProblemBase): This type extends NumericalOptimizerProblemBase.
    """

    def __init__(self) :
        """Initializes a new instance.  
        There will be 2 state variables, x (position of the block) and v (speed of the block), and 1 control variable 
        which ends up being the acceleration of the block.  The time range is [0.0, 1.0].  X boundary conditions are x(0)=0.0
        and x(1.0) = 1.0, and the speed boundary conditions are both 0.
        
        Args:
            n (int): The number of segments to discretize the solution into.
        """
        tSy = sy.Symbol('t')

        super().__init__(tSy)
        
        xSy = sy.Symbol('x')
        vSy = sy.Symbol('v')
        uSy = sy.Symbol('u')

        t0 = 0.0
        tf = 1.0
        x0Bc = 0.0
        vx0Bc = 0.0
        xfBc = 1.0
        vxfBc = 0.0

        self.T0 = t0
        self.Tf = tf
        # note that this class is responsible for the order of the states, 
        # other functions below can make the assumption, and it would be 
        # a breaking change to change the order.
        self.State.append(xSy)
        self.State.append(vSy)
        self.Control.append(uSy)

        self.InitialBoundaryConditions[xSy] = x0Bc
        self.InitialBoundaryConditions[vSy] = vx0Bc

        self.FinalBoundaryConditions[xSy] = xfBc
        self.FinalBoundaryConditions[vSy] = vxfBc
   
    def InitialGuessCallback(self, t : float) -> List[float] :
        """A function to produce an initial state at t, 

        Args:
            t (float): _description_

        Returns:
            List[float]: A list the values in the state followed by the values of the controls at t.
        """
        if t == self.T0 :
            return [self.InitialBoundaryConditions[self.State[0]], self.InitialBoundaryConditions[self.State[1]], 0.0]
        if t == self.Tf :
            return [self.FinalBoundaryConditions[self.State[0]], self.FinalBoundaryConditions[self.State[1]], 0.0]
        else :
            v = -1.0
            if(t <= (self.Tf-self.T0)/2.0) :
                v = 1.0            
            return [t, v, v/10.0]

    def EquationOfMotion(self, t : float, stateAndControlAtT : List[float]) -> List[float] :
        """The equations of motion.  

        Args:
            t (float): The time.  
            stateAndControlAtT (List[float]): The current state and control at t to evaluate the equations of motion.

        Returns:
            List[float]: The derivative of the state variables 
        """        
        return [stateAndControlAtT[1], stateAndControlAtT[2]]

    def CostFunction(self, t : List[float], stateAndControl : Dict[object, List[float]]) -> float:
        """The cost of the 1D block-moving problem.  The quadratic sum of the control values.

        Args:
            t (List[float]): The time array being evaluated.
            stateAndControl (Dict[object, List[float]]): The state and control.

        Returns:
            float: The cost.
        """
        uCostArray = []
        control = stateAndControl[self.Control[0]]
        for i in range(0, len(control)-1) :
            step = t[i+1]-t[i]
            uCostArray.append(0.5*(step)*(control[i+1]**2+control[i]**2))
        return sum(uCostArray)
    
    def AddResultsToFigure(self, figure : Figure, t : List[float], dictionaryOfValueArraysKeyedOffState : Dict[object, List[float]], label : str) -> None :
        """Adds the contents of dictionaryOfValueArraysKeyedOffState to the plot.

        Args:
            figure (matplotlib.figure.Figure): The figure the data is getting added to.
            t (List[float]): The time corresponding to the data in dictionaryOfValueArraysKeyedOffState.
            dictionaryOfValueArraysKeyedOffState (Dict[object, List[float]]): The x, v, and u to be plotted.
            label (str): A label for the data to use in the plot legend.
        """
        xAct = dictionaryOfValueArraysKeyedOffState[self.State[0]] #TODO: I don't like how this line knows that the 0th element is X (and the next two too)
        vAct = dictionaryOfValueArraysKeyedOffState[self.State[1]]
        uAct = dictionaryOfValueArraysKeyedOffState[self.Control[0]]
        plt.subplot(311)
        plt.title("1D Work Problem")
        plt.plot(t, xAct, label=label)
        plt.ylabel('x')

        plt.subplot(312)
        plt.plot(t, vAct, label=label)
        plt.ylabel('v_x')

        plt.subplot(313)
        plt.plot(t, uAct, label=label)
        plt.ylabel('u')
        plt.legend()


class AnalyticalAnswerToProblem :
    """A class holding the functions needed to solve the optimal block-moving example analytically.
    """
    def __init__(self) :
        """Initialize a new instance.
        """
        pass

    def OptimalX(self, time) :
        """Evaluates the optimal position.  This function takes advantage of the duck-typing of python 
        such that as long as time can be added, multiplied and raised to a power, you will get the expected 
        value back.

        Args:
            time : The time to evaluate at

        Returns:
            object : The optimal X for this problem.
        """
        return 3.0*time**2.0-2.0*time**3.0

    def OptimalV(self, time) :
        """Evaluates the optimal velocity.  This function takes advantage of the duck-typing of python 
        such that as long as time can be added, multiplied and raised to a power, you will get the expected 
        value back.

        Args:
            time : The time to evaluate at

        Returns:
            object : The optimal velocity of the block for this problem.
        """        
        return 6.0*time-6.0*time**2.0

    def OptimalControl(self, time) :
        """Evaluates the optimal control.  This function takes advantage of the duck-typing of python 
        such that as long as time can be added, multiplied and raised to a power, you will get the expected 
        value back.

        Args:
            time : The time to evaluate at

        Returns:
            object : The optimal control value for this problem.
        """           
        return 6.0-12.0*time

    def EvaluateAnswer(self, oneDWorkProblem : OneDWorkProblem, t : List[float]) -> Dict[object, List[float]]:
        """Evaluates the optimal solution for this problem.

        Args:
            oneDWorkProblem (OneDWorkProblem): A numerical version of the problem to provide additional data.
            t (List[float], optional): Optional array of time values, if not set it will be created from the 
            oneDWorkProblem.

        Returns:
            Dict[object, List[float]]: The analytical solution to this problem evaluated analytically, in a form 
            that is ready to plot with the plotting function in the oneDWorkProblem.
        """
        n = len(t)
        if(t == None) :
            t = oneDWorkProblem.CreateTimeRange(n)
        t=np.array(t)
        optXValues = self.OptimalX(t)
        optVValues = self.OptimalV(t)
        optUValues = self.OptimalControl(t)
        analytical = {oneDWorkProblem.State[0]:optXValues, oneDWorkProblem.State[1]:optVValues, oneDWorkProblem.Control[0]:optUValues}
        return analytical
