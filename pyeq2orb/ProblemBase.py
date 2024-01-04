import sympy as sy
from typing import List, Dict
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from abc import ABC
from collections import OrderedDict
from enum import Enum
import pyeq2orb.Numerical.LambdifyHelpers as LH
class IntegrationDirection(Enum) :
    Forward = 1
    Backward = -1

class Problem(ABC) :
    def __init__(self) :
        """Initialize a new instance. 
        """
        self._stateVariables = []
        self._controlVariables = []
        self._terminalCost = 0
        self._unIntegratedPathCost = 0
        self._equationsOfMotion = []
        self._boundaryConditions = []
        self._timeSymbol = None
        self._timeInitialSymbol = None
        self._timeFinalSymbol= None
        self._integrationDirection = IntegrationDirection.Forward
        self._substitutionDictionary = OrderedDict()

    @property
    def SubstitutionDictionary(self) -> Dict[sy.Expr, float] :
        """The dictionary that should be used to store constant values that may appear 
        in the various expressions.  Many helper functions elsewhere want this dictionary 
        passed to it.

        Returns:
            Dict[sy.Expr, float]: The expression-to-values to substitute into expressions.
        """
        return self._substitutionDictionary

    @property
    def StateVariables(self) -> List[sy.Expr]:
        """Gets the state variables for this problem.  These should be in terms of TimeSymbol. 
        This must be implemented by the derived type.

        Returns:
            List[sy.Expr]: The list of symbols in terms of TimeSymbol
        """
        return self._stateVariables
    
    @property
    def ControlVariables(self) -> List[sy.Symbol]:
        """Gets a list of the control variables.  These should be in terms of TimeSymbol. 
        This must be implemented by the derived type.

        Returns:
            List[sy.Expr]: The list of the control variables.
        """
        return self._controlVariables

    @property
    def CostFunction(self) -> sy.Expr :
        """Gets the cost function as an expression.  This combines the TerminalCost with 
        the integrated UnIntegratedPathCost over time.

        Returns:
            sy.Expr: The overall cost function.
        """
        return self.TerminalCost + sy.integrate(self.UnIntegratedPathCost, (self.TimeSymbol, self.TimeInitialSymbol, self.TimeFinalSymbol))

    @property
    def TerminalCost(self) -> sy.Expr :
        """Gets the terminal cost of the problem.  Defaults to 0.

        Returns:
            sy.Expr: The terminal cost of the problem.
        """        
        return self._terminalCost

    @TerminalCost.setter
    def TerminalCost(self, value : sy.Expr) :
        """Sets the Terminal Cost of the function.

        Args:
            value (sy.Expr): The new terminal cost of the function.
        """
        self._terminalCost = value

    @property
    def UnIntegratedPathCost(self) -> sy.Expr :
        """Gets the un-integrated path cost of the trajectory.  For a problem of Bolza, this is the expression in the integral.

        Returns:
            sy.Expr: The un-integrated path cost of the constraint.  
        """        
        return self._unIntegratedPathCost

    @UnIntegratedPathCost.setter
    def UnIntegratedPathCost(self, value: sy.Expr) :
        """Sets the un-integrated path cost of the trajectory.  For a problem of Bolza, this is the expression in the integral.

        Args:
            value (sy.Expr): The un-integrated path cost.
        """
        self._unIntegratedPathCost = value

    @property
    def EquationsOfMotion(self) -> List[sy.Eq]:
        """Gets the equations of motion for each of the state variables.  The LHS should be 
        the derivative of the state variable with respect to time, and the RHS should be the 
        expression (for example, dm/dt = mDot*t, but as symbols)

        Returns:
            List[sy.Eq]: The list of equations of motion for each of the state variables.
        """
        return self._equationsOfMotion
    
    @property
    def BoundaryConditions(self) ->List[sy.Expr] :
        """Gets the boundary conditions on the system.  These expressions 
        must equal 0 and symbols in them need to be in terms of Time0Symbol 
        or TimeFinalSymbol as appropriate.

        Returns:
            List[sy.Eq]: The boundary conditions. 
        """
        return self._boundaryConditions

    @property
    def TimeSymbol(self) -> sy.Symbol :
        """Gets the general time symbol.  Instead of using simple symbols for the state and 
        control variables, use sy.Function()(self.TimeSymbol) instead.

        Returns:
            sy.Expr: The time symbol.
        """        
        return self._timeSymbol

    @TimeSymbol.setter
    def TimeSymbol(self, value:sy.Symbol) :
        """Sets the general time symbol.  Instead of using simple symbols for the state and 
        control variables, use sy.Function()(self.TimeSymbol) instead.

        Args:
            value (sy.Expr): The time symbol. 
        """
        self._timeSymbol = value

    @property
    def TimeInitialSymbol(self) -> sy.Symbol :
        """Gets the symbol for the initial time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the start of the time span.

        Returns:
            sy.Expr: The initial time symbol.
        """        
        return self._timeInitialSymbol

    @TimeInitialSymbol.setter
    def TimeInitialSymbol(self, value:sy.Symbol) :
        """Sets the symbol for the initial time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the start of the time span.

        Args:
            value (sy.Expr): The initial time symbol.
        """
        self._timeInitialSymbol = value

    @property
    def TimeFinalSymbol(self) -> sy.Symbol :
        """Gets the symbol for the final time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the end of the time span.

        Returns:
            sy.Expr: The final time symbol.
        """        
        return self._timeFinalSymbol

    @TimeFinalSymbol.setter
    def TimeFinalSymbol(self, value : sy.Symbol) :
        """Sets the symbol for the final time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the end of the time span.

        Args:
            value (sy.Expr): The final time symbol.
        """
        self._timeFinalSymbol = value
    
    @property
    def Direction(self) -> IntegrationDirection:
        return self._integrationDirection

    @Direction.setter
    def Direction(self, value : IntegrationDirection) :
        self._integrationDirection = value

    def CreateLambdifyHelper(self) -> LH.OdeLambdifyHelperWithBoundaryConditions:
        helper = LH.OdeLambdifyHelperWithBoundaryConditions(self.TimeSymbol, self.TimeInitialSymbol, self.TimeFinalSymbol, self.EquationsOfMotion, self.BoundaryConditions, [], self.SubstitutionDictionary)
        return helper




# class IndirectOptimizationProblem(Problem) :
#     def __init__(self, wrappedProblem : Problem):
#         self._baseProblem =wrappedProblem
#         self._transversalityConditionType = TransversalityConditionType.Differential
#         self._