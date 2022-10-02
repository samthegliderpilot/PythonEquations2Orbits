from abc import abstractmethod
from typing import Callable, Dict, List
from matplotlib.figure import Figure
from sympy import lambdify, Expr
from pyeq2orb.NumericalOptimizerProblem import NumericalOptimizerProblemBase
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.Utilities.inherit import inherit_docstrings


class NumericalProblemFromSymbolicProblem(NumericalOptimizerProblemBase) :
    def __init__(self, wrappedProblem : SymbolicProblem, functionMap : Dict) :
        super().__init__(wrappedProblem.TimeSymbol)
        if functionMap == None :
            functionMap = {}

        self._wrappedProblem = wrappedProblem
        self.State.extend(wrappedProblem.StateVariables)
        self.Control.extend(wrappedProblem.ControlVariables)

        entireState = [wrappedProblem.TimeSymbol, *wrappedProblem.StateVariables, *wrappedProblem.ControlVariables]
        
        if isinstance(wrappedProblem, ScaledSymbolicProblem) and wrappedProblem.ScaleTime :
            entireState.append(wrappedProblem.TimeFinalSymbolOriginal)
        
        finalState = SymbolicProblem.SafeSubs(entireState, {wrappedProblem.TimeSymbol: wrappedProblem.TimeFinalSymbol})
        if wrappedProblem.TerminalCost == None or wrappedProblem.TerminalCost == 0 :
            self._terminalCost = 0.0
        else:
            self._terminalCost = lambdify([finalState], SymbolicProblem.SafeSubs(wrappedProblem._terminalCost , wrappedProblem.SubstitutionDictionary).simplify(), functionMap)
        self._unIntegratedPathCost = lambda t, s: 0.0
        if wrappedProblem.UnIntegratedPathCost != None and wrappedProblem.UnIntegratedPathCost != 0.0 :
            self._unIntegratedPathCost = lambdify(entireState, wrappedProblem.UnIntegratedPathCost.subs(wrappedProblem.SubstitutionDictionary).simplify(), functionMap)
        self._equationOfMotionList = []
        for (sv, eom) in wrappedProblem.EquationsOfMotion.items() :
            numericaEom = SymbolicProblem.SafeSubs(eom, wrappedProblem.SubstitutionDictionary)
            if isinstance(numericaEom, Expr)  :
                numericaEom=numericaEom.simplify()
            eomCb = lambdify(entireState, numericaEom, functionMap)
            self._equationOfMotionList.append(eomCb) 

        for bc in wrappedProblem.BoundaryConditions :
            numericaBc=SymbolicProblem.SafeSubs(bc, wrappedProblem.SubstitutionDictionary)
            if isinstance(numericaBc, Expr)  :
                pass
                #numericaBc=numericaEom.simplify()
            bcCallback = lambdify([finalState], numericaBc, functionMap)
            self.BoundaryConditionCallbacks.append(bcCallback)        
        self._controlCallback = None

    #initial guess callback
    #initial conditions
    #final conditions

    # @property
    # def ContolValueAtTCallbackForInitialGuess(self) -> Callable:
    #     """Gets the callback to evaluate initial guesses for the problem.

    #     Returns:
    #         Callable: The callback
    #     """
    #     return self._controlCallback

    # @ContolValueAtTCallbackForInitialGuess.setter
    # def setContolValueAtTCallbackForInitialGuess(self, callback : Callable) :
    #     """Sets the control callback for initial guesses.

    #     Args:
    #         callback (Callable): The callback to set.
    #     """
    #     self._controlCallback = callback

    @inherit_docstrings
    def InitialGuessCallback(self, t : float) -> List[float] :
        pass

    @inherit_docstrings
    def EquationOfMotion(self, t : float, stateAndControlAtT : List[float]) -> List[float] :
        ans = []
        for i in range(0, len(self._equationOfMotionList)) :
            ans.append(self.SingleEquationOfMotion(t, stateAndControlAtT, i))
        return ans

    @inherit_docstrings
    def SingleEquationOfMotion(self, t : float, stateAndControlAtT : List[float], indexOfEom : int) -> float :
        return self._equationOfMotionList[indexOfEom](t, *stateAndControlAtT) # This is clearly not as efficient as it could be, but it is ok for a default implementation

    @inherit_docstrings
    def SingleEquationOfMotionWithTInState(self, state, indexOfEom) :
        return self._equationOfMotionList[indexOfEom](state[0], *state[1:])

    @inherit_docstrings
    def UnIntegratedPathCost(self, t, stateAndControl) :
        self.EquationOfMotion
        return self._unIntegratedPathCost(t, stateAndControl)

    @inherit_docstrings
    def TerminalCost(self, tf, finalStateAndControl) :
        return self._terminalCost([*finalStateAndControl, tf])

    @inherit_docstrings
    def AddResultsToFigure(self, figure : Figure, t : List[float], dictionaryOfValueArraysKeyedOffState : Dict[object, List[float]], label : str) -> None:
        self._wrappedProblem.AddStandardResultsToFigure(figure, t, dictionaryOfValueArraysKeyedOffState, label)

