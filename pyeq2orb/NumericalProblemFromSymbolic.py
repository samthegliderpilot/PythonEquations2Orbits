from abc import abstractmethod
from typing import Callable, Dict, List
from matplotlib.figure import Figure # type: ignore
from sympy import lambdify, Expr
from pyeq2orb.NumericalOptimizerProblem import NumericalOptimizerProblemBase
from pyeq2orb.ScaledSymbolicProblem import ScaledSymbolicProblem
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.Utilities.inherit import inherit_docstrings
import sympy as sy
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs

class NumericalProblemFromSymbolicProblem(NumericalOptimizerProblemBase) :
    def __init__(self, wrappedProblem : SymbolicProblem, functionMap : Dict) :
        super().__init__(wrappedProblem.TimeSymbol)
        if functionMap == None :
            functionMap = {}

        self._wrappedProblem = wrappedProblem
        self.State.extend(wrappedProblem.StateVariables)
        self.Control.extend(wrappedProblem.ControlVariables)

        entireState = [wrappedProblem.TimeSymbol, *wrappedProblem.StateVariables, *wrappedProblem.ControlVariables] #type: List[sy.Expr]
        
        if isinstance(wrappedProblem, ScaledSymbolicProblem) and wrappedProblem.ScaleTime :
            entireState.append(wrappedProblem.TimeFinalSymbol)
        
        finalState = SafeSubs(entireState, {wrappedProblem.TimeSymbol: wrappedProblem.TimeFinalSymbol}) 
        if wrappedProblem.TerminalCost == None or wrappedProblem.TerminalCost == 0 :
            self._terminalCost = 0.0
        else:
            self._terminalCost = lambdify([finalState], SafeSubs(wrappedProblem._terminalCost , wrappedProblem.SubstitutionDictionary).simplify(), functionMap)
        self._unIntegratedPathCost = lambda t, s: 0.0
        if wrappedProblem.UnIntegratedPathCost != None and wrappedProblem.UnIntegratedPathCost != 0.0 :
            self._unIntegratedPathCost = lambdify(entireState, wrappedProblem.UnIntegratedPathCost.subs(wrappedProblem.SubstitutionDictionary).simplify(), functionMap)
        self._equationOfMotionList = []
        for i in range(0, len(wrappedProblem.StateVariableDynamics)) :
            numericaEom = SafeSubs(wrappedProblem.StateVariableDynamics[i], wrappedProblem.SubstitutionDictionary)
            eomCb = lambdify(entireState, numericaEom, functionMap)
            self._equationOfMotionList.append(eomCb) 

        for bc in wrappedProblem.BoundaryConditions :
            numericaBc=SafeSubs(bc, wrappedProblem.SubstitutionDictionary)
            if isinstance(numericaBc, Expr)  :
                numericaBc=numericaBc.simplify().expand().simplify()
            bcCallback = lambdify([finalState], numericaBc, functionMap)
            self.BoundaryConditionCallbacks.append(bcCallback)        
        self._controlCallback = None


    @inherit_docstrings
    #@abstractmethod
    def InitialGuessCallback(self, t : float) -> List[float] :
        return []

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
