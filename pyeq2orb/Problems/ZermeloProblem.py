import sympy as sy
from typing import List, Dict, Any
from pyeq2orb.Numerical import ScipyCallbackCreators
from pyeq2orb.ProblemBase import ProblemVariable, Problem
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper
from pyeq2orb.Utilities.inherit import inherit_docstrings
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs

@inherit_docstrings
class ZermelosProblem(Problem) :
    def __init__(self) :
        super().__init__()
        t = sy.Symbol('t')
        t0 = sy.Symbol('t_0')
        tf = sy.Symbol('t_f')
        v = sy.Symbol('V')
        theta = sy.Function(r'\theta')(t)
        x = ProblemVariable(sy.Function('x')(t), v*sy.cos(theta))
        y = ProblemVariable(sy.Function('y')(t), v*sy.sin(theta))

        xfDesiredValue = sy.Symbol('x_f')
        yfDesiredValue = sy.Symbol('y_f')
        xfBc = x.Element.subs(t, tf)-xfDesiredValue
        yfBc = y.Element.subs(t, tf)-yfDesiredValue

        self.AddStateVariable(x)
        self.AddStateVariable(y)

        self.BoundaryConditions.append(xfBc)
        self.BoundaryConditions.append(yfBc)

        self._terminalCost = tf
        self._timeFinalSymbol = tf
        self._timeInitialSymbol = t0
        self._timeSymbol = t

