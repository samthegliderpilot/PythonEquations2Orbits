from typing import Union
import sympy as sy
from numbers import Real

SymbolOrNumber = Union[sy.Expr, sy.Symbol, Real, float]
ExpressionOrNumber = Union[sy.Expr, Real, float]