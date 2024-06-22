import sympy as sy
from typing import List
from pyeq2orb.Utilities.Typing import SymbolOrNumber
class StateElement:
    def __init__(self):
        self.Symbol = None
        self.EquationOfMotion = None 

class StateElementManager:
    def __init__(self):
        self.StateElements = []

    def LHS(self) ->List[sy.Symbol]:
        lhs = []
        for se in self.StateElements:
            lhs.append(se.Symbol)
        return lhs

    def RHS(self) ->List[SymbolOrNumber]:
        rhs = []
        for se in self.StateElements:
            rhs.append(se.EquationOfMotion)
        return rhs        
