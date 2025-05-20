import sympy as sy
from typing import List, Union
from pyeq2orb import SafeSubs
symbolOrListOfSymbols = Union[sy.Symbol, List[sy.Symbol]]

class StateManager:
    def __init__(self, time : sy.Symbol, stateVariables : List[sy.Symbol], lambdaVariables : List[sy.Symbol], otherArgs : List[sy.Symbol]):
        self.Time = time
        self.StateVariables =stateVariables
        self.LambdaVariables =lambdaVariables
        self.OtherArgs = otherArgs

    def createDefaultSymbolState(self, includeOtherArgs : bool = False) -> List[symbolOrListOfSymbols]:
        state = []
        state.append(self.Time)
        state.append(self.StateVariables)
        state.append(self.LambdaVariables)
        if includeOtherArgs:
            state.append(self.OtherArgs)

    def createDefaultInitialState(self, t0:sy.Symbol, includeOtherArgs : bool = False) -> List[symbolOrListOfSymbols]:
        state = self.createDefaultSymbolState(includeOtherArgs)
        return SafeSubs(state, {self.Time: t0})

    def createDefaultFinalState(self, tf:sy.Symbol, includeOtherArgs : bool = False) -> List[symbolOrListOfSymbols]:
        state = self.createDefaultSymbolState(includeOtherArgs)
        return SafeSubs(state, {self.Time: tf})
