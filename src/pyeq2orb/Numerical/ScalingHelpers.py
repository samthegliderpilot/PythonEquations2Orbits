from typing import List, Callable, cast
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb import SafeSubs
from dataclasses import dataclass
import sympy as sy

@dataclass
class scaledEquationsOfMotionResult:
    newStateVariables: List[sy.Symbol]
    scaledFirstOrderDynamics : List[sy.Expr]

    def scaledDynamicsAsMatrix(self) -> sy.Matrix:
        return sy.Matrix(self.scaledFirstOrderDynamics)

    def descaleState(self, state : List[SymbolOrNumber], scalingFactors : List[SymbolOrNumber], timeScalingFactor : SymbolOrNumber = None):
        

    @staticmethod
    def ScaleStateVariablesInFirstOrderOdes(oldStateVariables : List[sy.Symbol], firstOrderEquationsOfMotion : List[SymbolOrNumber], newStateVariables : List[sy.Symbol], scaleValuesToDivideByOriginal: List[SymbolOrNumber]) -> "scaledEquationsOfMotionResult":

        if len(oldStateVariables) != len(newStateVariables):
            raise Exception("When scaling state variables, all state variables must be present (even if the scaling is just 1)")
        newEoms= []
        subsDictForNewSvs = {}
        for i in range(0, len(newStateVariables)):
            subsDictForNewSvs[oldStateVariables[i]] = newStateVariables[i]*scaleValuesToDivideByOriginal[i]
        # the expression for scaling expressions is (scaling x to x1):
        # dx1/dt = dx1/dx * dx/dt
        # So for the dynamics, substitute {x:x1} and then multiple by dx1/dt, which is just the scaling value
        for i in range(0, len(newStateVariables)):            
            newEoms.append(SafeSubs(firstOrderEquationsOfMotion[i], subsDictForNewSvs)/scaleValuesToDivideByOriginal[i])

        scaledHelper = scaledEquationsOfMotionResult(newStateVariables, newEoms)        

        return scaledHelper

    @staticmethod
    def ScaleTimeInFirstOrderOdes(originalStateSymbols : List[sy.Symbol], originalTimeSymbol : sy.Symbol, firstOrderEquationsOfMotion : List[SymbolOrNumber], newTimeSymbol : sy.Symbol = None, timeScaleValueToDivideByOriginalTime : SymbolOrNumber = None )-> "scaledEquationsOfMotionResult":
        newEoms : List[sy.Expr] = []
        tSubsDict = {originalTimeSymbol: timeScaleValueToDivideByOriginalTime*newTimeSymbol}
        newStateVariables = []
        for sv in originalStateSymbols:
            newSv= SafeSubs(sv, {originalTimeSymbol: newTimeSymbol})
            newStateVariables.append(newSv)
            tSubsDict[sv] =newSv 
        for i in range(0, len(firstOrderEquationsOfMotion)):
            newEoms.append(SafeSubs(firstOrderEquationsOfMotion[i], tSubsDict)*timeScaleValueToDivideByOriginalTime)

        scaledHelper = scaledEquationsOfMotionResult(newStateVariables, newEoms)    
        return scaledHelper


    @staticmethod
    def ScaleStateVariablesAndTimeInFirstOrderOdes(oldStateVariables : List[sy.Symbol], firstOrderEquationsOfMotion : List[SymbolOrNumber], newStateVariables : List[sy.Symbol], scaleValuesToDivideByOriginal: List[SymbolOrNumber], newTimeSymbol : sy.Symbol, timeScaleValueToDivideByOriginalTime : SymbolOrNumber) -> "scaledEquationsOfMotionResult":
        originalTimeSymbol = oldStateVariables[0].args[0]
        justScaledState = scaledEquationsOfMotionResult.ScaleStateVariablesInFirstOrderOdes(oldStateVariables, firstOrderEquationsOfMotion, newStateVariables, scaleValuesToDivideByOriginal)
        andScaledByTime = scaledEquationsOfMotionResult.ScaleTimeInFirstOrderOdes(justScaledState.newStateVariables, originalTimeSymbol, justScaledState.scaledFirstOrderDynamics, newTimeSymbol, timeScaleValueToDivideByOriginalTime)
        return andScaledByTime
