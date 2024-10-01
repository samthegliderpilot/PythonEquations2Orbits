from typing import List, Callable, cast, Tuple, Dict
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb import SafeSubs
from dataclasses import dataclass
import sympy as sy

@dataclass
class scaledEquationOfMotionHolder:
    newStateVariables: List[sy.Symbol]
    scaledFirstOrderDynamics : List[sy.Expr]
    otherSymbols : List[sy.Symbol]
    timeSymbol : sy.Symbol
    timeWasUpdated : bool

    def createCorrectedSubsDict(self, subsDict : Dict[sy.Expr, SymbolOrNumber], originalSymbols: List[sy.Symbol], originalTime : sy.Symbol = None)->Dict[sy.Expr, SymbolOrNumber]:
        newDict = {}
        nonSvSubsDict = {}


        for i in range(0, len(self.newStateVariables)):
            nonSvSubsDict[originalSymbols[i]] = self.newStateVariables[i]
        if originalTime is not None:
            nonSvSubsDict[originalTime] = self.timeSymbol
        for k,v in subsDict.items():
            newKey = SafeSubs(k, nonSvSubsDict)
            newValue = SafeSubs(v, nonSvSubsDict)
            newDict[newKey] = newValue
        return newDict

    def scaledDynamicsAsMatrix(self) -> sy.Matrix:
        return sy.Matrix(self.scaledFirstOrderDynamics)

    def descaleState(self, t : SymbolOrNumber, state : List[SymbolOrNumber], scalingFactors : List[SymbolOrNumber], timeScalingFactor : SymbolOrNumber = 1) ->Tuple[SymbolOrNumber, List[SymbolOrNumber]]:
        descaled = []
        for i in range(0, len(state)):
            val = state[i]
            descaled.append(val*scalingFactors[i])

        tDescaled = timeScalingFactor*t
        return (tDescaled, descaled)

    def descaleStates(self, t : List[SymbolOrNumber], states : List[List[SymbolOrNumber]], scalingFactors : List[SymbolOrNumber], timeScalingFactor : SymbolOrNumber = 1)->Tuple[List[float], List[List[float]]]:
        tDescaled = []
        statesDescaled = []
        for i in range(0, len(t)):
            (tD, sD) = self.descaleState(t[i], states[i], scalingFactors, timeScalingFactor)
            tDescaled.append(tD)
            statesDescaled.append(sD)
        return (tDescaled, statesDescaled)

    def scaleState(self, t : SymbolOrNumber, state : List[SymbolOrNumber], scalingFactors : List[SymbolOrNumber], timeScalingFactor : SymbolOrNumber = 1) ->Tuple[SymbolOrNumber, List[SymbolOrNumber]]:
        scaled = []
        for i in range(0, len(state)):
            val = state[i]
            scaled.append(val/scalingFactors[i])

        tScaled = t/timeScalingFactor
        return (tScaled, scaled)

    @staticmethod
    def CreateVariablesWithBar(originalVariables : List[sy.Symbol], timeSymbol :sy.Symbol) ->List[sy.Symbol] :
        """A helper function to make a set of elements that

        Args:
            originalVariables (List[sy.Expr]): _description_
            timeSymbol (sy.Expr): _description_

        Returns:
            _type_: _description_
        """
        baredVariables :List[sy.Symbol] = [] 
        for var in originalVariables :
            name = var.__getattribute__('name')                        
            baredVariables.append(sy.Function(r'\bar{' + name+ '}')(timeSymbol), **var.assumptions0)
        return baredVariables


    @staticmethod
    def ScaleStateVariablesInFirstOrderOdes(oldStateVariables : List[sy.Symbol], firstOrderEquationsOfMotion : List[SymbolOrNumber], newStateVariables : List[sy.Symbol], scaleValuesToDivideByOriginal: List[SymbolOrNumber]) -> "scaledEquationOfMotionHolder":

        if len(oldStateVariables) != len(newStateVariables):
            raise Exception("When scaling state variables, all state variables must be present (even if the scaling is just 1)")
        newEoms= []
        subsDictForNewSvs = {}
        for i in range(0, len(newStateVariables)):
            if scaleValuesToDivideByOriginal[i] != 1 and scaleValuesToDivideByOriginal[i] != 1.0:
                subsDictForNewSvs[oldStateVariables[i]] = newStateVariables[i]*scaleValuesToDivideByOriginal[i]
            else:
                subsDictForNewSvs[oldStateVariables[i]] = newStateVariables[i]
        # the expression for scaling expressions is (scaling x to x1):
        # dx1/dt = dx1/dx * dx/dt
        # So for the dynamics, substitute {x:x1} and then multiple by dx1/dt, which is just the scaling value
        for i in range(0, len(newStateVariables)):            
            newEoms.append(SafeSubs(firstOrderEquationsOfMotion[i], subsDictForNewSvs)/scaleValuesToDivideByOriginal[i])

        scaledHelper = scaledEquationOfMotionHolder(newStateVariables, newEoms, [], oldStateVariables[0].args[0], False)        

        return scaledHelper

    @staticmethod
    def ScaleTimeInFirstOrderOdes(originalStateSymbols : List[sy.Symbol], originalTimeSymbol : sy.Symbol, firstOrderEquationsOfMotion : List[SymbolOrNumber], newTimeSymbol : sy.Symbol = None, timeScaleValueToDivideByOriginalTime : SymbolOrNumber = None, otherSymbols : List[sy.Symbol] =None)-> "scaledEquationOfMotionHolder":
        newEoms : List[sy.Expr] = []
        tSubsDict = {originalTimeSymbol: timeScaleValueToDivideByOriginalTime*newTimeSymbol}
        newStateVariables = []
        for sv in originalStateSymbols:
            newSv= SafeSubs(sv, {originalTimeSymbol: newTimeSymbol})
            newStateVariables.append(newSv)
            tSubsDict[sv] =newSv 
        scaledOtherSymbols = []
        otherSymbolsSubsDict = {}
        if otherSymbols is not None:
            for otherSymbol in otherSymbols:
                scaledOtherSymbols.append(SafeSubs(otherSymbol, {originalTimeSymbol: newTimeSymbol}))
                otherSymbolsSubsDict[otherSymbol] = scaledOtherSymbols[-1]
        for i in range(0, len(firstOrderEquationsOfMotion)):
            partlyScaledEom = SafeSubs(firstOrderEquationsOfMotion[i].expand().simplify(), otherSymbolsSubsDict)
            partlyScaledEom = SafeSubs(partlyScaledEom, tSubsDict)*timeScaleValueToDivideByOriginalTime
            newEoms.append(partlyScaledEom)


        scaledHelper = scaledEquationOfMotionHolder(newStateVariables, newEoms, scaledOtherSymbols, newTimeSymbol, True)    
        return scaledHelper


    @staticmethod
    def ScaleStateVariablesAndTimeInFirstOrderOdes(oldStateVariables : List[sy.Symbol], firstOrderEquationsOfMotion : List[SymbolOrNumber], newStateVariables : List[sy.Symbol], scaleValuesToDivideByOriginal: List[SymbolOrNumber], newTimeSymbol : sy.Symbol, timeScaleValueToDivideByOriginalTime : SymbolOrNumber, otherSymbols : List[sy.Symbol] =None) -> "scaledEquationOfMotionHolder":
        originalTimeSymbol = oldStateVariables[0].args[0]
        justScaledState = scaledEquationOfMotionHolder.ScaleStateVariablesInFirstOrderOdes(oldStateVariables, firstOrderEquationsOfMotion, newStateVariables, scaleValuesToDivideByOriginal)
        andScaledByTime = scaledEquationOfMotionHolder.ScaleTimeInFirstOrderOdes(justScaledState.newStateVariables, originalTimeSymbol, justScaledState.scaledFirstOrderDynamics, newTimeSymbol, timeScaleValueToDivideByOriginalTime, otherSymbols)
        return andScaledByTime
