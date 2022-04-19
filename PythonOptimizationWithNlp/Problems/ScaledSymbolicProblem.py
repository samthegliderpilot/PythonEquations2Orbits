from textwrap import wrap
import sympy as sy
from typing import TypedDict, List, Dict
from collections import OrderedDict
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
import math
import matplotlib.pyplot as plt
import numpy as np
from PythonOptimizationWithNlp.Utilities.inherit import inherit_docstrings

@inherit_docstrings
class ScaledSymbolicProblem(SymbolicProblem) :
    def __init__(self, wrappedProblem : SymbolicProblem, newStateVariableSymbols, valuesToDivideStateVariablesWith : Dict, scaleTime : bool) :
        self._stateVariables = newStateVariableSymbols
        self._wrappedProblem = wrappedProblem
        newSvsInTermsOfOldSvs = []
        counter=0
        for sv in wrappedProblem.StateVariables :
            newSvsInTermsOfOldSvs.append(sv/valuesToDivideStateVariablesWith[sv])
            counter=counter+1

        self._scalingDict = valuesToDivideStateVariablesWith

        self._tf = wrappedProblem.TimeFinalSymbol
        self._t0 = wrappedProblem.Time0Symbol
        self.Ts = wrappedProblem.TimeSymbol
        self._constantSymbols = wrappedProblem.ConstantSymbols

        fullSubsDict = {}
        counter = 0
        for sv in wrappedProblem.StateVariables :
            fullSubsDict[sv] = newStateVariableSymbols[counter]*valuesToDivideStateVariablesWith[sv]            
            counter = counter+1

        self._controlVariables = wrappedProblem.ControlVariables
        scaledEoms = {}
        counter = 0
        for sv in wrappedProblem.StateVariables :
            if hasattr(wrappedProblem.EquationsOfMotion[sv], "subs") :
                scaledEoms[self._stateVariables[counter]] = wrappedProblem.EquationsOfMotion[sv].subs(fullSubsDict)*newSvsInTermsOfOldSvs[counter].diff(sv)
            else :
                scaledEoms[self._stateVariables[counter]] = wrappedProblem.EquationsOfMotion[sv]*newSvsInTermsOfOldSvs[counter].diff(sv)
            counter=counter+1
        self._equationsOfMotion = scaledEoms
        


        counter = 0
        bcSubsDict = {}
        for sv in wrappedProblem.StateVariables :
            svAtTf=sv.subs(wrappedProblem.TimeSymbol, wrappedProblem.TimeFinalSymbol)
            newSvAtTf = newStateVariableSymbols[counter].subs(self.TimeSymbol, self.TimeFinalSymbol)
            bcSubsDict[svAtTf] = newSvAtTf*valuesToDivideStateVariablesWith[sv]
            counter = counter+1
        bcs = []
        
        for bc in wrappedProblem.FinalBoundaryConditions :
            bcs.append(bc.subs(bcSubsDict))
            # counter = 0
            # newBc = bc
            # for sv in wrappedProblem.StateVariables :
            #     newBc = newBc.subs(sv.subs(self.TimeSymbol, self.TimeFinalSymbol), newStateVariableSymbols[counter].subs(self.TimeSymbol, self.TimeFinalSymbol))
            #     counter=counter+1
            # bcs.append(newBc)
        self._finalBoundaryConditions = bcs

        pcs = []
        for pc in wrappedProblem.PathConstraints :
            pcs.append(pc.subs(fullSubsDict)*valuesToDivideStateVariablesWith[sv])
        self._pathConstraints = pcs

        #if hasattr(wrappedProblem.TerminalCost, "subs") :
        #    self._terminalCost = wrappedProblem.TerminalCost.subs(fullSubsDict)
        #else :
        self._terminalCost = self.StateVariables[0]# wrappedProblem.TerminalCost

        if hasattr(wrappedProblem.UnIntegratedPathCost, "subs") :
            self._unIntegratedPathCost = wrappedProblem.UnIntegratedPathCost.subs(fullSubsDict)
        else :
            self._unIntegratedPathCost = wrappedProblem.UnIntegratedPathCost

    @property
    def ScalingVector(self) -> Dict :
        return self._scalingDict

    @property
    def WrappedProblem(self) -> SymbolicProblem :
        return self._wrappedProblem


    def DescaleResults(self, resultsDictionary : Dict[sy.Symbol, List[float]], subsDict) -> Dict[sy.Symbol, List[float]] :
        returnDict = {}
        counter = 0
        for key, value in resultsDictionary.items() :
            sv = key
            if sv in self.StateVariables:
                originalSv = self.WrappedProblem.StateVariables[self.StateVariables.index(sv)]
                convertedArray = np.array(value, copy=True)* SymbolicProblem.SafeSubs(self.ScalingVector[originalSv], subsDict)
                returnDict[originalSv] = convertedArray
                counter = counter+1
            else :
                returnDict[key]=value
        return returnDict

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
        return self._terminalCost

    @property
    def UnIntegratedPathCost(self) -> sy.Expr :
        return self._unIntegratedPathCost