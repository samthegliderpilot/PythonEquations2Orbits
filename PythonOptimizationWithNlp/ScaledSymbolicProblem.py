import sympy as sy
from typing import List, Dict
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
import numpy as np
from PythonOptimizationWithNlp.Utilities.inherit import inherit_docstrings

@inherit_docstrings
class ScaledSymbolicProblem(SymbolicProblem) :
    def __init__(self, wrappedProblem : SymbolicProblem, newStateVariableSymbols, valuesToDivideStateVariablesWith : Dict, scaleTime : bool) :
        
        self._wrappedProblem = wrappedProblem
        self._scaleTime=scaleTime
        self._substitutionDictionary = wrappedProblem._substitutionDictionary
        

        self._stateVariables = newStateVariableSymbols
        
        newSvsInTermsOfOldSvs = []
        counter=0
        for sv in wrappedProblem.StateVariables :
            newSvsInTermsOfOldSvs.append(sv/valuesToDivideStateVariablesWith[sv])
            counter=counter+1

        self._scalingDict = valuesToDivideStateVariablesWith

        self._timeFinalSymbol = wrappedProblem.TimeFinalSymbol
        self._timeInitialSymbol = wrappedProblem.TimeInitialSymbol
        self._timeSymbol = wrappedProblem.TimeSymbol

        fullSubsDict = {}
        counter = 0
        for sv in wrappedProblem.StateVariables :
            fullSubsDict[sv] = newStateVariableSymbols[counter]*valuesToDivideStateVariablesWith[sv]            
            fullSubsDict[sv.subs(wrappedProblem.TimeSymbol, wrappedProblem.TimeFinalSymbol)] = newStateVariableSymbols[counter].subs(wrappedProblem.TimeSymbol, wrappedProblem.TimeFinalSymbol)*valuesToDivideStateVariablesWith[sv]            
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
        
        for bc in wrappedProblem.BoundaryConditions :
            bcs.append(bc.subs(bcSubsDict))
        self._boundaryConditions = bcs
        
        self._unIntegratedPathCost = wrappedProblem.UnIntegratedPathCost
        self._terminalCost = wrappedProblem.TerminalCost
        if scaleTime :
            tau = sy.Symbol(r'\tau')
            tauF = sy.Symbol(r'\tau_f')
            tau0 = sy.Symbol(r'\tau_0')
            timeSubs = {self.TimeInitialSymbol: tau0, wrappedProblem.TimeSymbol: tau, wrappedProblem.TimeFinalSymbol:tauF}
            self._controlVariables = ScaledSymbolicProblem.SafeSubs(self._controlVariables, timeSubs)
            orgSv = self._stateVariables
            self._stateVariables = ScaledSymbolicProblem.SafeSubs(self._stateVariables, timeSubs)
            self._unIntegratedPathCost = ScaledSymbolicProblem.SafeSubs(self._unIntegratedPathCost, timeSubs)
            self._terminalCost = ScaledSymbolicProblem.SafeSubs(self._terminalCost, timeSubs)
            self._boundaryConditions = ScaledSymbolicProblem.SafeSubs(self._boundaryConditions, timeSubs)
            realEom = {}
            i = 0
            for sv in orgSv :
                realEom[self._stateVariables[i]] = ScaledSymbolicProblem.SafeSubs(self._equationsOfMotion[sv], timeSubs)*self.TimeFinalSymbol
                i=i+1
            self._equationsOfMotion = realEom
            self._tfOrg=self.TimeFinalSymbol
            self._timeSymbol = tau 
            self._timeInitialSymbol = tau0
            self._timeFinalSymbol = tauF

    @property
    def ScaleTime(self) -> bool :
        return self._scaleTime


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
    def TimeFinalSymbolOriginal(self)-> sy.Symbol:
        return self._tfOrg

    @staticmethod
    def CreateBarVariables(orgVariables : List[sy.Expr], timeSymbol :sy.Expr) :
        baredVariables = []
        for var in orgVariables :
            baredVariables.append(sy.Function(r'\bar{' + var.name+ '}')(timeSymbol))
        return baredVariables