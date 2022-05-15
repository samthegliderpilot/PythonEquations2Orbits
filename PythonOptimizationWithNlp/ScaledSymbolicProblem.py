from textwrap import wrap
import sympy as sy
from typing import List, Dict
from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem
import numpy as np
from PythonOptimizationWithNlp.Symbolics.Vectors import Vector
from PythonOptimizationWithNlp.Utilities.inherit import inherit_docstrings

@inherit_docstrings
class ScaledSymbolicProblem(SymbolicProblem) :
    def __init__(self, wrappedProblem : SymbolicProblem, newStateVariableSymbols, valuesToDivideStateVariablesWith : Dict, scaleTime : bool) :        
        self._wrappedProblem = wrappedProblem

        self._timeFinalSymbol = wrappedProblem.TimeFinalSymbol
        self._timeInitialSymbol = wrappedProblem.TimeInitialSymbol
        self._timeSymbol = wrappedProblem.TimeSymbol
        
        self._scaleTime=scaleTime
        self._substitutionDictionary = wrappedProblem._substitutionDictionary        
        self._stateVariables = newStateVariableSymbols
        
        self._scalingDict = valuesToDivideStateVariablesWith

        newSvsInTermsOfOldSvs = []
        counter=0
        for sv in wrappedProblem.StateVariables :
            newSvsInTermsOfOldSvs.append(sv/valuesToDivideStateVariablesWith[sv])
            counter=counter+1

        fullSubsDict = {} # don't add in sv(t_0), that will confuse common scaling paradigms
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
        
        # should the path cost get scaled?  I think so, but if you know better, or if it doesn't work...
        self._unIntegratedPathCost = SymbolicProblem.SafeSubs(wrappedProblem.UnIntegratedPathCost, fullSubsDict)
        # do not scale the cost function! conditions made from the cost get scaled later
        self._terminalCost = wrappedProblem.TerminalCost
        if scaleTime :
            self._tfOrg=wrappedProblem.TimeFinalSymbol
            tf = self._tfOrg
            tau = sy.Symbol(r'\tau')
            tauF = sy.Symbol(r'\tau_f')
            tau0 = sy.Symbol(r'\tau_0')
            timeSubs = {wrappedProblem.TimeInitialSymbol: tau0, wrappedProblem.TimeSymbol: tau, wrappedProblem.TimeFinalSymbol:tauF}
            self._controlVariables = ScaledSymbolicProblem.SafeSubs(self._controlVariables, timeSubs)
            orgSv = self._stateVariables
            self._stateVariables = ScaledSymbolicProblem.SafeSubs(self._stateVariables, timeSubs)
            self._unIntegratedPathCost = ScaledSymbolicProblem.SafeSubs(self._unIntegratedPathCost, timeSubs)
            self._terminalCost = ScaledSymbolicProblem.SafeSubs(self._terminalCost, timeSubs)
            self._boundaryConditions = ScaledSymbolicProblem.SafeSubs(self._boundaryConditions, timeSubs)

            # the correct thing to do is to make your state and control variables functions of time in Sympy
            # myCv = sy.Function('Control')(t)
            # BUT, we need to substitute t with tau*tf
            # BUT, we don't want myCv to be a function of tau*tf, just of tau
            # I don't know how to make the sympy subs function not go deep like that, so, we substitute these back...
            toSimple = {}
            fromSimple = {}
            correctVariablesSubsDict = {}
            for sv in self._stateVariables :
                adjustedSv = sv.subs(tau, self._wrappedProblem.TimeSymbol)
                toSimple[adjustedSv] = sy.Symbol(sv.name)
                fromSimple[toSimple[adjustedSv]] = sv
                #correctVariablesSubsDict[sv.subs(tau, tau*tf)] = sv
                
            for cv in self._controlVariables :
                adjustedCv = cv.subs(tau, self._wrappedProblem.TimeSymbol)
                toSimple[adjustedCv] = sy.Symbol(cv.name)
                fromSimple[toSimple[adjustedCv]] = cv

                #correctVariablesSubsDict[cv.subs(tau, tau*tf)] = cv
            realEom = {}
            i = 0
            timeSubs = { wrappedProblem.TimeSymbol: tau*tf}
            for sv in orgSv :
                realEom[self._stateVariables[i]] = ScaledSymbolicProblem.SafeSubs(self._equationsOfMotion[sv], toSimple)
                realEom[self._stateVariables[i]] = ScaledSymbolicProblem.SafeSubs(realEom[self._stateVariables[i]], timeSubs)*wrappedProblem.TimeFinalSymbol
                realEom[self._stateVariables[i]] = ScaledSymbolicProblem.SafeSubs(realEom[self._stateVariables[i]], fromSimple)
                i=i+1
            self._equationsOfMotion = realEom
            
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

    def ScaleExpressions(self, finalConditions):
        scaledFinalConditions = []
        simpleSubsDict={} # for terminal cost
        counter=0
        for sv in self.StateVariables :
            oldSv = self.WrappedProblem.StateVariables[counter]
            simpleSubsDict[oldSv] = sv*self.ScalingVector[oldSv]
            simpleSubsDict[oldSv.subs(self.WrappedProblem.TimeSymbol, self.WrappedProblem.TimeFinalSymbol)] = sv.subs(self.TimeSymbol, self.TimeFinalSymbol)*self.ScalingVector[oldSv]
            counter=counter+1

        for cond in finalConditions :
            scaledFinalConditions.append(cond.subs(simpleSubsDict))
        return scaledFinalConditions 

    def TransversalityConditionsByAugmentation(self, lambdas, nus) :
        finalConditions = self.WrappedProblem.TransversalityConditionsByAugmentation(lambdas, nus)
        return self.ScaleExpressions(finalConditions)
    
    def TransversalityConditionInTheDifferentialForm(self, hamiltonian, lambdasFinal, dtf) :
        finalConditions = self.WrappedProblem.TransversalityConditionInTheDifferentialForm(hamiltonian, lambdasFinal, dtf)
        return self.ScaleExpressions(finalConditions)
       