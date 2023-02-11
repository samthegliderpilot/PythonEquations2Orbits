from typing import List
import sympy as sy
from sympy.solvers.ode.systems import dsolve_system
from pyeq2orb import SafeSubs
from..DemosAndPrototypes import scipyPaperPrinter as jh

class OdeHelper :
    lambidfyStateFlattenOption = "flatten"
    lambidfyStateGroupedAllOption = "group"
    lambidfyStateGroupedAllButParametersOption = "groupFlattenParamerts"

    lambidfyStateOrderOptionTimeFirst = "Time,StateVariables,MissingInitialValues,Parameters"
    lambidfyStateOrderOptionTimeMiddle = "StateVariables,Time,MissingInitialValues,Parameters"
    def __init__(self, t) :
        self.equationsOfMotion = []
        self.initialSymbols = []
        self.stateSymbolsOfT = []
        self.t = t
        self.constants = {}
        self.lamdifyParameterSymbols = []

    def setStateElement(self, sympyFunctionSymbol, symbolicEom, initialSymbol) :
        self.equationsOfMotion.append(symbolicEom)
        self.stateSymbolsOfT.append(sympyFunctionSymbol)
        self.initialSymbols.append(initialSymbol)

    def makeStateForLambdififedFunction(self, groupOrFlatten=lambidfyStateGroupedAllButParametersOption, orderOption=lambidfyStateOrderOptionTimeFirst):
        arrayForLmd = []
        if orderOption == OdeHelper.lambidfyStateOrderOptionTimeFirst :
            arrayForLmd.append(self.t)
        stateArray = []    
        for svf in self.stateSymbolsOfT :
            stateArray.append(svf)
        if groupOrFlatten != OdeHelper.lambidfyStateFlattenOption :
            arrayForLmd.append(stateArray)    
        else :
            arrayForLmd.extend(stateArray)
        if orderOption == OdeHelper.lambidfyStateOrderOptionTimeMiddle :
            arrayForLmd.append(self.t)

        if len(self.lamdifyParameterSymbols) != 0 :
            if groupOrFlatten == OdeHelper.lambidfyStateGroupedAllButParametersOption or groupOrFlatten == OdeHelper.lambidfyStateFlattenOption:
                arrayForLmd.extend(self.lamdifyParameterSymbols)
            elif groupOrFlatten == OdeHelper.lambidfyStateGroupedAllOption :
                arrayForLmd.append(self.lamdifyParameterSymbols)
        return arrayForLmd

    def _createParameterOptionalWrapperOfLambdifyCallback(self, baseLambidfyCallback) :
        def callbackWraper(a, b, *args) :
            if len(self.lamdifyParameterSymbols) == 0 :
                return baseLambidfyCallback(a, b)
            else :
                return baseLambidfyCallback(a, b, *args)
        return callbackWraper

    def createLambdifiedCallback(self, groupOrFlatten=lambidfyStateGroupedAllButParametersOption, orderOption=lambidfyStateOrderOptionTimeFirst) :
        arrayForLmd=self.makeStateForLambdififedFunction(groupOrFlatten, orderOption)
        subsedEom = SafeSubs(self.equationsOfMotion, self.constants)
        baseLambidfyCallback = sy.lambdify(arrayForLmd, subsedEom, 'numpy')
        return self._createParameterOptionalWrapperOfLambdifyCallback(baseLambidfyCallback)

    def tryDeSolve(self):
        icSymbols = {}
        for i in range(0, len(self.initialSymbols)) :
            icSymbols[self.initialSymbols[i]] = self.initialSymbols[i]
        
        firstOrderSystem = []
        subsedEom = SafeSubs(self.equationsOfMotion, self.constants)
        for i in range(0, len(self.equationsOfMotion)) :
            firstOrderSystem.append(sy.Eq(sy.Derivative(self.stateSymbolsOfT[i]), subsedEom[i]))  
        deSolveAns = dsolve_system(firstOrderSystem, ics=icSymbols)  
        return deSolveAns

    def deSolveResultsToCallback(self, deSolveResults, initialStateValues : List[float], groupOrFlatten=lambidfyStateGroupedAllButParametersOption, orderOption=lambidfyStateOrderOptionTimeFirst) :
        if deSolveResults == None :
            deSolveResults = self.tryDeSolve()
        rhss = []
        icsSubs = {}
        for i in range(0, len(self.initialSymbols)):
            icsSubs[self.initialSymbols[i]]=initialStateValues[i]

        for i in range(0, len(deSolveResults[0])):
            rhss.append(deSolveResults[0][i].rhs.subs(icsSubs).doit().simplify())
         
        arrayForLmd=self.makeStateForLambdififedFunction(groupOrFlatten, orderOption)
        lmdified = sy.lambdify(arrayForLmd, rhss)
        return self._createParameterOptionalWrapperOfLambdifyCallback(lmdified)
