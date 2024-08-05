from typing import List
import sympy as sy
from sympy.solvers.ode.systems import dsolve_system
from pyeq2orb import SafeSubs
from..DemosAndPrototypes import scipyPaperPrinter as jh

class OdeDeSolveHelper :
    lambdifyStateFlattenOption = "flatten"
    lambdifyStateGroupedAllOption = "group"
    lambdifyStateGroupedAllButParametersOption = "groupFlattenParamerts"

    lambdifyStateOrderOptionTimeFirst = "Time,StateSymbols,MissingInitialValues,Parameters"
    lambdifyStateOrderOptionTimeMiddle = "StateSymbols,Time,MissingInitialValues,Parameters"
    def __init__(self, t) :
        self.equationsOfMotion = []
        self.initialSymbols = []
        self.stateSymbolsOfT = []
        self.t = t
        self.constants = {}
        self.lambdifyParameterSymbols = []

    def setStateElement(self, sympyFunctionSymbol, symbolicEom, initialSymbol) :
        self.equationsOfMotion.append(symbolicEom)
        self.stateSymbolsOfT.append(sympyFunctionSymbol)
        self.initialSymbols.append(initialSymbol)

    def _makeStateForLambdifiedFunction(self, groupOrFlatten=lambdifyStateGroupedAllButParametersOption, orderOption=lambdifyStateOrderOptionTimeFirst):
        arrayForLmd = []
        if orderOption == OdeDeSolveHelper.lambdifyStateOrderOptionTimeFirst :
            arrayForLmd.append(self.t)
        stateArray = []    
        for svf in self.stateSymbolsOfT :
            stateArray.append(svf)
        if groupOrFlatten != OdeDeSolveHelper.lambdifyStateFlattenOption :
            arrayForLmd.append(stateArray)    
        else :
            arrayForLmd.extend(stateArray)
        if orderOption == OdeDeSolveHelper.lambdifyStateOrderOptionTimeMiddle :
            arrayForLmd.append(self.t)

        if len(self.lambdifyParameterSymbols) != 0 :
            if groupOrFlatten == OdeDeSolveHelper.lambdifyStateGroupedAllButParametersOption or groupOrFlatten == OdeHelper.lambdifyStateFlattenOption:
                arrayForLmd.extend(self.lambdifyParameterSymbols)
            elif groupOrFlatten == OdeDeSolveHelper.lambdifyStateGroupedAllOption :
                arrayForLmd.append(self.lambdifyParameterSymbols)
        return arrayForLmd

    def _createParameterOptionalWrapperOfLambdifyCallback(self, baseLambdifyCallback) :
        def callbackWrapper(a, b, *args) :
            if len(self.lambdifyParameterSymbols) == 0 :
                return baseLambdifyCallback(a, b)
            else :
                return baseLambdifyCallback(a, b, *args)
        return callbackWrapper

    def createLambdifiedCallback(self, groupOrFlatten=lambdifyStateGroupedAllButParametersOption, orderOption=lambdifyStateOrderOptionTimeFirst) :
        arrayForLmd=self._makeStateForLambdifiedFunction(groupOrFlatten, orderOption)
        subbedEom = SafeSubs(self.equationsOfMotion, self.constants)
        baseLambdifyCallback = sy.lambdify(arrayForLmd, subbedEom, 'numpy')
        return self._createParameterOptionalWrapperOfLambdifyCallback(baseLambdifyCallback)

    def attemptDeSolve(self):
        icSymbols = {}
        for i in range(0, len(self.initialSymbols)) :
            icSymbols[self.initialSymbols[i]] = self.initialSymbols[i]
        
        firstOrderSystem = []
        subbedEom = SafeSubs(self.equationsOfMotion, self.constants)
        for i in range(0, len(self.equationsOfMotion)) :
            firstOrderSystem.append(sy.Eq(sy.Derivative(self.stateSymbolsOfT[i]), subbedEom[i]))  
        deSolveAns = dsolve_system(firstOrderSystem, ics=icSymbols)  
        return deSolveAns

    def deSolveResultsToCallback(self, deSolveResults, initialStateValues : List[float], groupOrFlatten=lambdifyStateGroupedAllButParametersOption, orderOption=lambdifyStateOrderOptionTimeFirst) :
        if deSolveResults == None :
            deSolveResults = self.attemptDeSolve()
        rhss = []
        icsSubs = {}
        for i in range(0, len(self.initialSymbols)):
            icsSubs[self.initialSymbols[i]]=initialStateValues[i]

        for i in range(0, len(deSolveResults[0])):
            rhss.append(deSolveResults[0][i].rhs.subs(icsSubs).doit().simplify())
         
        arrayForLmd=self._makeStateForLambdifiedFunction(groupOrFlatten, orderOption)
        lambdified = sy.lambdify(arrayForLmd, rhss)
        return self._createParameterOptionalWrapperOfLambdifyCallback(lambdified)
