from __future__ import annotations
import sympy as sy
import numpy as np
from typing import List, Dict, cast, Optional, Collection
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from abc import ABC
from collections import OrderedDict
from enum import Enum
from pyeq2orb import SafeSubs
from pyeq2orb.Symbolics.Vectors import Vector # type: ignore

class IntegrationDirection(Enum) :
    Forward = 1
    Backward = -1

class ProblemVariable():
    def __init__(self, element : sy.Symbol, firstOrderDynamics : Optional[sy.Expr]):
        self._element = element
        self._firstOrderDynamics = firstOrderDynamics

    @property
    def Element(self) -> sy.Symbol:
        return self._element
    
    @property
    def FirstOrderDynamics(self) -> Optional[sy.Expr]:
        return self._firstOrderDynamics

    @FirstOrderDynamics.setter
    def FirstOrderDynamics(self, value:sy.Expr) :
        self._firstOrderDynamics = value

    # def subs(self, subsDict : Dict[sy.Expr, SymbolOrNumber]):
    #     return ProblemVariable(self.Element, SafeSubs(self.FirstOrderDynamics, subsDict))


class Problem(ABC) :
    def __init__(self) :
        """Initialize a new instance. 
        """
        self._stateVariables = [] 
        self._costateElements = [] 
        self._controlVariables = [] 

        self._terminalCost = 0
        self._unIntegratedPathCost = 0        
        self._boundaryConditions = []
        self._timeSymbol = None
        self._timeInitialSymbol = None
        self._timeFinalSymbol= None

        self._integrationDirection = IntegrationDirection.Forward

        self._unscaledTimeSymbol = None
        self._unscaledTimeInitialSymbol = None
        self._unscaledTimeFinalSymbol = None
        self._timeScaleFactor = None
        self._scalingDict = {}
        self._descaleDict = {}

        self._timeValueToSolveFor = None

        self._substitutionDictionary = OrderedDict()

        self._otherArgs = []


    @staticmethod
    def CreateBarVariables(orgVariables : List[sy.Symbol], timeSymbol :sy.Symbol) ->List[sy.Symbol] :
        """A helper function to make a 

        Args:
            orgVariables (List[sy.Expr]): _description_
            timeSymbol (sy.Expr): _description_

        Returns:
            _type_: _description_
        """
        baredVariables = [] #type : List[sy.Expr]
        for var in orgVariables :
            name = var.__getattribute__('name')            
            baredVariables.append(sy.Function(r'\bar{' + name+ '}')(timeSymbol))
        return baredVariables

    @property
    def SubstitutionDictionary(self) -> Dict[sy.Expr, SymbolOrNumber] :
        """The dictionary that should be used to store constant values that may appear 
        in the various expressions.  Many helper functions elsewhere want this dictionary 
        passed to it.

        Returns:
            Dict[sy.Expr, float]: The expression-to-values to substitute into expressions.
        """
        return self._substitutionDictionary

    @property
    def StateVariables(self) -> List[sy.Symbol]:
        """Gets the state variables for this problem.  These should be in terms of TimeSymbol. 
        This must be implemented by the derived type.

        Returns:
            List[sy.Expr]: The list of symbols in terms of TimeSymbol
        """
        return [x.Element for x in self._stateVariables]
    
    @property
    def ControlVariables(self) -> List[sy.Symbol]:
        """Gets a list of the control variables.  These should be in terms of TimeSymbol. 
        This must be implemented by the derived type.

        Returns:
            List[sy.Expr]: The list of the control variables.
        """
        return self._controlVariables

    @property
    def CostFunction(self) -> sy.Expr :
        """Gets the cost function as an expression.  This combines the TerminalCost with 
        the integrated UnIntegratedPathCost over time.

        Returns:
            sy.Expr: The overall cost function.
        """
        return self.TerminalCost + sy.integrate(self.UnIntegratedPathCost, (self.TimeSymbol, self.TimeInitialSymbol, self.TimeFinalSymbol))

    @property
    def TerminalCost(self) -> sy.Expr :
        """Gets the terminal cost of the problem.  Defaults to 0.

        Returns:
            sy.Expr: The terminal cost of the problem.
        """        
        return self._terminalCost

    @TerminalCost.setter
    def TerminalCost(self, value : sy.Expr) :
        """Sets the Terminal Cost of the function.

        Args:
            value (sy.Expr): The new terminal cost of the function.
        """
        self._terminalCost = value

    @property
    def UnIntegratedPathCost(self) -> sy.Expr :
        """Gets the un-integrated path cost of the trajectory.  For a problem of Bolza, this is the expression in the integral.

        Returns:
            sy.Expr: The un-integrated path cost of the constraint.  
        """        
        return self._unIntegratedPathCost

    @UnIntegratedPathCost.setter
    def UnIntegratedPathCost(self, value: sy.Expr) :
        """Sets the un-integrated path cost of the trajectory.  For a problem of Bolza, this is the expression in the integral.

        Args:
            value (sy.Expr): The un-integrated path cost.
        """
        self._unIntegratedPathCost = value

    @property
    def StateVariableDynamics(self) -> List[sy.Expr]:
        """Gets the expression of the dynamics of the state variables (generally, the rhs of the equations of motion)
        for each of the state variables. (for example, the rhs of the expression dm/dt = mDot*t, as symbols).  It is 
        assumed the every dynamical expression is only a first derivative, (d2m/dt2 is not supported).

        Returns:
            List[sy.Expr]: The first order dynamics of the state variables in the same order of the state variables
        """
        return [x.FirstOrderDynamics for x in self._stateVariables]
    
    @property
    def BoundaryConditions(self) ->List[sy.Expr] :
        """Gets the boundary conditions on the system.  These expressions 
        must equal 0 and symbols in them need to be in terms of Time0Symbol 
        or TimeFinalSymbol as appropriate.

        Returns:
            List[sy.Eq]: The boundary conditions. 
        """
        return self._boundaryConditions

    @property
    def TimeSymbol(self) -> sy.Symbol :
        """Gets the general time symbol.  Instead of using simple symbols for the state and 
        control variables, use sy.Function()(self.TimeSymbol) instead.

        Returns:
            sy.Expr: The time symbol.
        """        
        return self._timeSymbol

    @TimeSymbol.setter
    def TimeSymbol(self, value:sy.Symbol) :
        """Sets the general time symbol.  Instead of using simple symbols for the state and 
        control variables, use sy.Function()(self.TimeSymbol) instead.

        Args:
            value (sy.Expr): The time symbol. 
        """
        self._timeSymbol = value

    @property
    def TimeInitialSymbol(self) -> sy.Symbol :
        """Gets the symbol for the initial time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the start of the time span.

        Returns:
            sy.Expr: The initial time symbol.
        """        
        return self._timeInitialSymbol

    @TimeInitialSymbol.setter
    def TimeInitialSymbol(self, value:sy.Symbol) :
        """Sets the symbol for the initial time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the start of the time span.

        Args:
            value (sy.Expr): The initial time symbol.
        """
        self._timeInitialSymbol = value

    @property
    def TimeFinalSymbol(self) -> sy.Symbol :
        """Gets the symbol for the final time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the end of the time span.

        Returns:
            sy.Expr: The final time symbol.
        """        
        return self._timeFinalSymbol

    @TimeFinalSymbol.setter
    def TimeFinalSymbol(self, value : sy.Symbol) :
        """Sets the symbol for the final time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the end of the time span.

        Args:
            value (sy.Expr): The final time symbol.
        """
        self._timeFinalSymbol = value
    
    @property
    def Direction(self) -> IntegrationDirection:
        return self._integrationDirection

    @Direction.setter
    def Direction(self, value : IntegrationDirection) :
        self._integrationDirection = value

    def AddStateVariable(self, stateVariable: ProblemVariable):
        self._stateVariables.append(stateVariable)

    def AddCostateVariable(self, costateVariable : sy.Symbol):
        self._costateElements.append(ProblemVariable(costateVariable, None))

    def AddCostateVariables(self, costateVariable : List[sy.Symbol]):
        for lmd in costateVariable:
            self.AddCostateVariable(lmd)

    @property
    def CostateSymbols(self) ->List[sy.Symbol]:
        return [x.Element for x in self._costateElements]

    @property
    def CostateElements(self) ->List[sy.Symbol]:
        return self._costateElements

    def AddCostateElement(self, costateElement : ProblemVariable):
        self._costateElements.append(costateElement)

    @property 
    def CostateDynamicsEquations(self) ->List[sy.Expr]:
        return [x.FirstOrderDynamics for x in self._costateElements]

    @property
    def EquationsOfMotionAsEquations(self) -> List[sy.Eq] :
        """The equations of motions as symbolic equations where the LHS is the state variable differentiated by time 
        and the RHS the expression from EquationsOfMotion.  Expect renames in short order to make this the primary property

        Returns:
            List[sy.Eq]: The equations of motion as symbolic equations,
        """
        equations = []
        for i in range(0, len(self.StateVariables)):
            equations.append(sy.Eq(self.StateVariables[i].diff(self.TimeSymbol), self.StateVariableDynamics[i]))
        return equations

    @property
    def OtherArguments(self) -> List[sy.Symbol]:
        return self._otherArgs

    def CreateEquationOfMotionsAsEquations(self) -> List[sy.Eq] :
        """Converts the equations of motion dictionary into a list in the order of the state variables.

        Returns:
            List[sy.Expr]: The equations of motion in a list in the same order as the state variables.
        """
        tempArray = []
        for i in range(0, len(self.StateVariables)) :
            tempArray.append(sy.Eq(self.StateVariables[i].diff(self.TimeSymbol), self.StateVariableDynamics[i]))
        return tempArray

    def CreateCostFunctionAsEquation(self, lhs : Optional[sy.Expr]=None) -> sy.Eq :
        """Creates a sympy Eq of the cost function.

        Args:
            lhs (sy.Expr, optional): The left hand side of the equation. Defaults to None, which will make it J.

        Returns:
            sy.Eq: An equation of the cost function.
        """
        if(lhs == None) :
            lhs = sy.Symbol('J')
        return sy.Eq(lhs, self.CostFunction)

    def EquationsOfMotionInMatrixForm(self) -> sy.Matrix :
        """Converts the RHS's of the equations of motion into a sy.Matrix.

        Returns:
            sy.Matrix: The equations of motion in matrix form.
        """
        tempArray = []
        for i in range(0, len(self.StateVariables)) :
            tempArray.append(self.StateVariableDynamics[i])
        return Vector.fromArray(tempArray)

    def StateVariablesInMatrixForm(self) -> sy.Matrix :
        """Takes the list of state variables and turns them into a sympy Matrix.

        Returns:
            sy.Matrix: The state variables in a sympy Matrix.
        """
        return Vector.fromArray(self.StateVariables)

    def ControlVariablesInMatrixForm(self) -> sy.Matrix :
        """Takes the list of control variables and turns them into a sympy Matrix.

        Returns:
            sy.Matrix: The control variables in a sympy Matrix.
        """
        return Vector.fromArray(self.ControlVariables)        

    def AddInitialValuesToDictionary(self, subsDict : Dict, initialValuesArray : List, lambdas : Optional[List[sy.Expr]]=None):
        """Adds the initial values to the provided dictionary.

        Args:
            subsDict (Dict): The dictionary to add the initial values to.
            initialValuesArray (List): The values to add to the initial values.
            lambdas (List, optional): The symbols for the lambdas. If this is provided, then it is assumed the 
            initialValuesArray includes values for the lambda's at the initial time, and they will be 
            added to the passed in dictionary as well.  Defaults to None.
        """
        i = 0
        for sv in self.StateVariables :
            subsDict[sv.subs(self.TimeSymbol, self.TimeInitialSymbol)] = initialValuesArray[i]
            i=i+1
        if lambdas == None :
            return
        lambdas = cast(List[sy.Expr], lambdas)
        for lm in lambdas :
            subsDict[lm.subs(self.TimeSymbol, self.TimeInitialSymbol)] = initialValuesArray[i]
            i = i+1

    def CreateVariablesAtTime0(self, thingWithSymbols):
        """Substitutes the problems TimeSymbol and TimeFinalSymbol with the TimeInitialSymbol.
        This is generally used to change state or control variables from being functions of TimeSymbol 
        to functions of TimeInitialSymbol.

        Args:
            thingWithSymbols: Some set of expressions.

        Returns:
            same type as list of expressions: The expressions where the time variables has been set to the TimeInitialSymbol.
        """
        return SafeSubs(thingWithSymbols, {self.TimeSymbol: self.TimeInitialSymbol, self.TimeFinalSymbol:self.TimeInitialSymbol})

    def CreateVariablesAtTimeFinal(self, thingWithSymbols):
        """Substitutes the problems TimeSymbol and TimeInitialSymbol with the TimeFinalSymbol.
        This is generally used to change state or control variables from being functions of TimeSymbol 
        to functions of TimeFinalSymbol.

        Args:
            thingWithSymbols: Some set of expressions.

        Returns:
            same type as list of expressions: The expressions where the time variables has been set to the TimeFinalSymbol.
        """
        
        return SafeSubs(thingWithSymbols, {self.TimeSymbol: self.TimeFinalSymbol, self.TimeFinalSymbol:self.TimeFinalSymbol})


    @property
    def ScaleTimeFactor(self) -> bool:
        return self._timeScaleFactor != None

    @property
    def TimeScaleFactor(self) -> SymbolOrNumber:
        return self._timeScaleFactor

    @TimeScaleFactor.setter
    def TimeScaleFactor(self, value : SymbolOrNumber) :
        self._timeScaleFactor = value

    def ScaleStateVariables(self, newSvs : List[sy.Symbol], dictOfOriginalSvsToNewSvs : Dict[sy.Symbol, SymbolOrNumber]) ->Problem:
        # newSvs = [rBar, uBar, ...]
        # dictOfOriginalSvs = {r:rBar*r0, u:uBar*v0, ...}
        
        dNewSvWrtOldSv = []
        bcSubsDict = {}
        descalingDict = {}
        simpleNewSvReplacement = {}
        if len(newSvs) != len(self.StateVariables):
            raise Exception("When scaling state variables, all state variables must be present (even if the scaling is just 1)")

        # the expression for scaling expressions is (scaling x to x1):
        # dx/dt = dx1/dx*dx/dt
        # So for the dynamics, substitute {x:x1} and then multiple by dx1/dt
        for i in range(0, len(newSvs)):            
            newSv = newSvs[i]
            oldSv = self.StateVariables[i]
            scaleExpEq = sy.Eq(oldSv, dictOfOriginalSvsToNewSvs[oldSv])
            newSvInTermsOfOld = sy.solve(scaleExpEq, newSv)[0].diff(oldSv) # this generally should be a simple expression and not too complicated #TODO: Do I need to specify both?
            descalingDict[newSv] = newSvInTermsOfOld
            dNewSvWrtOldSv.append(newSvInTermsOfOld)
            #bcSubsDict[oldSv.subs(self.TimeSymbol, self.TimeFinalSymbol)] = newSv.subs(self.TimeSymbol, self.TimeFinalSymbol)
            bcSubsDict[oldSv.subs(self.TimeSymbol, self.TimeFinalSymbol)] = SafeSubs(dictOfOriginalSvsToNewSvs[oldSv], {self.TimeSymbol: self.TimeFinalSymbol})
            simpleNewSvReplacement[oldSv] = newSv

        # fill in the new problem (non destructive modification)
        newProblem = Problem() #TODO: Clone/copy abstract method
        newProblem._timeSymbol = self._timeSymbol
        newProblem._timeInitialSymbol = self._timeInitialSymbol
        newProblem._timeFinalSymbol= self._timeFinalSymbol
        newProblem._substitutionDictionary = {}
        for (k,v) in self._substitutionDictionary.items() :
            newProblem._substitutionDictionary[k.subs(dictOfOriginalSvsToNewSvs)] = SafeSubs(v, dictOfOriginalSvsToNewSvs)
        newProblem._controlVariables = self.ControlVariables
        newProblem._scalingDict = dict(dictOfOriginalSvsToNewSvs) #make a copy since this problem is not the owner of the original
        newProblem._descaleDict = descalingDict
        for (k,v) in self._scalingDict:
            newProblem._scalingDict[k]=v

        # sv's and dynamics
        for i in range(0, len(self.StateVariables)):
            oldSv = self.StateVariables[i]
            newSv = newSvs[i]
            newProblem.AddStateVariable(ProblemVariable(newSv, SafeSubs(self.StateVariableDynamics[i], dictOfOriginalSvsToNewSvs)*dNewSvWrtOldSv[i]))

        # do the BC's (including transversality condition if already added)
        for bc in self.BoundaryConditions :
            newProblem._boundaryConditions.append(SafeSubs(bc,bcSubsDict))

        # do the costs
        newProblem._unIntegratedPathCost = SafeSubs(self.UnIntegratedPathCost, dictOfOriginalSvsToNewSvs)
        finalTimeSubsDict = {}
        timeToFinalTimeSubsDict = {newProblem.TimeSymbol: newProblem.TimeFinalSymbol}
        for k in range(0, len(self.StateVariables)):
            sv = self.StateVariables[k]
            #finalTimeSubsDict[sv.subs(timeToFinalTimeSubsDict)] = SafeSubs(dictOfOriginalSvsToNewSvs[sv], timeToFinalTimeSubsDict)
            finalTimeSubsDict[sv.subs(timeToFinalTimeSubsDict)] = SafeSubs(simpleNewSvReplacement[sv], timeToFinalTimeSubsDict)
        newProblem._terminalCost = SafeSubs(self.TerminalCost, finalTimeSubsDict)

        # if there are costate variables, copy them (not sure if this is a good idea)
        #TODO: See if this is valid

        for j in range(0, len(self.CostateSymbols)):
            newProblem.AddCostateElement(ProblemVariable(self.CostateSymbols[i],SafeSubs(self.CostateDynamicsEquations[i], dictOfOriginalSvsToNewSvs)*dNewSvWrtOldSv[j]))



        return newProblem

    def ScaleTime(self, newTimeSymbol, newInitialTimeSymbol, newFinalTimeSymbol, expressionForNewTime) ->Problem:        
        # for dx/dt = dx/dTau * dTau/dt
        # so evaluate dTau/dt, substitute t for expression with Tau and for dynamics multiply by dTau/dt
        dtDTau = 1/sy.solve(sy.Eq(self.TimeSymbol, expressionForNewTime), newTimeSymbol)[0].diff(self.TimeSymbol)
        dictOfOriginalSvsToNewSvs = {self.TimeSymbol: expressionForNewTime } #TODO: Need to subs time initial and time final?

        symbolSubs = {self.TimeSymbol: newTimeSymbol} #Dict[sy.Symbol, sy.Expr]

        # fill in the new problem (non destructive modification)
        newProblem = Problem() #TODO: Clone/copy abstract method
        newProblem._timeSymbol = newTimeSymbol
        newProblem._timeInitialSymbol = newInitialTimeSymbol
        newProblem._timeFinalSymbol= newFinalTimeSymbol
        newProblem._timeScaleFactor = expressionForNewTime
        newProblem._substitutionDictionary = {}
        for (k,v) in self._substitutionDictionary.items() :
            newProblem._substitutionDictionary[k.subs(symbolSubs)] = SafeSubs(v, symbolSubs)
        timeDescaleDict = {newTimeSymbol: self.TimeSymbol, newInitialTimeSymbol: self.TimeInitialSymbol, newFinalTimeSymbol: self.TimeFinalSymbol}    
        newProblem._descaleDict={}
        for k in self._descaleDict.keys():
            newProblem._descaleDict[SafeSubs(k, symbolSubs)] = SafeSubs(self._descaleDict[k], symbolSubs)
        #newProblem._controlVariables = self.ControlVariables # need to do control variables separately below
        newProblem._scalingDict = dict(dictOfOriginalSvsToNewSvs) #make a copy since this problem is not the owner of the original
        for (k,v) in self._scalingDict.items():
            newProblem._scalingDict[k]=v

        bcSubsDict = {}
        # sv's and dynamics
        newSvs = []
        for sv in self.StateVariables:
            newSv = SafeSubs(sv, symbolSubs)
            newSvs.append(newSv)
            dictOfOriginalSvsToNewSvs[sv] = newSv
            bcSubsDict[sv.subs(self.TimeSymbol, self.TimeFinalSymbol)] = newSv.subs(newTimeSymbol, newFinalTimeSymbol)

        # control exp
        badControlExpFix = {}
        for cv in self.ControlVariables:
            goodCv = SafeSubs(cv, symbolSubs)
            newProblem.ControlVariables.append(goodCv)
            badControlExpFix[SafeSubs(cv, dictOfOriginalSvsToNewSvs)] = goodCv

        for i in range(0, len(self.StateVariables)):
            oldSv = self.StateVariables[i]
            dynamics = SafeSubs(self.StateVariableDynamics[i], dictOfOriginalSvsToNewSvs)*dtDTau
            fixedDynamics = SafeSubs(dynamics, badControlExpFix)
            newProblem.AddStateVariable(ProblemVariable(newSvs[i], fixedDynamics))

        # do the BC's (including transversality condition if already added)
        for bc in self.BoundaryConditions :
            newProblem._boundaryConditions.append(SafeSubs(bc,bcSubsDict))

        # do the costs
        newProblem._unIntegratedPathCost = SafeSubs(self.UnIntegratedPathCost, dictOfOriginalSvsToNewSvs)
        dictOfFinalSymbols = {}
        simpleNewTimeFinalSubsDict = {self.TimeSymbol: newFinalTimeSymbol}
        oldTimeToTimeFinalSubsDict = {self.TimeSymbol: self.TimeFinalSymbol}
        for sv in self.StateVariables:
            dictOfFinalSymbols[sv.subs(oldTimeToTimeFinalSubsDict)] = sv.subs(simpleNewTimeFinalSubsDict)

        #newProblem._terminalCost = SafeSubs(self.TerminalCost, dictOfOriginalSvsToNewSvs)
        #newProblem._terminalCost = SafeSubs(newProblem._terminalCost, dictOfFinalSymbols)

        newProblem._terminalCost = SafeSubs(self.TerminalCost, dictOfFinalSymbols)
        # if there are costate variables, copy them (not sure if this is a good idea)
        #TODO: See if this is valid

        for j in range(0, len(self.CostateSymbols)):
            newProblem.AddCostateElement(ProblemVariable(SafeSubs(self.CostateSymbols[i], symbolSubs), SafeSubs(self.CostateDynamicsEquations[i], dictOfOriginalSvsToNewSvs)*dtDTau))

        newProblem._timeScaleFactor =self.TimeFinalSymbol #TODO: User needs to set this to some degree
        newProblem._otherArgs.append(self.TimeFinalSymbol)
        
        return newProblem        


    def ScaleExpressions(self, expressions : List[sy.Expr]) -> List[sy.Expr]:
        """For some expression (or list of expressions), scale them by the scaling factors set up in this problem.
        If the problem is not scaled, the original expressions will be returned.

        Args:
            expressions (List[sy.Expr]): The expressions to scale.

        Returns:
            List[sy.Expr]: The scaled expressions
        """
        # simpleSubsDict={} # for terminal cost
        # counter=0
        # for sv in self._unscaledStateVariables :
        #     oldSv = sv
        #     sv = self.StateVariables[counter]
        #     simpleSubsDict[oldSv] = sv*self._scalingDict[oldSv]
        #     simpleSubsDict[oldSv.subs(self._unscaledTimeSymbol, self._unscaledTimeFinalSymbol)] = sv.subs(self.TimeSymbol, self.TimeFinalSymbol)*self._scalingDict[oldSv]
        #     counter=counter+1

        return SafeSubs(expressions, self._scalingDict)        

    def DescaleResults(self, resultsDictionary : Dict[sy.Symbol, List[float]]) -> Dict[sy.Symbol, List[float]] :
        """After evaluating the problem numerically, descale the results to be back in terms of the original units.

        Args:
            resultsDictionary (Dict[sy.Symbol, List[float]]): The results dictionary.

        Returns:
            Dict[sy.Symbol, List[float]]: A new dictionary where the values are descaled AND the keys are the wrappedProblems's 
            state variables.
        """

        

        if self._descaleDict == None or len(self._descaleDict) == 0:
            return resultsDictionary
        returnDict = {} #type: Dict[sy.Symbol, List[float]]
        counter = 0
        scalingFactors ={}
        for k,v in self._descaleDict.items():
            factor = SafeSubs(v, self.SubstitutionDictionary)
            scalingFactors[k] = factor
        for key, value in resultsDictionary.items() :
            sv = key
            if sv in self.StateVariables and counter < len(self.StateVariables):
                originalSv = self.StateVariables[self.StateVariables.index(sv)]
                convertedArray = np.array(value, copy=True)/scalingFactors[sv]# SafeSubs(self._descaleDict[originalSv], self.SubstitutionDictionary) #TODO: This is probably a problem
                returnDict[originalSv] = convertedArray
                counter = counter+1
            else :
                returnDict[key]=value

        return returnDict

    @staticmethod
    def CreateCostateVariables(y, name : Optional[str] = None, t : Optional[sy.Symbol]=None) -> Vector:
        """Creates a co-vector for the entered y.

        Args:
            y: A sympy Symbol, Function, Matrix, ImmutableDenseMatrix.  Can also be a list of Symbols or Functions.
            name (str): The name to use for the co-vector variables.  The names of the original symbols will be appended as a subscript.
            t (sy.Symbol, optional): If y ought to be a sy.Function instead of a constant Symbol, this is the independent variable for that Function. Defaults to None.

        Returns:
            same type as y if supported: the costate value for the entered y
        """

        # this function is less ugly...
        if name == None :
            name = r'\lambda'
        name = cast(str, name) #TODO: shouldn't be necessary...
        if((y is sy.Symbol or y is sy.Function or (hasattr(y, "is_Function") and y.is_Symbol) or (hasattr(y, "is_Symbol") and y.is_Function)) and (not isinstance(y, sy.Matrix) and not isinstance(y, sy.ImmutableDenseMatrix))):
            if(t is None) :
                return sy.Symbol(name + '_{'+y.name+'}', real=True)
            else :
                return (sy.Function(name + '_{'+y.name+'}', real=True)(t))

        if(isinstance(y, list) ) :
            coVector = []
            for i in range(0, len(y)):
                coVector.append(Problem.CreateCostateVariables(y[i], name, t))
            return coVector

        coVector = Vector.zeros(y.shape[0])
        for i in range(0, y.shape[0]):
            coVector[i] = Problem.CreateCostateVariables(y[i], name, t)
        return coVector

    @staticmethod
    def CreateCostateElements(y, name : Optional[str] = None, t : Optional[sy.Symbol]=None) -> List[ProblemVariable]:
        lambdas = Problem.CreateCostateVariables(y, name, t)
        return [ProblemVariable(lmd, None) for lmd in lambdas]

    @staticmethod
    def CreateHamiltonianStatic(t, equationsOfMotionMatrix, unIntegratedPathCost, lambdas) -> sy.Expr:
        """Creates an expression for the Hamiltonian.

        Args:
            lambdas: The costate variables. Defaults to None in which case 
            they will be created.

        Returns:
            sy.Expr: The Hamiltonian.
        """
        if isinstance(lambdas, list) :
            lambdas = Vector.fromArray(lambdas)

        secTerm =  (lambdas.transpose()*equationsOfMotionMatrix)[0,0]
        return secTerm+unIntegratedPathCost    

    @staticmethod
    def CreateHamiltonianControlExpressionsStatic( hamiltonian : sy.Expr, controlVariables) -> sy.Matrix:
        return sy.Derivative(hamiltonian, controlVariables).doit()  

    @staticmethod
    def CreateControlExpressionsFromHamiltonian(hamiltonian : sy.Expr, controlVariables: List[sy.Symbol]) -> Dict[sy.Symbol, sy.Expr] :
        ans = {}
        for control in controlVariables:
            controlExpression = Problem.CreateHamiltonianControlExpressionsStatic(hamiltonian, control)
            thisAns = sy.solve(sy.Eq(0, controlExpression), control)
            ans[control] = thisAns[0]
        return ans

    @staticmethod
    def CreateLambdaDotConditionStatic(hamiltonian, stateVariables) :
        return -1*sy.Derivative(hamiltonian, stateVariables).doit()

    @staticmethod
    def TransversalityConditionInTheDifferentialFormStatic(hamiltonian : sy.Expr, dtf, lambdasFinal : List[sy.Symbol], terminalCost : sy.Expr, tf : sy.Symbol, boundaryConditions : List[sy.Expr], finalStateVariables : List[sy.Symbol]) ->List[sy.Expr]:
        """Creates the transversality conditions by with the differential form of the transversality conditions. 

        Args:
            hamiltonian (sy.Expr): The hamiltonian in terms of the costate values (as opposed to the control variable)
            dtf (_type_): Either 0 if the final time is fixed, or a symbol indicating that the final time is not fixed.
            lambdasFinal (List[sy.Symbol]): The costate symbols at the final time.  If None it will use the problems
            CostateSymbols at the final time, and if those are not set, then an exception will be raised.

        Returns:
            List[sy.Expr]: The list of transversality conditions, that ought to be treated like normal boundary conditions.
        """
       
        variationVector = []
        valuesAtEndSymbols = []
        if isinstance(terminalCost, float) :
            valuesAtEndDiffTerm = 0
        else :
            valuesAtEndDiffTerm = sy.diff(terminalCost, tf).expand().simplify()

        # work on the dPsi term
        for bc in boundaryConditions:
            expr = sy.diff(bc, tf).doit().powsimp().simplify()
            if(expr != None and expr != 0) :
                valuesAtEndSymbols.append(expr)

        # create the variation vector
        # If BC's are such that the final optimal value of the sv at tf is fixed (its value is a float and derivative is 0)
        # then the variation vector must be 0 for that final value.
        # If however, BC's are such that the final value of the state variable can be different along 
        # the optimal trajectory, then the variation vector for it is the symbol d_stateValue/d_tf, which 
        # for a well posed problem will give us several equations we can use as additional BC's when 
        # we build the entire transversality condition equation and solve for coefficients to be 0
        finalSvs = finalStateVariables
        for sv in finalSvs :     
            if sv in lambdasFinal :
                continue  
            dxfdtf = sy.diff(sv, tf).doit() 
            notFixed = True # by default we should assume that state variables are not fixed
            for bc in boundaryConditions :
                derVal = bc.diff(tf)/dxfdtf   
                # if it does not have is_float, then it is some other sympy expression which means the SV is variable at tf
                # and if it does have is_float AND is_float, well then it truly is fixed and its variation is 0
                notFixed = not(hasattr(derVal, "is_Number") and derVal.is_Number)
                if not notFixed:
                    break 
                # we know that this sv can vary WRT the final time, so stop looking through the BC's and add
                # dxfdtf to the variation vector
                
            if not notFixed : # so it is fixed...
                dxfdtf = 0.0 # still need to apply the 0 case
            variationVector.append(dxfdtf)

        variationVector = Vector.fromArray(variationVector)    
        lambdasFinalVector = Vector.fromArray(lambdasFinal)
        overallCond = hamiltonian*dtf - (lambdasFinalVector.transpose()*variationVector)[0,0] + valuesAtEndDiffTerm
        overallCond = overallCond.expand()
        
        for vv in variationVector :
            found = False
            for dPsiTerm in valuesAtEndSymbols:        
                simpleSolve = sy.solve(dPsiTerm, vv)
                if(len(simpleSolve) > 0) :
                    overallCond = overallCond.subs(vv, simpleSolve[0])
                    found = True
            if found :
                continue

        transversalityConditions = []
        for dx in variationVector :
            coef = overallCond.coeff(dx)
            if(coef != 0.0) :
                transversalityConditions.append(coef)    

        return transversalityConditions  

    @staticmethod
    def CreateLambdaDotEquationsStatic(hamiltonian : sy.Expr, t : sy.Symbol, stateVariables, lambdaSymbols) :
        rightHandSides = []
        for i in range(0, len(stateVariables)):
            dThisDH = -1*sy.Derivative(hamiltonian, stateVariables[i]).doit()
            rightHandSides.append(dThisDH)
        eqs = []
        for i in range(0, len(lambdaSymbols)) :
            eqs.append(sy.Eq(lambdaSymbols[i].diff(t), rightHandSides[i]))
        return eqs

    @staticmethod
    def TransversalityConditionsByAugmentationStatic(xf, terminalCost, boundaryConditions, nus : List[sy.Symbol], lambdasFinal : List[sy.Expr]) -> List[sy.Expr]:
        termFunc = terminalCost + (Vector.fromArray(nus).transpose()*Vector.fromArray(boundaryConditions))[0,0]
        transversalityConditions = []
        i=0
        for lmd in lambdasFinal :
            cond = termFunc.diff(xf[i])
            transversalityConditions.append(lmd-cond)
            i=i+1

        return transversalityConditions
    
    def CreateHamiltonian(self, lambdas = None) -> sy.Expr:
        """Creates an expression for the Hamiltonian.

        Args:
            lambdas (optional): The costate variables. Defaults to None in which case 
            they will be created.

        Returns:
            sy.Expr: The Hamiltonian.
        """
        if lambdas == None:
            lambdas = self.CostateSymbols
        return Problem.CreateHamiltonianStatic(self.TimeSymbol, self.EquationsOfMotionInMatrixForm(), self.UnIntegratedPathCost, lambdas)

    def CreateHamiltonianControlExpressions(self, hamiltonian : sy.Expr) -> sy.Matrix:
        """Creates the an expression that can be used to solve for values of the control scalars 
        in terms of the co-state equations. Note that this is difficult to generalize symbolically 
        as often there is some novel or unique operations that can be performed to simplify this 
        expression more than what simply solving for the variables would yield.

        Args:
            hamiltonian (sy.Expr): The Hamiltonian.

        Returns:
            sy.Expr: An expression that can be used to solve for the control variables.
        """
        u = self.ControlVariablesInMatrixForm()
        return Problem.CreateHamiltonianControlExpressionsStatic(hamiltonian, u)

    def TransversalityConditionsByAugmentation(self, nus : List[sy.Symbol], lambdasFinal : List[sy.Expr]) -> List[sy.Expr]:
        """Creates the transversality conditions by augmenting the terminal constraints to the terminal cost.

        Args:
            nus (List[sy.Symbol]): The constant parameters to augment the constraints to the terminal cost with.
            lambdasFinal (List[sy.Symbol]): The costate symbols at the final time.  If None it will use the problems
            CostateSymbols at the final time, and if those are not set, then an exception will be raised.

        Returns:
            List[sy.Expr]: The list of transversality conditions, that ought to be treated like normal boundary conditions. It is assumed that these 
            expressions should be solved such that they equal 0
        """
        lambdasFinal = cast(List[sy.Expr], lambdasFinal)
        termFunc = self.TerminalCost.subs(self.TimeSymbol, self.TimeFinalSymbol) + (Vector.fromArray(nus).transpose()*Vector.fromArray(self.BoundaryConditions))[0,0]
        finalConditions = []
        i=0
        for x in self.StateVariables :
            if i >= len(lambdasFinal) :
                break
            xf = x.subs(self.TimeSymbol, self.TimeFinalSymbol)
            cond = termFunc.diff(xf)
            finalConditions.append(lambdasFinal[i]-cond)
            i=i+1

        return finalConditions

    def TransversalityConditionInTheDifferentialForm(self, hamiltonian : sy.Expr, dtf, lambdasFinal : List[sy.Symbol]) ->List[sy.Expr]:
        problemToWorkWith = self
        # if self._wrappedProblem != None :
        #     problemToWorkWith = self._wrappedProblem                
        finalSvs = SafeSubs(problemToWorkWith.StateVariables, {problemToWorkWith.TimeSymbol: problemToWorkWith.TimeFinalSymbol})
        for lmd in lambdasFinal :
            if lmd in finalSvs :
                finalSvs.remove(lmd)
        bcsToUse = problemToWorkWith.BoundaryConditions
        #if problemToWorkWith.TimeScaleFactor != None:
        #    bcsToUse = bcsToUse[:-1]
        transversalityConditions = Problem.TransversalityConditionInTheDifferentialFormStatic(hamiltonian, dtf, lambdasFinal, problemToWorkWith.TerminalCost, problemToWorkWith.TimeFinalSymbol, bcsToUse, finalSvs)
        return transversalityConditions

    def CreateLambdaDotCondition(self, hamiltonian) :
        """For optimal control problems, create the differential equations for the 
        costate scalars.

        Args:
            hamiltonian (_type_): The Hamiltonian equation of the system.

        Returns:
            _type_: The derivatives of the costate values in the same order as the state 
            variables.
        """
        x = self.StateVariablesInMatrixForm()
        return Problem.CreateLambdaDotConditionStatic(hamiltonian, x)

    def EvaluateHamiltonianAndItsFirstTwoDerivatives(self, solution : Dict[sy.Symbol, List[float]], tArray: Collection[float], hamiltonian : sy.Expr, controlSolved :Dict[sy.Expr, sy.Expr], moreSubs :Dict[sy.Symbol, float]) ->List[List[float]]:
        """Evaluates the Hamiltonian and its first 2 derivatives.  This is useful to 
        see if the related conditions are truly satisfied.

        Args:
            solution (Dict[sy.Expr, List[float]]): The solution of the optimal control problem.
            tArray (List[float]): The time corresponding to the solution.
            hamiltonian (sy.Expr): The Hamiltonian expression.
            controlSolved ([sy.Expr, sy.Expr]): The Hamiltonian is likely in terms of the original control variable instead of the costate values.  If that is the case, this should be the expression of the control variables in terms of the costate variables.
            moreSubs (Dict[sy.Expr, float]): Any additional values to substitute into the expressions (if the final time was solved for, or if there were other parameters not included in the problems SubstitutionDictionary).

        Returns:
            List[List[float]]: The values of the Hamiltonian, its first derivative and second derivative for the entered solution.
        """

        stateForEom = [self.TimeSymbol, *self.StateVariables, *self.CostateSymbols]

        constantsSubsDict = self.SubstitutionDictionary

        dHdu = self.CreateHamiltonianControlExpressions(hamiltonian).doit()[0]
        d2Hdu2 = sy.diff(hamiltonian, self.ControlVariables[0], 2)
        #d2Hdu2 =  self.CreateHamiltonianControlExpressions(dHdu).doit()[0] # another way to calculate it, but doesn't seem to be as good
        toEval = hamiltonian.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict)
        hamiltonianExpression = sy.lambdify(stateForEom, toEval)
        solArray = []
        for sv in self.StateVariables :
            solArray.append(np.array(solution[sv]))
        for sv in self.CostateSymbols :
            solArray.append(np.array(solution[sv]))            
        # for sv in self.CostateSymbols :
        #     solArray.append(np.array(solution[sv]))
        hamiltonianValues = hamiltonianExpression(tArray, *solArray)
        dHdUExp = sy.lambdify(stateForEom, dHdu.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict))
        
        dHdUValues = dHdUExp(tArray, *solArray)       
        if not hasattr(dHdUValues, "__len__") or len(dHdUValues) != len(hamiltonianValues) :
            dHdUValues = [dHdUValues] * len(hamiltonianValues)
        d2hdu2Exp = sy.lambdify(stateForEom, d2Hdu2.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict))
        d2hdu2Values = d2hdu2Exp(tArray, *solArray)
        return [hamiltonianValues, dHdUValues, d2hdu2Values]
