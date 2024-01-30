import sympy as sy
import numpy as np
from typing import List, Dict, cast, Optional
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from abc import ABC
from collections import OrderedDict
from enum import Enum
from pyeq2orb import SafeSubs
from pyeq2orb.Symbolics.Vectors import Vector # type: ignore

class IntegrationDirection(Enum) :
    Forward = 1
    Backward = -1

class Problem(ABC) :
    def __init__(self) :
        """Initialize a new instance. 
        """
        self._stateVariables = []
        self._controlVariables = []
        self._terminalCost = 0
        self._unIntegratedPathCost = 0
        self._stateVariableDynamics = []
        self._boundaryConditions = []
        self._timeSymbol = None
        self._timeInitialSymbol = None
        self._timeFinalSymbol= None
        self._integrationDirection = IntegrationDirection.Forward
        self._substitutionDictionary = OrderedDict()
        self._timeScaleFactor = None
        self._scalingDict = {}

        self._wrappedProblem = None #TODO: I want to get rid of this...

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
        return self._stateVariables
    
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
        return self._stateVariableDynamics
    
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

    def AddStateVariable(self, stateVariable, stateVariableDynamics):
        self.StateVariables.append(stateVariable)
        self.StateVariableDynamics.append(stateVariableDynamics)

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
    def ScaleTime(self) -> bool:
        return self._timeScaleFactor != None

    @property
    def TimeScaleFactor(self) -> SymbolOrNumber:
        return self._timeScaleFactor

    @TimeScaleFactor.setter
    def TimeScaleFactor(self, value : SymbolOrNumber) :
        self._timeScaleFactor = value


    def ScaleProblem(self, newStateVariableSymbols : List[sy.Symbol], valuesToDivideStateVariablesWith : Dict[sy.Symbol, SymbolOrNumber], timeScaleFactor : Optional[SymbolOrNumber] = None):
        newProblem = Problem()
        newProblem._scaleProblem(self, newStateVariableSymbols, valuesToDivideStateVariablesWith, timeScaleFactor)
        return newProblem

    def _scaleProblem(self, originalProblem, newStateVariableSymbols : List[sy.Symbol], valuesToDivideStateVariablesWith : Dict[sy.Symbol, SymbolOrNumber], timeScaleFactor : Optional[SymbolOrNumber] = None):
        wrappedProblem = originalProblem
        newProblem = self
        newProblem._wrappedProblem = wrappedProblem #TODO: Want to get rid of this...
        newProblem._timeFinalSymbol = wrappedProblem.TimeFinalSymbol
        newProblem._timeInitialSymbol = wrappedProblem.TimeInitialSymbol
        newProblem._timeSymbol = wrappedProblem.TimeSymbol
        
        newProblem._timeScaleFactor = timeScaleFactor 
        newProblem._substitutionDictionary = wrappedProblem._substitutionDictionary        
        newProblem._stateVariables = newStateVariableSymbols
        
        newProblem._scalingDict = valuesToDivideStateVariablesWith

        #newProblem._stateVariables.extend(newStateVariableSymbols)

        svAtTf=SafeSubs(wrappedProblem.StateVariables, {wrappedProblem.TimeSymbol: wrappedProblem.TimeFinalSymbol})
        newSvAtTf=SafeSubs(newStateVariableSymbols, {wrappedProblem.TimeSymbol: wrappedProblem.TimeFinalSymbol})
        svAtTfSubsDict = dict(zip(svAtTf, newSvAtTf))
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


        newProblem._controlVariables = wrappedProblem.ControlVariables
        scaledEquationsOfMotion = []
        for i in range(0, len(wrappedProblem.StateVariableDynamics)) :
            sv = wrappedProblem.StateVariables[i]
            newLhs = wrappedProblem.StateVariableDynamics[i].subs(valuesToDivideStateVariablesWith)
            if hasattr(wrappedProblem.StateVariableDynamics[i], "subs") :
                scaledEquation = wrappedProblem.StateVariableDynamics[i].subs(fullSubsDict)*newSvsInTermsOfOldSvs[i].diff(sv)
            else :
                scaledEquation = wrappedProblem.StateVariableDynamics[i]*newSvsInTermsOfOldSvs[i].diff(sv)
            scaledEquationsOfMotion.append(scaledEquation)
        newProblem._stateVariableDynamics = scaledEquationsOfMotion
        
        counter = 0
        bcSubsDict = {}
        for sv in wrappedProblem.StateVariables :
            newSvAtTf = newStateVariableSymbols[counter].subs(wrappedProblem.TimeSymbol, wrappedProblem.TimeFinalSymbol)
            bcSubsDict[svAtTf[counter]] = newSvAtTf*valuesToDivideStateVariablesWith[sv]
            counter = counter+1
        bcs = []
        
        for bc in wrappedProblem.BoundaryConditions :
            bcs.append(bc.subs(bcSubsDict))
        newProblem._boundaryConditions = bcs
        
        updatedSubsDict = {}

        # should the path cost get scaled?  I think so, but if you know better, or if it doesn't work...
        newProblem._unIntegratedPathCost = SafeSubs(wrappedProblem.UnIntegratedPathCost, fullSubsDict)
        #TODO: Scale the costs!  the way we are doing the transversality conditions are fine with them scaled here (I think)
        newProblem._terminalCost = SafeSubs(wrappedProblem.TerminalCost, svAtTfSubsDict)
        if newProblem.ScaleTime :
            timeScaleFactor = cast(SymbolOrNumber, timeScaleFactor)
            tf = wrappedProblem.TimeFinalSymbol
            tau = sy.Symbol(r'\tau')
            tauF = sy.Symbol(r'\tau_f')
            tau0 = sy.Symbol(r'\tau_0')
            timeSubs = {wrappedProblem.TimeInitialSymbol: tau0, wrappedProblem.TimeSymbol: tau, wrappedProblem.TimeFinalSymbol:tauF}
            newProblem._controlVariables = SafeSubs(newProblem._controlVariables, timeSubs)
            orgSv = newProblem._stateVariables
            newProblem._stateVariables = SafeSubs(newProblem._stateVariables, timeSubs)
            newProblem._unIntegratedPathCost = SafeSubs(newProblem._unIntegratedPathCost, timeSubs)
            newProblem._terminalCost = SafeSubs(newProblem._terminalCost, timeSubs)
            newProblem._boundaryConditions = SafeSubs(newProblem._boundaryConditions, timeSubs)

            # the correct thing to do is to make your state and control variables functions of time in Sympy
            # myCv = sy.Function('Control')(t)
            # BUT, we need to substitute t with tau*tf
            # BUT, we don't want myCv to be a function of tau*tf, just of tau
            # I don't know how to make the sympy subs function not go deep like that, so, we substitute these back...
            toSimple = {}
            fromSimple = {}
            for sv in newProblem._stateVariables :
                adjustedSv = sv.subs(tau, wrappedProblem.TimeSymbol)
                if not hasattr(sv, "name") :
                    raise Exception('State variable ' + str(sv) + " needs to have a name attribute")
                toSimple[adjustedSv] = sy.Symbol(sv.name) #type: ignore
                fromSimple[toSimple[adjustedSv]] = sv
                
            for cv in newProblem._controlVariables :
                adjustedCv = cv.subs(tau, wrappedProblem.TimeSymbol)
                toSimple[adjustedCv] = sy.Symbol(cv.name)
                fromSimple[toSimple[adjustedCv]] = cv
                
            realEom = []
            #timeSubs = { wrappedProblem.TimeSymbol: tau*timeScaleFactor}
            for i in range(0, len(newProblem.StateVariableDynamics)) :
                # substitute in the dummy symbols in terms of something other than time or tau
                thisUpdatedEom = SafeSubs(newProblem.StateVariableDynamics[i], toSimple)
                # substitute in the scaled values
                thisUpdatedEom = SafeSubs(thisUpdatedEom, timeSubs)*timeScaleFactor
                # substitute back in the state variables in terms of time
                thisUpdatedEom = SafeSubs(thisUpdatedEom, fromSimple)
                realEom.append(thisUpdatedEom)
                
            newProblem._stateVariableDynamics = realEom
            
            newProblem._timeSymbol = tau 
            newProblem._timeInitialSymbol = tau0
            newProblem._timeFinalSymbol = tauF       
            #newProblem.ControlVariables.append(cast(sy.Symbol, timeScaleFactor))
        
            #updatedSubsDict[tauF] = 1
            #updatedSubsDict[tau0] = 0
            for (k,v) in newProblem.SubstitutionDictionary.items():
                newK = SafeSubs(k,timeSubs)
                newV = SafeSubs(v, timeSubs)
                updatedSubsDict[newK] = newV
                #newProblem.SubstitutionDictionary[newK] = newV
        for (k,v) in updatedSubsDict.items() :
            newProblem.SubstitutionDictionary[k]=v

        #return newProblem

    def ScaleExpressions(self, expressions : List[sy.Expr]) -> List[sy.Expr]:
        """For some expression (or list of expressions), 

        Args:
            expressions (List[sy.Expr]): The expressions to scale.

        Returns:
            List[sy.Expr]: The scaled expressions
        """
        simpleSubsDict={} # for terminal cost
        counter=0
        for sv in self._wrappedProblem.StateVariables :
            oldSv = sv
            sv = self.StateVariables[counter]
            simpleSubsDict[oldSv] = sv*self._scalingDict[oldSv]
            simpleSubsDict[oldSv.subs(self._wrappedProblem.TimeSymbol, self._wrappedProblem.TimeFinalSymbol)] = sv.subs(self.TimeSymbol, self.TimeFinalSymbol)*self._scalingDict[oldSv]
            counter=counter+1

        return SafeSubs(expressions, simpleSubsDict)        

    def DescaleResults(self, resultsDictionary : Dict[sy.Symbol, List[float]]) -> Dict[sy.Symbol, List[float]] :
        """After evaluating the problem numerically, descale the results to be back in terms of the original units.

        Args:
            resultsDictionary (Dict[sy.Symbol, List[float]]): The results dictionary.

        Returns:
            Dict[sy.Symbol, List[float]]: A new dictionary where the values are descaled AND the keys are the wrappedProblems's 
            state variables.
        """
        if not hasattr(self, '_wrappedProblem '):
            return resultsDictionary
        returnDict = {} #type: Dict[sy.Symbol, List[float]]
        counter = 0
        for key, value in resultsDictionary.items() :
            sv = key
            if sv in self.StateVariables and counter < len(self.StateVariables):
                originalSv = self.StateVariables[self.StateVariables.index(sv)]
                convertedArray = np.array(value, copy=True)* SafeSubs(self._scalingDict[originalSv], self.SubstitutionDictionary)
                returnDict[originalSv] = convertedArray
                counter = counter+1
            else :
                returnDict[key]=value
        return returnDict