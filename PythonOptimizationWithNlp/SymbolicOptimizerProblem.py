from inspect import BoundArguments
import sympy as sy
from typing import List, Dict
from collections import OrderedDict
from PythonOptimizationWithNlp.Symbolics.Vectors import Vector
from abc import abstractmethod, ABC
import numpy as np
import matplotlib.pyplot as plt

class SymbolicProblem(ABC) :
    def __init__(self) :
        """Initialize a new instance.
        """
        self._stateVariables = []
        self._controlVariables = []
        self._terminalCost = 0
        self._unIntegratedPathCost = 0
        self._equationsOfMotion = OrderedDict()
        self._boundaryConditions = []
        self._timeSymbol = None
        self._timeInitialSymbol = None
        self._timeFinalSymbol= None
        self._substitutionDictionary = {}
        self._integrationSymbols= []
        self._costateSymbols = []

    def RegisterConstantValue(self, symbol :sy.Expr, value : float) :
        """Registers a constant into the instance's substitution dictionary.

        Args:
            symbol (sy.Expr): Some sympy expression
            value (float): The value to substitution into various expressions.
        """
        self._substitutionDictionary[symbol] = value

    @property
    def SubstitutionDictionary(self) -> Dict[sy.Expr, float] :
        """The dictionary that should be used to store constant values that may appear 
        in the various expressions.  

        Returns:
            Dict[sy.Expr, float]: The expression-to-values to substitute into expressions.
        """
        return self._substitutionDictionary

    @staticmethod
    def CreateCoVector(y, name : str, t : sy.Symbol =None) :
        """Creates a co-vector for the entered y.

        Args:
            y: A sympy Symbol, Function, Matrix, ImmutableDenseMatrix.  Can also be a list of Symbols or Functions.
            name (str): The name to use for the co-vector variables.  The names of the original symbols will be appended as a subscript.
            t (sy.Symbol, optional): If y ought to be a sy.Function instead of a constant Symbol, this is the independent variable for that Function. Defaults to None.

        Returns:
            same type as y if supported: the costate value for the entered y
        """

        # this function is less ugly...
        if((y is sy.Symbol or y is sy.Function or (hasattr(y, "is_Function") and y.is_Symbol) or (hasattr(y, "is_Symbol") and y.is_Function)) and (not isinstance(y, sy.Matrix) and not isinstance(y, sy.ImmutableDenseMatrix))):
            if(t is None) :
                return sy.Symbol(name + '_{'+y.name+'}', real=True)
            else :
                return (sy.Function(name + '_{'+y.name+'}', real=True)(t))

        if(isinstance(y, list) ) :
            coVector = []
            for i in range(0, len(y)):
                coVector.append(SymbolicProblem.CreateCoVector(y[i], name, t))
            return coVector

        coVector = Vector.zeros(y.shape[0])
        for i in range(0, y.shape[0]):
            coVector[i, 0] = SymbolicProblem.CreateCoVector(y[i], name, t)
        return coVector

    def CreateHamiltonian(self, lambdas = None) -> sy.Expr:
        """Creates an expression for the Hamiltonian.

        Args:
            lambdas (optional): The costate variables. Defaults to None in which case 
            they will be created.

        Returns:
            sy.Expr: The Hamiltonian.
        """
        if(lambdas == None) :
            lambdas = SymbolicProblem.CreateCoVector(self.StateVariables, r'\lambda', self.TimeSymbol)

        if isinstance(lambdas, list) :
            lambdas = Vector.fromArray(lambdas)

        secTerm =  (lambdas.transpose()*self.EquationsOfMotionInMatrixForm())[0,0]
        return secTerm+self.UnIntegratedPathCost     

    def CreateHamiltonianControlExpressions(self, hamiltonian : sy.Expr) -> sy.Expr:
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
        return sy.Derivative(hamiltonian, u).doit()  

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
        return -1*sy.Derivative(hamiltonian, x).doit()

    def TransversalityConditionsByAugmentation(self, lambdas : List[sy.Expr], nus : List[sy.Symbol]) -> List[sy.Expr]:
        """Creates the transversality conditions by augmenting the terminal constraints to the terminal cost.

        Args:
            lambdas (List[sy.Expr]): The costate variables.
            nus (List[sy.Symbol]): The constant parameters to augment the constraints to the terminal cost with.

        Returns:
            List[sy.Expr]: The list of the final conditions that need to be solved to 0 for the solution to be optimal.
        """

        termFunc = self.TerminalCost.subs(self.TimeSymbol, self.TimeFinalSymbol) + (Vector.fromArray(nus).transpose()*Vector.fromArray(self.BoundaryConditions))[0,0]
        finalConditions = []
        i=0
        for x in self.StateVariables :
            if i >= len(lambdas) :
                break
            xf = x.subs(self.TimeSymbol, self.TimeFinalSymbol)
            cond = termFunc.diff(xf)
            finalConditions.append(lambdas[i]-cond)
            i=i+1

        return finalConditions

    def TransversalityConditionInTheDifferentialForm(self, hamiltonian : sy.Expr, lambdasFinal : List[sy.Expr], dtf) ->List[sy.Expr]:
        """Creates the differential form of the transversality condition.

        Args:
            hamiltonian (sy.Expr): The hamiltonian.
            lambdasFinal (List[sy.Expr]): The costate variables at the final time.
            dtf (_type_): If the final time is fixed, then this should be 0.  However, if the final time is not fixed then 
            this should be a sy.Expr for the final time (it can be as simple as sy.Symbol('dt_f').

        Returns:
            List[sy.Expr]: The transversality conditions.
        """
        variationVector = []
        valuesAtEndSymbols = []
        if isinstance(self.TerminalCost, float) :
            valuesAtEndDiffTerm = 0
        else :
            valuesAtEndDiffTerm = sy.diff(self.TerminalCost, self.TimeFinalSymbol).expand().simplify()

        # work on the dPsi term
        for bc in self.BoundaryConditions:
            expr = sy.diff(bc, self.TimeFinalSymbol).doit().powsimp().simplify()
            if(expr != None and expr != 0) :
                valuesAtEndSymbols.append(expr)

        # create the variation vector
        # If BC's are such that the final optimal value of the sv at tf is fixed (its derivative is a float)
        # then the variation vector must be 0 for that final value.
        # If however, BC's are such that the final value of the state variable can be different along 
        # the optimal trajectory, then the variation vector for it is the symbol d_stateValue/d_tf, which 
        # for a well posed problem will give us several equations we can use as additional BC's when 
        # we build the entire transversality condition equation and solve for coefficients to be 0
        finalSvs = self.SafeSubs(self.StateVariables, {self.TimeSymbol: self.TimeFinalSymbol})
        for sv in finalSvs :     
            if sv in lambdasFinal :
                continue  
            dxfdtf = sy.diff(sv, self.TimeFinalSymbol).doit() 
            notFixed = True # by default we should assume that state variables are not fixed
            for bc in self.BoundaryConditions :
                derVal = bc.diff(self.TimeFinalSymbol)/dxfdtf   
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
        lambdasFinal = Vector.fromArray(lambdasFinal)
        overallCond = hamiltonian*dtf - (lambdasFinal.transpose()*variationVector)[0,0] + valuesAtEndDiffTerm
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

        xversConds = []
        for dx in variationVector :
            coef = overallCond.coeff(dx)
            if(coef != 0.0) :
                xversConds.append(coef)    

        return xversConds  

    @property
    def StateVariables(self) -> List[sy.Symbol]:
        """Gets the state variables for this problem.  These should be in terms of TimeSymbol. 
        This must be implimented by the derived type.

        Returns:
            List[sy.Symbol]: The list of symbols in terms of TimeSymbol
        """
        return self._stateVariables
    
    @property
    def ControlVariables(self) -> List[sy.Symbol]:
        """Gets a list of the control variables.  These should be in terms of TimeSymbol. 
        This must be implimented by the derived type.

        Returns:
            List[sy.Symbol]: The list of the control variables.
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
    def set_TerminalCost(self, value : sy.Expr) :
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
    def set_UnIntegratedPathCost(self, value: sy.Expr) :
        """Sets the un-integrated path cost of the trajectory.  For a problem of Bolza, this is the expression in the integral.

        Args:
            value (sy.Expr): The un-integrated path cost.
        """
        self._unIntegratedPathCost = value

    @property
    def EquationsOfMotion(self) -> Dict[sy.Symbol, sy.Expr]:
        """Gets the equations of motion for each of the state variables.

        Returns:
            Dict[sy.Symbol, sy.Expr]: The equations of motion for each of the state variables.
        """
        return self._equationsOfMotion
    
    @property
    def BoundaryConditions(self) ->List[sy.Eq] :
        """Gets the boundary conditions on the system.  These expressions 
        must equal 0 and symbols in them need to be in terms of Time0Symbol 
        or TimeFinalSymbol as appropriate.

        Returns:
            List[sy.Eq]: The boundary conditions. 
        """
        return self._boundaryConditions

    @property
    def TimeSymbol(self) -> sy.Expr :
        """Gets the general time symbol.  Instead of using simple symbols for the state and 
        control variables, use sy.Function()(self.TimeSymbol) instead.

        Returns:
            sy.Expr: The time symbol.
        """        
        return self._timeSymbol

    @TimeSymbol.setter
    def set_TimeSymbol(self, value:sy.Expr) :
        """Sets the general time symbol.  Instead of using simple symbols for the state and 
        control variables, use sy.Function()(self.TimeSymbol) instead.

        Args:
            value (sy.Expr): The time symbol. 
        """
        self._timeSymbol = value

    @property
    def TimeInitialSymbol(self) -> sy.Expr :
        """Gets the symbol for the initial time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the start of the time span.

        Returns:
            sy.Expr: The initial time symbol.
        """        
        return self._timeInitialSymbol

    @TimeInitialSymbol.setter
    def set_TimeInitialSymbol(self, value:sy.Expr) :
        """Sets the symbol for the initial time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the start of the time span.

        Args:
            value (sy.Expr): The initial time symbol.
        """
        self._timeInitialSymbol = value

    @property
    def TimeFinalSymbol(self) -> sy.Expr :
        """Gets the symbol for the final time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the end of the time span.

        Returns:
            sy.Expr: The final time symbol.
        """        
        return self._timeFinalSymbol

    @TimeFinalSymbol.setter
    def set_TimeFinalSymbol(self, value : sy.Expr) :
        """Sets the symbol for the final time.  Note that boundary 
        conditions ought to use this as the independent variable 
        of sympy Functions for boundary conditions at the end of the time span.

        Args:
            value (sy.Expr): The final time symbol.
        """
        self._timeFinalSymbol = value

    @property
    def IntegrationSymbols(self) :
        return self._integrationSymbols

    @property
    def CostateSymbols(self) :
        return self._costateSymbols

    @property
    def Lambdas(self) :
        return self.CostateSymbols

    def CreateEquationOfMotionsAsEquations(self) -> List[sy.Expr] :
        """Converts the equations of motion dictionary into a list in the order of the state variables.

        Returns:
            List[sy.Expr]: The equations of motion in a list in the same order as the state variables.
        """
        eqs = []
        for sv in self.StateVariables :
            eqs.append(sy.Eq(sy.diff(sv, self.TimeSymbol).doit(), self.EquationsOfMotion[sv]))
        return eqs

    def CreateCostFunctionAsEquation(self, lhs : sy.Expr=None) -> sy.Eq :
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
        """Converts the equations of motion into a sy.Matrix.

        Returns:
            sy.Matrix: The equations of motion in matrix form.
        """
        tempArray = []
        for sv in self.StateVariables :
            tempArray.append(self.EquationsOfMotion[sv])
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

    def AddInitialValuesToDictionary(self, subsDict : Dict, initialValuesArray : List, lambdas : List =None):
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
        
        return SymbolicProblem.SafeSubs(thingWithSymbols, {self.TimeSymbol: self.TimeInitialSymbol, self.TimeFinalSymbol:self.TimeInitialSymbol})

    @staticmethod
    def SafeSubs(thingWithSymbols, substitutionDictionary : Dict) :
        """Safely substitute a dictionary into something with sympy expressions returning 
        the same type as thingsWithSymbols.

        Args:
            thingWithSymbols: Either a sympy Expression, or a List of expressions, or a sy.Matrix.  If this is a float, it will be returned
            substitutionDictionary (Dict): The dictionary of things to substitution into thingWithSymbols

        Raises:
            Exception: If this function doesn't know how to do the substitution, an exception will be thrown.

        Returns:
            (same type as thingWithSymbols) : thingWithSymbols substituted with substitutionDictionary
        """
        if isinstance(thingWithSymbols, Dict) :
            for (k,v) in thingWithSymbols.items() :
                thingWithSymbols[k] = SymbolicProblem.SafeSubs(v, substitutionDictionary)
            return

        if isinstance(thingWithSymbols, float) or isinstance(thingWithSymbols, int) or ((hasattr(thingWithSymbols, "is_Float") and thingWithSymbols.is_Float)):
            return thingWithSymbols # it's float, send it back

        if hasattr(thingWithSymbols, "subs") :
            if thingWithSymbols in substitutionDictionary :
                return substitutionDictionary[thingWithSymbols]
            return thingWithSymbols.subs(substitutionDictionary)
        
        if hasattr(thingWithSymbols, "__len__") :
            tbr = []
            for thing in thingWithSymbols :
                tbr.append(SymbolicProblem.SafeSubs(thing, substitutionDictionary))
            return tbr
        raise Exception("Don't know how to do the subs")

    def DescaleResults(self, resultsDictionary : Dict[sy.Symbol, List[float]], subsDict : Dict[sy.Symbol, float]) -> Dict[sy.Symbol, List[float]] :
        return resultsDictionary

    def EvaluateHamiltonianAndItsFirstTwoDerivatives(self, solution : Dict[sy.Expr, List[float]], tArray: List[float], hamiltonian : sy.Expr, controlSolved :Dict[sy.Expr, sy.Expr], moreSubs :Dict[sy.Expr, float]) ->List[List[float]]:
        """Evaluates the Hamiltonian and its first 2 derivatives.  This is useful to 
        see if the related conditions are truly satisfied.

        Args:
            solution (Dict[sy.Expr, List[float]]): The solution of the optimal control problem.
            tArray (List[float]): The time coorsiponding to the solution.
            hamiltonian (sy.Expr): The Hamiltonian expression.
            controlSolved ([sy.Expr, sy.Expr]): The Hamiltonian is likely in terms of the original control variable instead of the costate values.  If that is the case, this should be the expression of the control variables in terms of the costate variables.
            moreSubs (Dict[sy.Expr, float]): Any additional values to substitute into the expressions (if the final time was solved for, or if there were other parameters not included in the problems SubstitutionDictionary).

        Returns:
            List[List[float]]: The values of the Hamiltonian, its first derivative and second derivative for the entered solution.
        """

        stateForEom = [self.TimeSymbol, *self.IntegrationSymbols]

        constantsSubsDict = self.SubstitutionDictionary

        dHdu = self.CreateHamiltonianControlExpressions(hamiltonian).doit()[0]
        d2Hdu2 = sy.diff(hamiltonian, self.ControlVariables[0], 2)
        #d2Hdu2 =  self.CreateHamiltonianControlExpressions(dHdu).doit()[0]
        toEval = hamiltonian.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict)
        hamltEpx = sy.lambdify(stateForEom, toEval)
        solArray = []
        for sv in self.IntegrationSymbols :
            solArray.append(np.array(solution[sv]))

        hamltVals = hamltEpx(tArray, *solArray)
        dhduExp = sy.lambdify(stateForEom, dHdu.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict))
        
        dhduValus = dhduExp(tArray, *solArray)       
        if not hasattr(dhduValus, "__len__") or len(dhduValus) != len(hamltVals) :
            dhduValus = [dhduValus] * len(hamltVals)
        d2hdu2Exp = sy.lambdify(stateForEom, d2Hdu2.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict))
        d2hdu2Valus = d2hdu2Exp(tArray, *solArray)
        return [hamltVals, dhduValus, d2hdu2Valus]
     