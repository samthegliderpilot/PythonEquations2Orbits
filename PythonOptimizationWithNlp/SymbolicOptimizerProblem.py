from inspect import BoundArguments
import sympy as sy
from typing import List, Dict
from collections import OrderedDict
from PythonOptimizationWithNlp.Symbolics.Vectors import Vector
from abc import abstractmethod, ABC

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

    def TransversalityConditionsByAugmentation(self, lambdas, nus) :
        termFunc = self.TerminalCost.subs(self.TimeSymbol, self.TimeFinalSymbol) + (Vector.fromArray(nus).transpose()*Vector.fromArray(self.BoundaryConditions))[0,0]
        finalConditions = []
        i=0
        for x in self.StateVariables :
            xf = x.subs(self.TimeSymbol, self.TimeFinalSymbol)
            cond = termFunc.diff(xf)
            finalConditions.append(lambdas[i]-cond)
            i=i+1

        return finalConditions

    def CreateDifferentialTransversalityConditions(self, hamiltonian, lambdasFinal, dtf) :
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
    
    # @StateVariables.setter
    # def set_stateVariable(self, stateVariables : List[sy.Symbol]) :
    #     """Sets the state variables for this problem.  Note that this will swap out the entire 
    #     list of variables.  These should by sympy expressions of time if appropriate.

    #     Args:
    #         stateVariables (List[sy.Symbol]): The new set of state variables
    #     """
    #     self._stateVariables = stateVariables

    @property
    def ControlVariables(self) -> List[sy.Symbol]:
        """Gets a list of the control variables.  These should be in terms of TimeSymbol. 
        This must be implimented by the derived type.

        Returns:
            List[sy.Symbol]: The list of the control variables.
        """
        return self._controlVariables

    # @ControlVariables.setter
    # def set_ControlVariables(self, value : List[sy.Symbol]) :
    #     """Sets the control variables.  This will swap out the entire list of variables.  These 
    #     should by sympy expressions of time if appropriate.

    #     Args:
    #         value (List[sy.Symbol]): The new list of control variables.
    #     """
    #     self._controlVariables = value

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
    
    # @EquationsOfMotion.setter
    # def set_EquationsOfMotion(self, value: Dict[sy.Symbol, sy.Expr]):
    #     """Sets the entire equations of motion dictionary.

    #     Args:
    #         value (Dict[sy.Symbol, sy.Expr]): The equations of motion for each of the state variables.
    #     """
    #     self._equationsOfMotion = value

    @property
    def BoundaryConditions(self) ->List[sy.Eq] :
        """Gets the boundary conditions on the system.  These expressions 
        must equal 0 and symbols in them need to be in terms of Time0Symbol 
        or TimeFinalSymbol as appropriate.

        Returns:
            List[sy.Eq]: The boundary conditions. 
        """
        return self._boundaryConditions

    # @BoundaryConditions.setter
    # def set_BoundaryConditions(self, value :List[sy.Eq]) :
    #     """Sets the boundary conditions on the system.  These expressions 
    #     must equal 0 and symbols in them need to be in terms of Time0Symbol 
    #     or TimeFinalSymbol as appropriate.

    #     Args:
    #         value (List[sy.Eq]): The boundary conditions.
    #     """
    #     self._boundaryConditions = value

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

    def CreateVariablesAtTime0(self, listToSubs = None) :
        if listToSubs == None :
            listToSubs = self.StateVariables

        return SymbolicProblem.SafeSubs(listToSubs, {self.TimeSymbol: self.TimeInitialSymbol, self.TimeFinalSymbol:self.TimeInitialSymbol})

    @staticmethod
    def SafeSubs(thingWithSymbols, substitutionDictionary : Dict) :
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