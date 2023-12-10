import sympy as sy
from typing import List, Dict, Optional, cast, Collection
from collections import OrderedDict
from pyeq2orb.Symbolics.Vectors import Vector # type: ignore
from abc import ABC, abstractmethod
import numpy as np
from matplotlib.figure import Figure # type: ignore
from collections import OrderedDict
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs
import pyeq2orb
# it is likely that this class will get split up into a problem definition and an 
# indirect solver in the near future

"""Base class for optimization problems where the equations of motion and the 
boundary conditions are created with sympy.
"""
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
        self._substitutionDictionary = OrderedDict()
        self._costateSymbols = []

    def RegisterConstantValue(self, symbol :sy.Expr, value : float) :
        """Registers a constant into the instance's substitution dictionary.  Note that you can just 
        get the SubstitutionDictionary and add to it directly.

        Args:
            symbol (sy.Expr): Some sympy expression
            value (float): The value to substitution into various expressions.
        """
        self._substitutionDictionary[symbol] = value

    @property
    def SubstitutionDictionary(self) -> Dict[sy.Expr, float] :
        """The dictionary that should be used to store constant values that may appear 
        in the various expressions.  Many helper functions elsewhere want this dictionary 
        passed to it.

        Returns:
            Dict[sy.Expr, float]: The expression-to-values to substitute into expressions.
        """
        return self._substitutionDictionary

    @staticmethod
    def CreateCoVector(y, name : Optional[str] = None, t : Optional[sy.Symbol]=None) -> Vector:
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
                coVector.append(SymbolicProblem.CreateCoVector(y[i], name, t))
            return coVector

        coVector = Vector.zeros(y.shape[0])
        for i in range(0, y.shape[0]):
            coVector[i] = SymbolicProblem.CreateCoVector(y[i], name, t)
        return coVector

    @staticmethod
    def CreateHamiltonianStatic(stateVariables, t, eomsMatrix, unIntegratedPathCost, lambdas = None) -> sy.Expr:
        """Creates an expression for the Hamiltonian.

        Args:
            lambdas (optional): The costate variables. Defaults to None in which case 
            they will be created.

        Returns:
            sy.Expr: The Hamiltonian.
        """
        if(lambdas == None) :
            lambdas = SymbolicProblem.CreateCoVector(stateVariables, r'\lambda', t)

        if isinstance(lambdas, list) :
            lambdas = Vector.fromArray(lambdas)

        secTerm =  (lambdas.transpose()*eomsMatrix)[0,0]
        return secTerm+unIntegratedPathCost    

    def CreateHamiltonian(self, lambdas = None) -> sy.Expr:
        """Creates an expression for the Hamiltonian.

        Args:
            lambdas (optional): The costate variables. Defaults to None in which case 
            they will be created.

        Returns:
            sy.Expr: The Hamiltonian.
        """
        return SymbolicProblem.CreateHamiltonianStatic(self.StateVariables, self.TimeSymbol, self.EquationsOfMotionInMatrixForm(), self.UnIntegratedPathCost, lambdas)

    @staticmethod
    def CreateHamiltonianControlExpressionsStatic( hamiltonian : sy.Expr, controlVariables) -> sy.Matrix:
        return sy.Derivative(hamiltonian, controlVariables).doit()  

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
        return SymbolicProblem.CreateHamiltonianControlExpressionsStatic(hamiltonian, u)
        
    @staticmethod
    def CreateLambdaDotConditionStatic(hamiltonian, stateVariables) :
        return -1*sy.Derivative(hamiltonian, stateVariables).doit()
    
    @staticmethod
    def CreateLambdaDotEquationsStatic(hamiltonian : sy.Expr, t : sy.Symbol, stateVariables, lambdaSymbols) :
        rightHandSides = -1*sy.Derivative(hamiltonian, stateVariables).doit()
        eqs = []
        for i in range(0, len(lambdaSymbols)) :
            eqs.append(sy.Eq(lambdaSymbols[i].diff(t), rightHandSides[i]))
        return eqs

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
        return SymbolicProblem.CreateLambdaDotConditionStatic(hamiltonian, x)

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
    
    def TransversalityConditionsByAugmentation(self, nus : List[sy.Symbol], lambdasFinal : Optional[List[sy.Expr]] = None) -> List[sy.Expr]:
        """Creates the transversality conditions by augmenting the terminal constraints to the terminal cost.

        Args:
            nus (List[sy.Symbol]): The constant parameters to augment the constraints to the terminal cost with.
            lambdasFinal (List[sy.Symbol]): The costate symbols at the final time.  If None it will use the problems
            CostateSymbols at the final time, and if those are not set, then an exception will be raised.

        Returns:
            List[sy.Expr]: The list of transversality conditions, that ought to be treated like normal boundary conditions.
        """
        if lambdasFinal == None :
            if self.CostateSymbols != None and len(self.CostateSymbols) > 0:
                lambdasFinal = SafeSubs(self.CostateSymbols, {self.TimeSymbol: self.TimeFinalSymbol})
            else :
                raise Exception("No source of costate symbols.")
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

    def TransversalityConditionInTheDifferentialForm(self, hamiltonian : sy.Expr, dtf, lambdasFinal : Optional[List[sy.Expr]]= None) ->List[sy.Expr]:
        """Creates the transversality conditions by with the differential form of the transversality conditions. 

        Args:
            hamiltonian (sy.Expr): The hamiltonian in terms of the costate values (as opposed to the control variable)
            dtf (_type_): Either 0 if the final time is fixed, or a symbol indicating that the final time is not fixed.
            lambdasFinal (List[sy.Symbol]): The costate symbols at the final time.  If None it will use the problems
            CostateSymbols at the final time, and if those are not set, then an exception will be raised.

        Returns:
            List[sy.Expr]: The list of transversality conditions, that ought to be treated like normal boundary conditions.
        """
        if lambdasFinal == None :
            if self.CostateSymbols != None and len(self.CostateSymbols) > 0:
                lambdasFinal = SafeSubs(self.CostateSymbols, {self.TimeSymbol: self.TimeFinalSymbol})
            else :
                raise Exception("No source of costate symbols.")        
        lambdasFinal = cast(List[sy.Expr], lambdasFinal)
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
        finalSvs = SafeSubs(self.StateVariables, {self.TimeSymbol: self.TimeFinalSymbol})
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




    @property
    def IntegrationSymbols(self) -> List[sy.Expr]:
        """Gets the list of values that values that are going to be integrated by the equations of motion. 
        Calling code needs to manage the order of the EquationsOfMotion.

        Returns:
            List[sy.Symbol]: The values that will be integrated by the equations of motion.
        """
        return list(self.EquationsOfMotion.keys() )

    @property
    def CostateSymbols(self) :
        return self._costateSymbols

    @property
    def Lambdas(self) :
        return self.CostateSymbols


    @property
    def StateVariables(self) -> List[sy.Expr]:
        """Gets the state variables for this problem.  These should be in terms of TimeSymbol. 
        This must be implemented by the derived type.

        Returns:
            List[sy.Expr]: The list of symbols in terms of TimeSymbol
        """
        return self._stateVariables
    
    @property
    def ControlVariables(self) -> List[sy.Expr]:
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
    def EquationsOfMotion(self) -> Dict[sy.Expr, sy.Expr]:
        """Gets the equations of motion for each of the state variables.  This is an ordered dictionary,
        and the integration state is the keys of this ordered dict.

        Returns:
            Dict[sy.Expr, sy.Expr]: The ordered dictionary equations of motion for each of the state variables.
        """
        return self._equationsOfMotion
    
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



    def CreateEquationOfMotionsAsEquations(self) -> List[sy.Eq] :
        """Converts the equations of motion dictionary into a list in the order of the state variables.

        Returns:
            List[sy.Expr]: The equations of motion in a list in the same order as the state variables.
        """
        eqs = [] #type: List[sy.Eq]
        for sv in self.StateVariables :
            eqs.append(sy.Eq(sy.diff(sv, self.TimeSymbol).doit(), self.EquationsOfMotion[sv]))
        return eqs

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


    # @staticmethod
    # def SafeSubs(thingWithSymbols, substitutionDictionary : Dict) :
    #     """Safely substitute a dictionary into something with sympy expressions returning 
    #     the same type as thingsWithSymbols.

    #     Args:
    #         thingWithSymbols: Either a sympy Expression, or a List of expressions, or a sy.Matrix.  If this is a float, it will be returned
    #         substitutionDictionary (Dict): The dictionary of things to substitution into thingWithSymbols

    #     Raises:
    #         Exception: If this function doesn't know how to do the substitution, an exception will be thrown.

    #     Returns:
    #         (same type as thingWithSymbols) : thingWithSymbols substituted with substitutionDictionary
    #     """
    #     if isinstance(thingWithSymbols, Dict) :
    #         for (k,v) in thingWithSymbols.items() :
    #             thingWithSymbols[k] = SafeSubs(v, substitutionDictionary)
    #         return

    #     if isinstance(thingWithSymbols, float) or isinstance(thingWithSymbols, int) or ((hasattr(thingWithSymbols, "is_Float") and thingWithSymbols.is_Float)):
    #         return thingWithSymbols # it's float, send it back

    #     if hasattr(thingWithSymbols, "subs") :
    #         if thingWithSymbols in substitutionDictionary :
    #             return substitutionDictionary[thingWithSymbols]
    #         finalExp = thingWithSymbols
    #         finalExp = finalExp.subs(substitutionDictionary).doit(deep=True)
    #         # for k,v in substitutionDictionary.items() :
    #         #     finalExp = finalExp.subs(k, v).doit(deep=True) # this makes a difference?!?
    #         return finalExp
        
    #     if hasattr(thingWithSymbols, "__len__") :
    #         tbr = []
    #         for thing in thingWithSymbols :
    #             tbr.append(SafeSubs(thing, substitutionDictionary))
    #         return tbr
    #     raise Exception("Don't know how to do the subs")

    def DescaleResults(self, resultsDictionary : Dict[sy.Expr, List[float]]) -> Dict[sy.Expr, List[float]] :
        """Returns the resultsDictionary.  Although there is a derived type that has scaling factors that can be applied, making 
        this function on the base type helps switching back and forth between the scaled and unscaled problem.

        Args:
            resultsDictionary (Dict[sy.Symbol, List[float]]): The results of some run where the keys are the symbol and the list of floats are the 
            time history of that symbol.

        Returns:
            Dict[sy.Symbol, List[float]]: The same instance of the resultsDictionary
        """
        return resultsDictionary # the subsDict is included because the substitution factors might be symbols themselves.  By the time we have results those values ought to be in the substitution dictionary already

    def EvaluateHamiltonianAndItsFirstTwoDerivatives(self, solution : Dict[sy.Expr, List[float]], tArray: Collection[float], hamiltonian : sy.Expr, controlSolved :Dict[sy.Expr, sy.Expr], moreSubs :Dict[sy.Symbol, float]) ->List[List[float]]:
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

        stateForEom = [self.TimeSymbol, *self.IntegrationSymbols]

        constantsSubsDict = self.SubstitutionDictionary

        dHdu = self.CreateHamiltonianControlExpressions(hamiltonian).doit()[0]
        d2Hdu2 = sy.diff(hamiltonian, self.ControlVariables[0], 2)
        #d2Hdu2 =  self.CreateHamiltonianControlExpressions(dHdu).doit()[0] # another way to calculate it, but doesn't seem to be as good
        toEval = hamiltonian.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict)
        hamiltonianExpression = sy.lambdify(stateForEom, toEval)
        solArray = []
        for sv in self.IntegrationSymbols :
            solArray.append(np.array(solution[sv]))

        hamiltonianValues = hamiltonianExpression(tArray, *solArray)
        dhduExp = sy.lambdify(stateForEom, dHdu.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict))
        
        dhduValues = dhduExp(tArray, *solArray)       
        if not hasattr(dhduValues, "__len__") or len(dhduValues) != len(hamiltonianValues) :
            dhduValues = [dhduValues] * len(hamiltonianValues)
        d2hdu2Exp = sy.lambdify(stateForEom, d2Hdu2.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict))
        d2hdu2Values = d2hdu2Exp(tArray, *solArray)
        return [hamiltonianValues, dhduValues, d2hdu2Values]

    @abstractmethod
    def AddStandardResultsToFigure(self, figure : Figure, t : List[float], dictionaryOfValueArraysKeyedOffState : Dict[sy.Expr, List[float]], label : str) -> None:
        """Adds the contents of dictionaryOfValueArraysKeyedOffState to the plot.

        Args:
            figure (matplotlib.figure.Figure): The figure the data is getting added to.
            t (List[float]): The time corresponding to the data in dictionaryOfValueArraysKeyedOffState.
            dictionaryOfValueArraysKeyedOffState (Dict[sy.Expr, List[float]]): The data to get added.  The keys must match the values in self.State and self.Control.
            label (str): A label for the data to use in the plot legend.
        """
        pass

    @property
    def EquationsOfMotionAsEquations(self) -> List[sy.Eq] :
        """The equations of motions as symbolic equations where the LHS is the state variable differentiated by time 
        and the RHS the expression from EquationsOfMotion.  Expect renames in short order to make this the primary property

        Returns:
            List[sy.Eq]: The equations of motion as symbolic equations,
        """
        equationsOfMotion = [] #type: List[sy.Eq]

        for i in range(0, len(self.StateVariables)) :
            equationsOfMotion.append(sy.Eq(self.StateVariables[i].diff(self.TimeSymbol), self.EquationsOfMotion[self.StateVariables[i]]))

        for i in range(0, len(self.CostateSymbols)) :
            equationsOfMotion.append(sy.Eq(self.CostateSymbols[i].diff(self.TimeSymbol), self.EquationsOfMotion[self.CostateSymbols[i]]))            
        return equationsOfMotion             