import sympy as sy
from typing import List, Dict, Optional, cast, Collection
from collections import OrderedDict
from pyeq2orb.Symbolics.Vectors import Vector # type: ignore
from abc import ABC, abstractmethod
import numpy as np
from matplotlib.figure import Figure # type: ignore
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs
import pyeq2orb
from pyeq2orb.ProblemBase import Problem
from enum import Enum

class TransversalityConditionType(Enum):
    Differential = 1
    Adjoined = 2

"""Class for optimization problems where the equations of motion and the 
boundary conditions are created with sympy.
"""
class SymbolicProblem(Problem) :
    def __init__(self) :
        """Initialize a new instance. 
        """
        super().__init__()
        self._costateSymbols = []

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

    @staticmethod
    def CreateHamiltonianControlExpressionsStatic( hamiltonian : sy.Expr, controlVariables) -> sy.Matrix:
        return sy.Derivative(hamiltonian, controlVariables).doit()  

    @staticmethod
    def CreateControlExpressionsFromHamiltonian(hamiltonian : sy.Expr, controlVariables: List[sy.Symbol]) -> Dict[sy.Symbol, sy.Expr] :
        ans = {}
        for control in controlVariables:
            controlExpression = SymbolicProblem.CreateHamiltonianControlExpressionsStatic(hamiltonian, control)
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
        # If BC's are such that the final optimal value of the sv at tf is fixed (its derivative is a float)
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
        rightHandSides = -1*sy.Derivative(hamiltonian, stateVariables).doit()
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
        return SymbolicProblem.CreateHamiltonianStatic(self.StateVariables, self.TimeSymbol, self.EquationsOfMotionInMatrixForm(), self.UnIntegratedPathCost, lambdas)

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

    def TransversalityConditionsByAugmentation(self, nus : List[sy.Symbol], lambdasFinal : Optional[List[sy.Expr]] = None) -> List[sy.Expr]:
        """Creates the transversality conditions by augmenting the terminal constraints to the terminal cost.

        Args:
            nus (List[sy.Symbol]): The constant parameters to augment the constraints to the terminal cost with.
            lambdasFinal (List[sy.Symbol]): The costate symbols at the final time.  If None it will use the problems
            CostateSymbols at the final time, and if those are not set, then an exception will be raised.

        Returns:
            List[sy.Expr]: The list of transversality conditions, that ought to be treated like normal boundary conditions. It is assumed that these 
            expressions should be solved such that they equal 0
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

    def TransversalityConditionInTheDifferentialForm(self, hamiltonian : sy.Expr, dtf, lambdasFinal : Optional[List[sy.Symbol]]= None) ->List[sy.Expr]:
        problemToWorkWith = self
        if self._wrappedProblem != None :
            problemToWorkWith = self._wrappedProblem
        xvers = SymbolicProblem.TransversalityConditionInTheDifferentialFormStatic(hamiltonian, dtf, problemToWorkWith.ControlVariables, problemToWorkWith.TerminalCost, problemToWorkWith.TimeFinalSymbol, problemToWorkWith.BoundaryConditions, SafeSubs(problemToWorkWith.StateVariables, {problemToWorkWith.TimeSymbol: problemToWorkWith.TimeFinalSymbol}))
        if self._wrappedProblem != None :
            xvers = self.ScaleExpressions(xvers)
        return xvers

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

    @property
    def CostateSymbols(self) :
        return self._costateSymbols

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

        stateForEom = [self.TimeSymbol, *self.StateVariables]

        constantsSubsDict = self.SubstitutionDictionary

        dHdu = self.CreateHamiltonianControlExpressions(hamiltonian).doit()[0]
        d2Hdu2 = sy.diff(hamiltonian, self.ControlVariables[0], 2)
        #d2Hdu2 =  self.CreateHamiltonianControlExpressions(dHdu).doit()[0] # another way to calculate it, but doesn't seem to be as good
        toEval = hamiltonian.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict)
        hamiltonianExpression = sy.lambdify(stateForEom, toEval)
        solArray = []
        for sv in self.StateVariables :
            solArray.append(np.array(solution[sv]))

        hamiltonianValues = hamiltonianExpression(tArray, *solArray)
        dhduExp = sy.lambdify(stateForEom, dHdu.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict))
        
        dhduValues = dhduExp(tArray, *solArray)       
        if not hasattr(dhduValues, "__len__") or len(dhduValues) != len(hamiltonianValues) :
            dhduValues = [dhduValues] * len(hamiltonianValues)
        d2hdu2Exp = sy.lambdify(stateForEom, d2Hdu2.subs(controlSolved).subs(moreSubs).trigsimp(deep=True).subs(constantsSubsDict))
        d2hdu2Values = d2hdu2Exp(tArray, *solArray)
        return [hamiltonianValues, dhduValues, d2hdu2Values]

#TODO: Refactor the other xversality to have a static version...
    # def TransversalityConditionsByAugmentation(self, nus : List[sy.Symbol], lambdasFinal : Optional[List[sy.Expr]]=None) -> List[sy.Expr]:
    #     """Creates the transversality conditions by augmenting the terminal constraints to the terminal cost. Note that 
    #     this calls the wrapped problems TransversalityConditionsByAugmentation and then scales that expression.

    #     Args:
    #         nus (List[sy.Symbol]): The constant parameters to augment the constraints to the terminal cost with.
    #         lambdasFinal (List[sy.Symbol]): The costate symbols at the final time.  If None it will use the problems
    #         CostateSymbols at the final time, and if those are not set, then an exception will be raised.

    #     Returns:
    #         List[sy.Expr]: The list of transversality conditions, that ought to be treated like normal boundary conditions.
    #     """
    #     if lambdasFinal == None :
    #         if self.CostateSymbols != None and len(self.CostateSymbols) > 0:
    #             lambdasFinal = SafeSubs(self.CostateSymbols, {self.TimeSymbol: self.TimeFinalSymbol})
    #         else :
    #             raise Exception("No source of costate symbols.") 

    #     finalConditions = self._wrappedProblem.TransversalityConditionsByAugmentation(nus, lambdasFinal)
    #     return self.ScaleExpressions(finalConditions)
    
