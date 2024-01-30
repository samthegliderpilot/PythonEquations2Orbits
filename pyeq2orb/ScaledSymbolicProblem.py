from matplotlib.figure import Figure # type: ignore
import sympy as sy
from typing import List, Dict, Optional, cast
import pyeq2orb
from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem
from pyeq2orb.Utilities.Typing import SymbolOrNumber
from pyeq2orb.Symbolics.SymbolicUtilities import SafeSubs
import numpy as np

"""A Symbolic problem that has scaling factors over another problem.  Those 
factors can be constants, or symbols themselves that are in the substitution 
dictionary that get used by various solvers.
"""
class ScaledSymbolicProblem(SymbolicProblem) :
    def __init__(self, wrappedProblem : SymbolicProblem, newStateVariableSymbols : List[sy.Symbol], valuesToDivideStateVariablesWith : Dict[sy.Expr, SymbolOrNumber], scaleTime : bool) :        
        """Initializes a new instance.

        Args:
            wrappedProblem (SymbolicProblem): The problem to scale.
            newStateVariableSymbols (Dict): The new state variables.  Often these are the same ones with some accent on them.  These should still be in terms 
            the wrappedProblems time symbol. All of the state variables must be present (and can be scaled by 1 if they are already well scaled).
            valuesToDivideStateVariablesWith (Dict): The scaling factors on the state variables. These can be floats, or symbols of some sort that 
            end up being constants in the SubstitutionDictionary
            scaleTime (bool): Should the time be scaled to be between 0 and 1.
        """
        super().__init__()


    # def DescaleResults(self, resultsDictionary : Dict[sy.Expr, List[float]]) -> Dict[sy.Expr, List[float]] :
    #     """After evaluating the problem numerically, descale the results to be back in terms of the original units.

    #     Args:
    #         resultsDictionary (Dict[sy.Expr, List[float]]): The results dictionary.

    #     Returns:
    #         Dict[sy.Expr, List[float]]: A new dictionary where the values are descaled AND the keys are the wrappedProblems's 
    #         state variables.
    #     """
    #     returnDict = {} #type: Dict[sy.Expr, List[float]]
    #     counter = 0
    #     for key, value in resultsDictionary.items() :
    #         sv = key
    #         if sv in self.StateVariables and counter < len(self.StateVariables):
    #             originalSv = self.StateVariables[self.StateVariables.index(sv)]
    #             convertedArray = np.array(value, copy=True)* SafeSubs(self.ScalingValues[originalSv], self.SubstitutionDictionary)
    #             returnDict[originalSv] = convertedArray
    #             counter = counter+1
    #         else :
    #             returnDict[key]=value
    #     return returnDict
    
    # @staticmethod
    # def CreateBarVariables(orgVariables : List[sy.Expr], timeSymbol :sy.Symbol) ->List[sy.Expr] :
    #     """A helper function to make a 

    #     Args:
    #         orgVariables (List[sy.Expr]): _description_
    #         timeSymbol (sy.Expr): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     baredVariables = [] #type : List[sy.Expr]
    #     for var in orgVariables :
    #         name = var.__getattribute__('name')            
    #         baredVariables.append(sy.Function(r'\bar{' + name+ '}')(timeSymbol))
    #     return baredVariables

    # def ScaleExpressions(self, expressions : List[sy.Expr]) -> List[sy.Expr]:
    #     """For some expression (or list of expressions), 

    #     Args:
    #         expressions (List[sy.Expr]): The expressions to scale.

    #     Returns:
    #         List[sy.Expr]: The scaled expressions
    #     """
    #     simpleSubsDict={} # for terminal cost
    #     counter=0
    #     for sv in self.WrappedProblem.StateVariables :
    #         oldSv = sv
    #         sv = self.StateVariables[counter]
    #         simpleSubsDict[oldSv] = sv*self.ScalingValues[oldSv]
    #         simpleSubsDict[oldSv.subs(self.WrappedProblem.TimeSymbol, self.WrappedProblem.TimeFinalSymbol)] = sv.subs(self.TimeSymbol, self.TimeFinalSymbol)*self.ScalingValues[oldSv]
    #         counter=counter+1

    #     return SafeSubs(expressions, simpleSubsDict)

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
    
    # def TransversalityConditionInTheDifferentialForm(self, hamiltonian : sy.Expr, dtf, lambdasFinal : Optional[List[sy.Symbol]]=None) ->List[sy.Expr]:
    #     """Creates the transversality conditions by with the differential form of the transversality conditions. Note that 
    #     this calls the wrapped problems TransversalityConditionsByAugmentation and then scales that expression.

    #     Args:
    #         hamiltonian (sy.Expr): The hamiltonian in terms of the costate values (as opposed to the control variable)
    #         dtf (_type_): Either 0 if the final time is fixed, or a symbol indicating that the final time is not fixed.
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

    #     finalConditions = self._wrappedProblem.TransversalityConditionInTheDifferentialForm(hamiltonian, dtf, lambdasFinal)
    #     return self.ScaleExpressions(finalConditions)

