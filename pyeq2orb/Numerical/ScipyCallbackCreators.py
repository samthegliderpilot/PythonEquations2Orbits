from collections import OrderedDict
from typing import List, Dict
import sympy as sy

from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem



# def CreateSimpleCallbackForSolveIvp(timeSymbol : sy.Expr, integrationVariableSymbols : List[sy.Expr], equationsOfMotion : Dict[sy.Expr, sy.Expr], substitutionDictionary : Dict[sy.Expr, float], otherArgs: List[sy.Expr]= None) : 
#     """Creates a callback to evaluate equations of motion with scipy.solveIvp.

#     Args:
#         timeSymbol (sy.Expr): The time symbol.
#         integrationVariableSymbols (List[sy.Expr]): The integration state variable.s
#         equationsOfMotion (Dict[sy.Expr, sy.Expr]): The equations of motion.
#         substitutionDictionary (Dict[sy.Expr, float]): Constants that ought to be substituted into the equations of motion ahead of time
#         otherArgs (List[sy.Expr], optional): Symbols of other constants to be passed in as args to the callback.. Defaults to None.

#     Returns:
#         _type_: A callback taking in (time, state, args)
#     """
#     valuesArray = [timeSymbol, integrationVariableSymbols]
#     if otherArgs != None :
#         valuesArray.append(otherArgs)

#     eomList = []
#     for sv in integrationVariableSymbols :
#         thisEom = equationsOfMotion[sv].subs(substitutionDictionary)        
#         eomList.append(thisEom)   
#     eomCallback = sy.lambdify(valuesArray, eomList)

#     def callbackFunc(t, y, *args) :
#         return eomCallback(t, y, args)
#     return callbackFunc

# def CreateSimpleCallbackForOdeint(timeSymbol : sy.Expr, integrationVariableSymbols : List[sy.Expr], equationsOfMotion : Dict[sy.Expr, sy.Expr], substitutionDictionary : Dict[sy.Expr, float], otherArgs: List[sy.Expr]= None) : 
#     """Creates a callback to evaluate equations of motion with scipy.odeint.

#     Args:
#         timeSymbol (sy.Expr): The time symbol.
#         integrationVariableSymbols (List[sy.Expr]): The integration state variable.s
#         equationsOfMotion (Dict[sy.Expr, sy.Expr]): The equations of motion.
#         substitutionDictionary (Dict[sy.Expr, float]): Constants that ought to be subsitituted into the equations of motion ahead of time
#         otherArgs (List[sy.Expr], optional): Symbols of other constants to be passed in as args to the callback.. Defaults to None.

#     Returns:
#         _type_: A callback taking in (state, time, args)
#     """
#     valuesArray = [timeSymbol, integrationVariableSymbols]
#     if otherArgs != None :
#         valuesArray.append(otherArgs)

#     eomList = []
#     for sv in integrationVariableSymbols :
#         thisEom = equationsOfMotion[sv].subs(substitutionDictionary)        
#         eomList.append(thisEom)   
#         sy.lambdify(valuesArray, thisEom)
#     eomCallback = sy.lambdify(valuesArray, eomList)

#     def callbackFunc(y, t, *args) :
#         return eomCallback(t, y, args)
#     return callbackFunc



def ConvertOdeIntResultsToDictionary(odeintSymbolicState : List[sy.Expr], odeintResults : List) ->Dict[sy.Expr, List[float]]:
    """Converts the results from an odeint call into a dictionary mapping the symbolic expressions to lists of floats.

    Args:
        odeintSymbolicState (List[sy.Expr]): The symbolic state expressions of the independent variables that were integrated
        odeintResults (List): The results from the odeint call.

    Returns:
        Dict[sy.Expr, List[float]]: The mapping of the symbolic state variables to the list of the results for that variable.
    """
    asDict = OrderedDict()
    i = 0
    if len(odeintResults[0]) != len(odeintResults[1])  : # this is not a good check for if full_output was true or not, but it is good enough in most cases
        odeintResults = odeintResults[0]
    for sv in odeintSymbolicState :
        asDict[sv] = odeintResults[:,i]
        i=i+1
    return asDict

def ConvertSolveIvptResultsToDictionary(integrationState : List[sy.Expr], solveIvpResults) ->Dict[sy.Expr, List[float]]:
    """Converts the results from an odeint call into a dictionary mapping the symbolic expressions to lists of floats.

    Args:
        integrationState (List[sy.Expr]): The symbolic state expressions of the independent variables that were integrated
        solveIvpResults: The results from the solve_ivp call (the whole object).

    Returns:
        Dict[sy.Expr, List[float]]: The mapping of the symbolic state variables to the list of the results for that variable.
    """
    asDict = OrderedDict()
    i = 0
    for sv in range(0,len(integrationState)) :
        asDict[integrationState[sv]] = solveIvpResults.y[i]
        i=i+1
    return asDict

def ConvertEitherIntegratorResultsToDictionary(integrationState : List[sy.Expr], integratorResults) ->Dict[sy.Expr, List[float]]:
    """Converts either an odeint results or a solve_ivp results to a dictionary of history of values keyed off of the 
    passed in integration values.

    Args:
        integrationState (List[sy.Expr]): The integration symbols.
        integratorResults (_type_): The results from an odeint or solve_ivp run.

    Returns:
        Dict[sy.Expr, List[float]]: The results dictionary.
    """
    if hasattr(integratorResults, "y") :
        return ConvertSolveIvptResultsToDictionary(integrationState, integratorResults)
    else :
        return ConvertOdeIntResultsToDictionary(integrationState, integratorResults)

def GetInitialStateFromIntegratorResults(integratorResults) -> List[float] :
    """Gets the initial state from the integrator results, regardless if it was evaluated with odeint or solve_ivp

    Args:
        integratorResults: The results from solve_ivp or odeint

    Returns:
        List[float]: The initial values of the results.
    """
    if hasattr(integratorResults, "y") :
        return [y[0] for y in integratorResults.y]
    else :
        if len(integratorResults[0]) != len(integratorResults[1])  : # this is not a good check for if full_output was true or not, but it is good enough in most cases
            integratorResults = integratorResults[0]
        return integratorResults[0]

def GetFinalStateFromIntegratorResults(integratorResults) -> List[float] :
    """Gets the final state from the integrator results, regardless if it was evaluated with odeint or solve_ivp

    Args:
        integratorResults: The results from solve_ivp or odeint

    Returns:
        List[float]: The final values of the results.
    """
    if hasattr(integratorResults, "y") :
        return [y[-1] for y in integratorResults.y]
    else :
        if len(integratorResults[0]) != len(integratorResults[1])  : # this is not a good check for if full_output was true or not, but it is good enough in most cases
            integratorResults = integratorResults[0]        
        return integratorResults[-1]        