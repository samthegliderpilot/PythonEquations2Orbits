from collections import OrderedDict
from typing import List, Dict,Any
import sympy as sy
#from pyeq2orb.SymbolicOptimizerProblem import SymbolicProblem

def ConvertOdeIntResultsToDictionary(odeintSymbolicState : List[sy.Symbol], odeintResults : List[Any]) ->Dict[sy.Symbol, List[float]]:
    """Converts the results from an odeint call into a dictionary mapping the symbolic expressions to lists of floats.

    Args:
        odeintSymbolicState (List[sy.Expr]): The symbolic state expressions of the independent variables that were integrated
        odeintResults (List): The results from the odeint call.

    Returns:
        Dict[sy.Expr, List[float]]: The mapping of the symbolic state variables to the list of the results for that variable.
    """
    asDict = OrderedDict() #type: Dict[sy.Symbol, List[float]]
    i = 0
    if len(odeintResults[0]) != len(odeintResults[1])  : # this is not a good check for if full_output was true or not, but it is good enough in most cases
        odeintResults = odeintResults[0]
    for sv in odeintSymbolicState :
        asDict[sv] = odeintResults[:,i] #type: ignore
        i=i+1
    return asDict

def ConvertSolveIvpResultsToDictionary(integrationState : List[sy.Symbol], solveIvpResults) ->Dict[sy.Symbol, List[float]]:
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
        asDict[integrationState[sv]] = list(solveIvpResults.y[i])
        i=i+1
    return asDict

def ConvertEitherIntegratorResultsToDictionary(integrationState : List[sy.Symbol], integratorResults) ->Dict[sy.Symbol, List[float]]:
    """Converts either an odeint results or a solve_ivp results to a dictionary of history of values keyed off of the 
    passed in integration values.

    Args:
        integrationState (List[sy.Expr]): The integration symbols.
        integratorResults (_type_): The results from an odeint or solve_ivp run.

    Returns:
        Dict[sy.Expr, List[float]]: The results dictionary.
    """
    if hasattr(integratorResults, "y") :
        return ConvertSolveIvpResultsToDictionary(integrationState, integratorResults)
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

