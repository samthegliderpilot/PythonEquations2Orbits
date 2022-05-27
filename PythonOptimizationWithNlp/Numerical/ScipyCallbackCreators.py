from collections import OrderedDict
from typing import List, Dict
import sympy as sy

from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem

def CreateSimpleCallbackForSolveIvp(timeSymbol : sy.Expr, integrationVariableSymbols : List[sy.Expr], equationsOfMotion : Dict[sy.Expr, sy.Expr], substitutionDictionary : Dict[sy.Expr, float], otherArgs: List[sy.Expr]= None) : 
    """Creates a callback to evaluate equations of motion with scipy.solveIvp.

    Args:
        timeSymbol (sy.Expr): The time symbol.
        integrationVariableSymbols (List[sy.Expr]): The integration state variable.s
        equationsOfMotion (Dict[sy.Expr, sy.Expr]): The equations of motion.
        substitutionDictionary (Dict[sy.Expr, float]): Constants that ought to be subsitituted into the equations of motion ahead of time
        otherArgs (List[sy.Expr], optional): Symbols of other constants to be passed in as args to the callback.. Defaults to None.

    Returns:
        _type_: A callback taking in (time, state, args)
    """
    valuesArray = [timeSymbol, integrationVariableSymbols]
    if otherArgs != None :
        valuesArray.append(otherArgs)

    eomList = []
    for sv in integrationVariableSymbols :
        thisEom = equationsOfMotion[sv].subs(substitutionDictionary)        
        eomList.append(thisEom)   
        sy.lambdify(valuesArray, thisEom)
    eomCallback = sy.lambdify(valuesArray, eomList)

    def callbackFunc(t, y, *args) :
        return eomCallback(t, y, args)
    return callbackFunc

def CreateSimpleCallbackForOdeint(timeSymbol : sy.Expr, integrationVariableSymbols : List[sy.Expr], equationsOfMotion : Dict[sy.Expr, sy.Expr], substitutionDictionary : Dict[sy.Expr, float], otherArgs: List[sy.Expr]= None) : 
    """Creates a callback to evaluate equations of motion with scipy.odeint.

    Args:
        timeSymbol (sy.Expr): The time symbol.
        integrationVariableSymbols (List[sy.Expr]): The integration state variable.s
        equationsOfMotion (Dict[sy.Expr, sy.Expr]): The equations of motion.
        substitutionDictionary (Dict[sy.Expr, float]): Constants that ought to be subsitituted into the equations of motion ahead of time
        otherArgs (List[sy.Expr], optional): Symbols of other constants to be passed in as args to the callback.. Defaults to None.

    Returns:
        _type_: A callback taking in (state, time, args)
    """
    valuesArray = [timeSymbol, integrationVariableSymbols]
    if otherArgs != None :
        valuesArray.append(otherArgs)

    eomList = []
    for sv in integrationVariableSymbols :
        thisEom = equationsOfMotion[sv].subs(substitutionDictionary)        
        eomList.append(thisEom)   
        sy.lambdify(valuesArray, thisEom)
    eomCallback = sy.lambdify(valuesArray, eomList)

    def callbackFunc(y, t, *args) :
        return eomCallback(t, y, args)
    return callbackFunc

def CreateLambdifiedExpressions(stateExpressionList : List[sy.Expr], expressionsToLambdify : List[sy.Expr], constantsSubstitutionDictionary : Dict[sy.Expr, float]) ->sy.Expr :
    """ A helper function to create a lambdified callback of some expressions while also substituting in constant values into the expressions. One common problem that 
    might come up is if the constantsSubstitutionDictionary contains an independent variable of one of the symbols in the state (for example, if one of your state 
    variables is x(t) and you put a constant value of t into the constantsSubstitutionDictionary, this turns x(t) into x(2) and things get confusing quickly). Generally
    you shouldn't really want to do this and should re-structure your code to avoid this.

    Args:
        boundaryConditionState (List[sy.Expr]): The state (independent variables) for the returned lambdified expressions.
        expressionsToLambdify (List[sy.Expr]): The expressions to labmdify
        constantsSubstitutionDictionary (Dict[sy.Expr, float]): Constant values to bake into the expressionsToLambdify ahead of time

    Returns:
        _type_: A callback that numerically evaluates the expressionsToLambdify.
    """
    bcs = []
    for exp in expressionsToLambdify :
        bc = SymbolicProblem.SafeSubs(exp, constantsSubstitutionDictionary)
        bcs.append(bc)
    return sy.lambdify(stateExpressionList, bcs)    

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
    if hasattr(integratorResults, "y") :
        return ConvertSolveIvptResultsToDictionary(integrationState, integratorResults)
    else :
        return ConvertOdeIntResultsToDictionary(integrationState, integratorResults)

def GetInitialStateFromIntegratorResults(integratorResults) -> List[float] :
    if hasattr(integratorResults, "y") :
        return [y[0] for y in integratorResults.y]
    else :
        integratorResults[0]

def GetFinalStateFromIntegratorResults(integratorResults) -> List[float] :
    if hasattr(integratorResults, "y") :
        return [y[-1] for y in integratorResults.y]
    else :
        integratorResults[-1]        