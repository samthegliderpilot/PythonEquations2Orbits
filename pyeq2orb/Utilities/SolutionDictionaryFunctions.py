from typing import Dict, List, OrderedDict
from pyeq2orb.Utilities.Typing import SymbolOrNumber
import sympy as sy
def GetValueFromStateDictionaryAtIndex(fullSolutionDictionary : Dict[sy.Expr, List[float]], index : int) ->Dict[sy.Expr, float] :
    """From a dictionary containing lists of something (often the evaluated floats), get a 
    similar dictionary that is just the final values.

    Args:
        fullSolutionDictionary (Dict[sy.Expr, List[object]]): The full solution.

    Returns:
        Dict[sy.Expr, List[float]]: The final values from the solution.
    """    
    finalValues = OrderedDict()
    for (key, value) in fullSolutionDictionary.items() :
        finalValues[key] = value[index]
    return finalValues 

def GetInitialStateDictionary(fullSolutionDictionary : Dict[sy.Expr, List[float]]) ->Dict[sy.Expr, float] :
    """From a dictionary containing lists of something (often the evaluated floats), get a 
    similar dictionary that is the initial values.

    Args:
        fullSolutionDictionary (Dict[sy.Expr, List[float]]): The full solution.

    Returns:
        Dict[sy.Expr, List[float]]: The initial values from the solution.
    """
    return GetValueFromStateDictionaryAtIndex(fullSolutionDictionary, 0)

def GetFinalStateDictionary(fullSolutionDictionary : Dict[sy.Expr, List[float]]) ->Dict[sy.Expr, float] :
    """From a dictionary containing lists of something (often the evaluated floats), get a 
    similar dictionary that is the final values.

    Args:
        fullSolutionDictionary (Dict[sy.Expr, List[object]]): The full solution.

    Returns:
        Dict[sy.Expr, List[Float]]: The final values from the solution.
    """    
    return GetValueFromStateDictionaryAtIndex(fullSolutionDictionary, -1)

    