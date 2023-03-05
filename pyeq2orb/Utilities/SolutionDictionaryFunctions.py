from typing import Dict, List, OrderedDict

def GetValueFromStateDictionaryAtIndex(fullSolutionDictionary : Dict[object, List[float]], index : int) ->Dict[object, float] :
    """From a dictionary containing lists of something (often the evaluated floats), get a 
    similar dictionary that is the final values.

    Args:
        fullSolutionDictionary (Dict[object, List[object]]): The full solution.

    Returns:
        Dict[object, object]: The final values from the solution.
    """    
    finalValues = OrderedDict()
    for (key, value) in fullSolutionDictionary.items() :
        finalValues[key] = value[index]
    return finalValues 

def GetInitialStateDictionary(fullSolutionDictionary : Dict[object, List[float]]) ->Dict[object, float] :
    """From a dictionary containing lists of something (often the evaluated floats), get a 
    similar dictionary that is the initial values.

    Args:
        fullSolutionDictionary (Dict[object, List[object]]): The full solution.

    Returns:
        Dict[object, object]: The initial values from the solution.
    """
    return GetValueFromStateDictionaryAtIndex(fullSolutionDictionary, 0)

def GetFinalStateDictionary(fullSolutionDictionary : Dict[object, List[float]]) ->Dict[object, float] :
    """From a dictionary containing lists of something (often the evaluated floats), get a 
    similar dictionary that is the final values.

    Args:
        fullSolutionDictionary (Dict[object, List[object]]): The full solution.

    Returns:
        Dict[object, object]: The final values from the solution.
    """    
    return GetValueFromStateDictionaryAtIndex(fullSolutionDictionary, -1)

    