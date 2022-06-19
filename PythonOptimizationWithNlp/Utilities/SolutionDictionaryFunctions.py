from typing import Dict, List

def GetValueFromStateDictionaryAtIndex(fullSolutionDictionary : Dict[object, List[object]], index : int) ->Dict[object, object] :
    """From a dictionary containing lists of something (often the evaluated floats), get a 
    similar dictionary that is the final values.

    Args:
        fullSolutionDictionary (Dict[object, List[object]]): The full solution.

    Returns:
        Dict[object, object]: The final values from the solution.
    """    
    finalValues = {}
    for (key, value) in fullSolutionDictionary.items() :
        finalValues[key] = value[index]
    return finalValues 

def GetInitialStateDictionary(fullSolutionDictionary : Dict[object, List[object]]) ->Dict[object, object] :
    """From a dictionary containing lists of something (often the evaluated floats), get a 
    similar dictionary that is the initial values.

    Args:
        fullSolutionDictionary (Dict[object, List[object]]): The full solution.

    Returns:
        Dict[object, object]: The initial values from the solution.
    """
    return GetValueFromStateDictionaryAtIndex(fullSolutionDictionary, 0)

def GetFinalStateDictionary(fullSolutionDictionary : Dict[object, List[object]]) ->Dict[object, object] :
    """From a dictionary containing lists of something (often the evaluated floats), get a 
    similar dictionary that is the final values.

    Args:
        fullSolutionDictionary (Dict[object, List[object]]): The full solution.

    Returns:
        Dict[object, object]: The final values from the solution.
    """    
    return GetValueFromStateDictionaryAtIndex(fullSolutionDictionary, -1)

    