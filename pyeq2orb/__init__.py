from typing import Dict
def SafeSubs(thingWithSymbols, substitutionDictionary : Dict) :
    """Safely substitute a dictionary into something with sympy expressions returning 
    the same type as thingsWithSymbols.

    Args:
        thingWithSymbols: Either a sympy Expression, or a List of expressions, or a sy.Matrix.  If this is a float, it will be returned
        substitutionDictionary (Dict): The dictionary of things to substitution into thingWithSymbols

    Raises:
        Exception: If this function doesn't know how to do the substitution, an exception will be thrown.

    Returns:
        (same type as thingWithSymbols) : thingWithSymbols substituted with substitutionDictionary
    """
    if isinstance(thingWithSymbols, Dict) :
        for (k,v) in thingWithSymbols.items() :
            thingWithSymbols[k] = SafeSubs(v, substitutionDictionary)
        return

    if isinstance(thingWithSymbols, float) or isinstance(thingWithSymbols, int) or ((hasattr(thingWithSymbols, "is_Float") and thingWithSymbols.is_Float)):
        return thingWithSymbols # it's float, send it back

    if hasattr(thingWithSymbols, "subs") :
        if thingWithSymbols in substitutionDictionary :
            return substitutionDictionary[thingWithSymbols]
        return thingWithSymbols.subs(substitutionDictionary)
    
    if hasattr(thingWithSymbols, "__len__") :
        tbr = []
        for thing in thingWithSymbols :
            tbr.append(SafeSubs(thing, substitutionDictionary))
        return tbr
    raise Exception("Don't know how to do the subs")