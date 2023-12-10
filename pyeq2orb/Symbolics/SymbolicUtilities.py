import sympy as sy
from typing import Optional, List, cast, Dict

def MakeMatrixOfSymbols(baseString : str, rows: int, cols : int,funcArgs : Optional[List[sy.Expr]]= None) :
    endString = ''
    if baseString.endswith('}') :
        baseString = baseString[:-1]
        endString = '}'
    mat = sy.Matrix.zeros(rows, cols)
    makeFunctions = funcArgs == None or len(cast(List[sy.Expr], funcArgs)) == 0
    for r in range(0, rows) :
        for c in range(0, cols):
            if makeFunctions:
                mat[r,c] = sy.Symbol(baseString + "_{" + str(r) + "," + str(c)+"}" + endString)
            else:
                mat[r,c] = sy.Function(baseString + "_{" + str(r) + "," + str(c)+"}"+ endString)(*funcArgs)
    return mat


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
        finalExp = thingWithSymbols
        finalExp = finalExp.subs(substitutionDictionary).doit(deep=True)
        # for k,v in substitutionDictionary.items() :
        #     finalExp = finalExp.subs(k, v).doit(deep=True) # this makes a difference?!?
        return finalExp
    
    if hasattr(thingWithSymbols, "__len__") :
        tbr = []
        for thing in thingWithSymbols :
            tbr.append(SafeSubs(thing, substitutionDictionary))
        return tbr
    raise Exception("Don't know how to do the subs")    