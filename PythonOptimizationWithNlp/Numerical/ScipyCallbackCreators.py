from collections import OrderedDict
from typing import List, Dict
import sympy as sy

from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem

def CreateSimpleCallbackForSolveIvp(timeSymbol, integrationVariableSymbols : List[sy.Expr], equationsOfMotion : Dict[sy.Expr, sy.Expr], substitutionDictionary : Dict[sy.Expr, float], otherArgs: List[sy.Expr]= None) : 
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

def CreateSimpleCallbackForOdeInt(timeSymbol, integrationVariableSymbols : List[sy.Expr], equationsOfMotion : Dict[sy.Expr, sy.Expr], substitutionDictionary : Dict[sy.Expr, float], otherArgs: List[sy.Expr]= None) : 
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

def createBoundaryConditionCallback(boundaryConditionState, allBcAndTransConditions, constantsSubsDict) :
    bcs = []
    for exp in allBcAndTransConditions :
        bc = SymbolicProblem.SafeSubs(exp, constantsSubsDict)
        bcs.append(bc)
    return sy.lambdify(boundaryConditionState, bcs)    

def ConvertOdeIntResultsToDictionary(odeIntSymbolicState, odeIntResults) :
    asDict = OrderedDict()
    i = 0
    for sv in odeIntSymbolicState :
        asDict[sv] = odeIntResults[:,i]
        i=i+1
    return asDict