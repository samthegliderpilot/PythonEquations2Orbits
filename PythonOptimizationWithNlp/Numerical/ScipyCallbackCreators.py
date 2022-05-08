from collections import OrderedDict
from typing import List, Dict
import sympy as sy

from PythonOptimizationWithNlp.SymbolicOptimizerProblem import SymbolicProblem

def CreateSimpleCallbackForSolveIvp(timeSymbol, integrationVariableSymbols : List[sy.Expr], equationsOfMotion : Dict[sy.Expr, sy.Expr], substitutionDictionary : Dict[sy.Expr, float]) :
    eomList = []
    for sv in integrationVariableSymbols :
        thisEom = equationsOfMotion[sv].subs(substitutionDictionary)
        eomList.append(thisEom)   
    eomCallback = sy.lambdify([timeSymbol, integrationVariableSymbols], eomList)
    return lambda t,y : eomCallback(t, y)

def CreateSimpleCallbackForOdeInt(timeSymbol, integrationVariableSymbols : List[sy.Expr], equationsOfMotion : Dict[sy.Expr, sy.Expr], substitutionDictionary : Dict[sy.Expr, float]) :
    eomList = []
    for sv in integrationVariableSymbols :
        thisEom = equationsOfMotion[sv].subs(substitutionDictionary)
        eomList.append(thisEom)   
    eomCallback = sy.lambdify([timeSymbol, integrationVariableSymbols], eomList)
    return lambda y,t : eomCallback(t, y)

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