from typing import List
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import ModifiedEquinoctialElements
import sympy as sy

def CreateTwoBodyMotionMatrix(eqElements : ModifiedEquinoctialElements, useSymbolsForAuxVariables = False) ->sy.Matrix :
    mu = eqElements.GravitationalParameter
    p = eqElements.SemiParameter    
    feq = eqElements.EccentricityCosTermF
    geq = eqElements.EccentricitySinTermG    
    leq = eqElements.TrueLongitude
    #wsy = sy.Symbol("w")#(feq, geq, leq)
    if useSymbolsForAuxVariables :
        w = eqElements.WSymbol
    else :
        w = 1+feq*sy.cos(leq)+geq*sy.sin(leq)
    #s2 = sy.Symbol('s^2')#(heq, keq) # note this is not s but s^2!!! This is a useful cheat
    #s2Func = 1+heq**2+keq**2

    f = sy.Matrix([[0],[0],[0],[0],[0],[sy.sqrt(mu*p)*((w/p)**2)]])
    return f

def CreateTwoBodyListForModifiedEquinoctialElements(eqElements : ModifiedEquinoctialElements, useSymbolsForAuxVariables = False) -> List[sy.Eq] :
    mat = CreateTwoBodyMotionMatrix(eqElements, useSymbolsForAuxVariables)
    rows = []
    for i in range(0, 6) :
        rows.append(sy.Eq(eqElements[i].diff(eqElements[i].args[0]), mat.row(i)[0]))
    return rows


# def CreateCartesianOdeAsArray(mu):
#     def twoBodyLambda(t, x):
#         rSqr = x[0]**2+x[1]**2+x[2]**2
#         muOverR2 = mu/(rSqr)
#         return [x[3], x[4], x[5], x[0]*muOverR2, x[1]*muOverR2, x[2]*muOverR2]
#     return twoBodyLambda