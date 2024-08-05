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

def TwoBodyAccelerationDifferentialExpression(x, y, z, mu) -> sy.Matrix:
    rSquared = x**2+y**2+z**2    
    xdd = -1*mu*x/(sy.Pow(rSquared, 1.5))
    ydd = -1*mu*y/(sy.Pow(rSquared, 1.5))
    zdd = -1*mu*z/(sy.Pow(rSquared, 1.5))
    return sy.Matrix([xdd, ydd, zdd])