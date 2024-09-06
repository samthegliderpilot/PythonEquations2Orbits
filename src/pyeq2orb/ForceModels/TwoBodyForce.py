from typing import List, Dict, Optional
from pyeq2orb.Coordinates.ModifiedEquinoctialElementsModule import ModifiedEquinoctialElements
import sympy as sy

def CreateTwoBodyMotionMatrix(eqElements : ModifiedEquinoctialElements, substitutionDictionary : Optional[Dict[sy.Expr, sy.Expr]] = None) ->sy.Matrix :
    mu = eqElements.GravitationalParameter
    p = eqElements.SemiParameter    
    feq = eqElements.EccentricityCosTermF
    geq = eqElements.EccentricitySinTermG    
    leq = eqElements.TrueLongitude
    wsy = eqElements.WSymbol
    if substitutionDictionary is not None :
        substitutionDictionary[wsy] = 1+feq*sy.cos(leq)+geq*sy.sin(leq)
    else :
        wsy = 1+feq*sy.cos(leq)+geq*sy.sin(leq)
    #s2 = sy.Symbol('s^2')#(heq, keq) # note this is not s but s^2!!! This is a useful cheat
    #s2Func = 1+heq**2+keq**2

    f = sy.Matrix([[0],[0],[0],[0],[0],[sy.sqrt(mu*p)*((wsy/p)**2)]])
    return f

def CreateTwoBodyListForModifiedEquinoctialElements(eqElements : ModifiedEquinoctialElements, substitutionDictionary : Optional[Dict[sy.Expr, sy.Expr]] = None) -> List[sy.Eq] :
    mat = CreateTwoBodyMotionMatrix(eqElements, substitutionDictionary)
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