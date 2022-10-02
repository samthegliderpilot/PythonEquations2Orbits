from typing import List
from pyeq2orb.Coordinates.EquinoctialElements import EquinoctialElements
import sympy as sy

def CreateTwoBodyMotionMatrix(eqElements : EquinoctialElements) ->sy.Matrix :
    mu = eqElements.GravitationalParameter
    p = eqElements.PeriapsisRadius
    keq = eqElements.InclinationSinTermK
    heq = eqElements.InclinationCosTermH
    feq = eqElements.EccentricityCosTermF
    geq = eqElements.EccentricitySinTermG
    leq = eqElements.TrueLongitude
    #wsy = sy.Symbol("w")#(feq, geq, leq)
    w = 1+feq*sy.cos(leq)+geq*sy.sin(leq)
    #s2 = sy.Symbol('s^2')#(heq, keq) # note this is not s but s^2!!! This is a useful cheat
    #s2Func = 1+heq**2+keq**2

    f = sy.Matrix([[0],[0],[0],[0],[0],[sy.sqrt(mu*p)*((w/p)**2)]])
    return f

def CreateTwoBodyList(eqElements : EquinoctialElements) -> List[sy.Expr] :
    mat = CreateTwoBodyMotionMatrix(eqElements)
    rows = []
    for i in range(0, 6) :
        rows.append(mat.row(i))
    return rows