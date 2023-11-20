#%%
import sympy as sy
import math
from IPython.display import display, Latex
from collections import OrderedDict
from typing import Dict
sy.init_printing()
import scipyPaperPrinter as jh #type: ignore
import numpy as np
sy.init_printing(use_unicode=True, wrap_line=False)

t= sy.Symbol('t', real=True)
x = sy.Function('x', real=True)(t)
xDot = x.diff(t)
xDDot = xDot.diff(t)
R = sy.Function('R', real=True)(t)
diffExpr = xDDot+x
difeq = sy.Eq(R, diffExpr)
c1 = sy.Function('c_1', real=True)(t)
c2 = sy.Function('c_2', real=True)(t)
c1k = sy.Symbol('c_1', real=True)
c2k = sy.Symbol('c_2', real=True)

sol = c1*sy.sin(t) + c2*sy.cos(t)
solWithKs = c1k*sy.sin(t) + c2k*sy.cos(t)
jh.showEquation(xDot, sol.diff(t), None, True)


rRhs1 = solWithKs.diff(t)
rRhs2 = solWithKs.diff(c1k)*c1.diff(t) + solWithKs.diff(c2k)*c2.diff(t)

def altWay(exprWithConstants, time, coefsWrtTimeDict : Dict[sy.Symbol, sy.Expr]) :
    finalExpr = exprWithConstants.diff(t)
    for k, v in coefsWrtTimeDict.items() :
        finalExpr = finalExpr + exprWithConstants.diff(k)*v.diff(time)
    return finalExpr

otherXDotExpr = altWay(solWithKs, t, {c1k: c1, c2k:c2}).subs(c1.diff(t), 0).subs(c2.diff(t), 0)


jh.showEquation(xDot, rRhs1+rRhs2)
jh.showEquation(xDot, otherXDotExpr)
# but due to some reason, rRhs2 = 0
# rRhs2 = 0
# xDotFromProcess = rRhs1+rRhs2
#jh.showEquation(xDot, xDotFromProcess)

#%%
# because we are trying to compare a perturbed solution to an ideal one AT THE SAME STATE, if 
# the constants are differentiated, if in the immediate area of our state, we look at how the state 
# changes due to differential changes in these 'constants', it must be 0 since it is the same state...?
simplexDdot = sol.diff(t).subs(c1.diff(t), 0).subs(c2.diff(t), 0).diff(t)
jh.showEquation(xDDot, simplexDdot)
altXDdot = altWay(otherXDotExpr, t, {c1k: c1, c2k:c2}).doit()
jh.showEquation(xDDot, altXDdot)

subBackIn = altXDdot +solWithKs
jh.showEquation(R, subBackIn.simplify())

#%%
d2x_dtdc1 = solWithKs.diff(t).diff(c1k)
d2x_dtdc2 = solWithKs.diff(t).diff(c2k)
expr1 = xDot.diff(c1k)*c1.diff(t) + xDot.diff(c2k)*c2.diff(t)
expr2 = xDot.diff(c1k)*c1.diff(t) + xDot.diff(c2k)*c2.diff(t)

def lagrangeBrackets(qs, ps, us, vs):
    finalExpr = 0
    for i in range(0, len(qs)) :
        term1 = qs[i].diff(us[i])*ps[i].diff(vs[i])
        term2 = ps[i].diff(us[i])*qs[i].diff(vs[i])
        finalExpr = finalExpr + term1- term2
    return finalExpr

def poissonBrackets(qs, ps, us, vs):
    finalExpr = 0
    for i in range(0, len(qs)) :
        term1 = us[i].diff(qs[i])*vs[i].diff(ps[i])
        term2 = us[i].diff(ps[i])*vs[i].diff(qs[i])
        finalExpr = finalExpr + term1- term2
    return finalExpr    
display(solWithKs)
display(otherXDotExpr)
display(lagrangeBrackets([solWithKs], [otherXDotExpr], [c1k], [c1k]))
display(lagrangeBrackets([solWithKs], [otherXDotExpr], [c1k], [c2k]))
display(lagrangeBrackets([solWithKs], [otherXDotExpr], [c2k], [c1k]))
display(lagrangeBrackets([solWithKs], [otherXDotExpr], [c2k], [c2k]))