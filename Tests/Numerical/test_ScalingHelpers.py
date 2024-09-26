import pytest
import math
import sympy as sy
from pyeq2orb.Numerical.ScalingHelpers import scaledEquationsOfMotionResult


def testScalingStatesInOdes() :
    t = sy.Symbol('t', real=True)
    r = sy.Function('r', real=True, positive=True)(t)
    u = sy.Function('u', real=True)(t)
    a = sy.Symbol('a', real=True)

    rDot = u
    uDot = a

    r0= sy.Symbol('r_0', real=True, positive=True)
    u0= sy.Symbol('u_0', real=True, positive=True)

    rBar = sy.Function(r'\bar{r}', real=True, positive=True)(t)
    uBar = sy.Function(r'\bar{u}', real=True)(t)

    expectedScaledRDot = uBar*5.0/r0
    expectedScaledUDot = a/5.0

    scaledExpressions = scaledEquationsOfMotionResult.ScaleStateVariablesInFirstOrderOdes([r,u], [rDot, uDot], [rBar, uBar], [r0, 5.0])
    assert expectedScaledRDot == scaledExpressions.scaledFirstOrderDynamics[0]
    assert expectedScaledUDot == scaledExpressions.scaledFirstOrderDynamics[1]

def testScalingStatesAndTimesInOdes() :
    t = sy.Symbol('t', real=True)
    r = sy.Function('r', real=True, positive=True)(t)
    u = sy.Function('u', real=True)(t)
    a = sy.Symbol('a', real=True)

    rDot = u
    uDot = a

    r0= sy.Symbol('r_0', real=True, positive=True)
    u0= sy.Symbol('u_0', real=True, positive=True)
    tf = sy.Symbol('t_f', real=True)

    tau = sy.Symbol(r'\tau', real=True, positive=True)
    rBar = sy.Function(r'r', real=True, positive=True)(tau)
    uBar = sy.Function(r'u', real=True)(tau)

    expectedScaledRDot = uBar*tf
    expectedScaledUDot = a*tf

    scaledExpressions = scaledEquationsOfMotionResult.ScaleTimeInFirstOrderOdes([r,u], t, [rDot, uDot], tau, tf)
    assert expectedScaledRDot == scaledExpressions.scaledFirstOrderDynamics[0]
    assert expectedScaledUDot == scaledExpressions.scaledFirstOrderDynamics[1]

def testScalingStatesAndTimeInOdes() :
    t = sy.Symbol('t', real=True)
    r = sy.Function('r', real=True, positive=True)(t)
    u = sy.Function('u', real=True)(t)
    a = sy.Symbol('a', real=True)

    rDot = u
    uDot = a

    r0= sy.Symbol('r_0', real=True, positive=True)
    u0= sy.Symbol('u_0', real=True, positive=True)
    tf = sy.Symbol('t_f', real=True)

    tau = sy.Symbol(r'\tau', real=True, positive=True)
    rBar = sy.Function(r'\bar{r}', real=True, positive=True)(t)
    uBar = sy.Function(r'\bar{u}', real=True)(t)

    rBarTau = sy.Function(r'\bar{r}', real=True, positive=True)(tau)
    uBarTau = sy.Function(r'\bar{u}', real=True)(tau)

    expectedScaledRDot = uBarTau*5.0*tf/r0
    expectedScaledUDot = a*tf/5.0

    scaledExpressions = scaledEquationsOfMotionResult.ScaleStateVariablesAndTimeInFirstOrderOdes([r,u], [rDot, uDot], [rBar, uBar], [r0, 5.0], tau, tf)
    assert expectedScaledRDot == scaledExpressions.scaledFirstOrderDynamics[0]
    assert expectedScaledUDot == scaledExpressions.scaledFirstOrderDynamics[1]
