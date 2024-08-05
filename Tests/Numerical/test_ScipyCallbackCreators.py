import pytest
import sympy as sy
from scipy.integrate import odeint, solve_ivp # type: ignore
import pyeq2orb.Numerical.ScipyCallbackCreators as scipyCreator # type: ignore
from pyeq2orb.Numerical.LambdifyHelpers import OdeLambdifyHelper # type: ignore
from pyeq2orb.Utilities.utilitiesForTest import assertAlmostEquals #type: ignore

def testOdeIntHelperFunctions() :
    t = sy.Symbol('t')
    x = sy.Function('x')(t)
    u = sy.Function('u')(t)
    a = sy.Symbol('a')
    b = sy.Symbol('b')
    xDot = 2*u*a
    uDot = 5*t*b
    lambdifyHelper = OdeLambdifyHelper(t, [x,u], [xDot, uDot], [b], {a:3})
    callback = lambdifyHelper.CreateSimpleCallbackForOdeint()
    tSpan = [0,1]
    answer = odeint(callback,[7,8], tSpan, args=(13,))
    toDict = scipyCreator.ConvertOdeIntResultsToDictionary([x,u], answer)
    assert 7== toDict[x][0], "x=0 value"
    assert 8== toDict[u][0], "u=0 value"
    assert pytest.approx(toDict[x][1], 0.001)== 120.0, "x=1 value"
    assert pytest.approx(toDict[u][1], 0.001) == 40.5, "u=1 value"

    toDict = scipyCreator.ConvertEitherIntegratorResultsToDictionary([x,u], answer)
    assert 7== toDict[x][0], "x=0 value from either"
    assert 8== toDict[u][0], "u=0 value value from either"
    assert pytest.approx(toDict[x][1], 0.001)== 120.0, "x=1 value"
    assert pytest.approx(toDict[u][1], 0.001) == 40.5, "u=1 value"

    initialState = scipyCreator.GetInitialStateFromIntegratorResults(answer)
    assert pytest.approx(initialState[0], 0.001) == 7.0, "x initial value value from either" 
    assert pytest.approx(initialState[1], 0.001) ==8.0, "u initial value value from either"

    finalState = scipyCreator.GetFinalStateFromIntegratorResults(answer)
    assert pytest.approx(finalState[0], 0.001)== 120.0, "x=1 value"
    assert pytest.approx(finalState[1], 0.001) == 40.5, "u=1 value"

def testOdeIntHelperFunctionsFullOutput() :
    t = sy.Symbol('t')
    x = sy.Function('x')(t)
    u = sy.Function('u')(t)
    a = sy.Symbol('a')
    b = sy.Symbol('b')
    xDot = 2*u*a
    uDot = 5*t*b
    lambdifyHelper = OdeLambdifyHelper(t, [x,u], [xDot, uDot], [b], {a:3})
    callback = lambdifyHelper.CreateSimpleCallbackForOdeint()
    tSpan = [0,1]
    answer = odeint(callback,[7,8], tSpan, args=(13,), full_output=True)
    toDict = scipyCreator.ConvertOdeIntResultsToDictionary([x,u], answer)
    assert 7== toDict[x][0], "x=0 value"
    assert 8== toDict[u][0], "u=0 value"
    assertAlmostEquals(120, toDict[x][1], 3, "x=1 value")
    assertAlmostEquals(40.5, toDict[u][1], 3,"u=1 value")

    toDict = scipyCreator.ConvertEitherIntegratorResultsToDictionary([x,u], answer)
    assert 7== toDict[x][0], "x=0 value from either"
    assert 8== toDict[u][0], "u=0 value value from either"
    assertAlmostEquals(120, toDict[x][1], 3, "x=1 value value from either") 
    assertAlmostEquals(40.5, toDict[u][1], 3,"u=1 value value from either")

    initialState = scipyCreator.GetInitialStateFromIntegratorResults(answer)
    assertAlmostEquals(7, initialState[0], 3, "x initial value value from either") 
    assertAlmostEquals(8, initialState[1], 3,"u initial value value from either")

    finalState = scipyCreator.GetFinalStateFromIntegratorResults(answer)
    assertAlmostEquals(120, finalState[0], 3, "x final value value from either") 
    assertAlmostEquals(40.5, finalState[1], 3,"u final value value from either")

def testSolveIvpHelperFunctions() :
    t = sy.Symbol('t')
    x = sy.Function('x')(t)
    u = sy.Function('u')(t)
    a = sy.Symbol('a')
    b = sy.Symbol('b')
    xDot = 2*u*a
    uDot = 5*t*b
    lambdifyHelper = OdeLambdifyHelper(t, [x,u], [xDot, uDot], [b], {a:3})
    callback = lambdifyHelper.CreateSimpleCallbackForSolveIvp()
    answer = solve_ivp(callback, [0.0, 1.0], [7,8], args=(13,))
    toDict = scipyCreator.ConvertSolveIvpResultsToDictionary([x,u], answer)
    assert 7== toDict[x][0], "x=0 value"
    assert 8== toDict[u][0], "u=0 value"
    assertAlmostEquals(120, toDict[x][-1], 3, "x=1 value") 
    assertAlmostEquals(40.5, toDict[u][-1], 3,"u=1 value")

    toDict = scipyCreator.ConvertEitherIntegratorResultsToDictionary([x,u], answer)
    assert 7== toDict[x][0], "x=0 value from either"
    assert 8== toDict[u][0], "u=0 value value from either"
    assertAlmostEquals(120, toDict[x][-1], 3, "x=1 value value from either") 
    assertAlmostEquals(40.5, toDict[u][-1], 3,"u=1 value value from either")

    initialState = scipyCreator.GetInitialStateFromIntegratorResults(answer)
    assertAlmostEquals(7, initialState[0], 3, "x initial value value from either") 
    assertAlmostEquals(8, initialState[1], 3,"u initial value value from either")

    finalState = scipyCreator.GetFinalStateFromIntegratorResults(answer)
    assertAlmostEquals(120, finalState[0], 3, "x final value value from either") 
    assertAlmostEquals(40.5, finalState[1], 3,"u final value value from either")

def testSolveIvpHelperFunctionsDenseOutput() :
    t = sy.Symbol('t')
    x = sy.Function('x')(t)
    u = sy.Function('u')(t)
    a = sy.Symbol('a')
    b = sy.Symbol('b')
    xDot = 2*u*a
    uDot = 5*t*b
    lambdifyHelper = OdeLambdifyHelper(t, [x,u], [xDot, uDot], [b], {a:3})
    callback = lambdifyHelper.CreateSimpleCallbackForSolveIvp()
    answer = solve_ivp(callback, [0.0, 1.0], [7,8], args=(13,), dense_output=True)
    toDict = scipyCreator.ConvertSolveIvpResultsToDictionary([x,u], answer)
    assert 7== toDict[x][0], "x=0 value"
    assert 8== toDict[u][0], "u=0 value"
    assertAlmostEquals(120, toDict[x][-1], 3, "x=1 value") 
    assertAlmostEquals(40.5, toDict[u][-1], 3,"u=1 value")

    toDict = scipyCreator.ConvertEitherIntegratorResultsToDictionary([x,u], answer)
    assert 7== toDict[x][0], "x=0 value from either"
    assert 8== toDict[u][0], "u=0 value value from either"
    assertAlmostEquals(120, toDict[x][-1], 3, "x=1 value value from either") 
    assertAlmostEquals(40.5, toDict[u][-1], 3,"u=1 value value from either")

    initialState = scipyCreator.GetInitialStateFromIntegratorResults(answer)
    assertAlmostEquals(7, initialState[0], 3, "x initial value value from either") 
    assertAlmostEquals(8, initialState[1], 3,"u initial value value from either")

    finalState = scipyCreator.GetFinalStateFromIntegratorResults(answer)
    assertAlmostEquals(120, finalState[0], 3, "x final value value from either") 
    assertAlmostEquals(40.5, finalState[1], 3,"u final value value from either")        