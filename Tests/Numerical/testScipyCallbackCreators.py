import unittest
import sympy as sy
from scipy.integrate import odeint, solve_ivp
import PythonOptimizationWithNlp.Numerical.ScipyCallbackCreators as scipyCreator

class testScipyCallbackCreators(unittest.TestCase) :

    def testOdeintCallbackCreator(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        u = sy.Function('u')(t)
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        xDot = 2*u*a
        uDot = 5*t*b
        callback = scipyCreator.CreateSimpleCallbackForOdeint(t, [x,u], {x:xDot, u:uDot}, {a:3}, [b])
        answer = callback([7,8], 2, 13)
        self.assertEqual(48, answer[0], msg="x dot val")
        self.assertEqual(130, answer[1], msg="u dot val")

    def testSolveIvpCallbackCreator(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        u = sy.Function('u')(t)
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        xDot = 2*u*a
        uDot = 5*t*b
        callback = scipyCreator.CreateSimpleCallbackForSolveIvp(t, [x,u], {x:xDot, u:uDot}, {a:3}, [b])
        answer = callback(2, [7,8], 13)
        self.assertEqual(48, answer[0], msg="x dot val")
        self.assertEqual(130, answer[1], msg="u dot val")   

    def testCreateLambdifiedExpressions(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        u = sy.Function('u')(t)
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        xDot = 2*u*a
        uDot = 5*2*b
        callback = scipyCreator.CreateLambdifiedExpressions([x,u], [xDot, uDot], {a:3, b:13})
        answer = callback(7,8) 
        self.assertEqual(48, answer[0], msg="x dot val")
        self.assertEqual(130, answer[1], msg="u dot val")   

    def testOdeIntHelperFunctions(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        u = sy.Function('u')(t)
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        xDot = 2*u*a
        uDot = 5*t*b
        callback = scipyCreator.CreateSimpleCallbackForOdeint(t, [x,u], {x:xDot, u:uDot}, {a:3}, [b])
        tSpan = [0,1]
        answer = odeint(callback,[7,8], tSpan, args=(13,))
        toDict = scipyCreator.ConvertOdeIntResultsToDictionary([x,u], answer)
        self.assertEqual(7, toDict[x][0], msg="x=0 value")
        self.assertEqual(8, toDict[u][0], msg="u=0 value")
        self.assertAlmostEqual(120, toDict[x][1], places=3, msg="x=1 value") 
        self.assertAlmostEqual(40.5, toDict[u][1], places=3,msg="u=1 value")

        toDict = scipyCreator.ConvertEitherIntegratorResultsToDictionary([x,u], answer)
        self.assertEqual(7, toDict[x][0], msg="x=0 value from either")
        self.assertEqual(8, toDict[u][0], msg="u=0 value value from either")
        self.assertAlmostEqual(120, toDict[x][1], places=3, msg="x=1 value value from either") 
        self.assertAlmostEqual(40.5, toDict[u][1], places=3,msg="u=1 value value from either")

        initialState = scipyCreator.GetInitialStateFromIntegratorResults(answer)
        self.assertAlmostEqual(7, initialState[0], places=3, msg="x initial value value from either") 
        self.assertAlmostEqual(8, initialState[1], places=3,msg="u initial value value from either")

        finalState = scipyCreator.GetFinalStateFromIntegratorResults(answer)
        self.assertAlmostEqual(120, finalState[0], places=3, msg="x final value value from either") 
        self.assertAlmostEqual(40.5, finalState[1], places=3,msg="u final value value from either")

    def testOdeIntHelperFunctionsFullOutput(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        u = sy.Function('u')(t)
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        xDot = 2*u*a
        uDot = 5*t*b
        callback = scipyCreator.CreateSimpleCallbackForOdeint(t, [x,u], {x:xDot, u:uDot}, {a:3}, [b])
        tSpan = [0,1]
        answer = odeint(callback,[7,8], tSpan, args=(13,), full_output=True)
        toDict = scipyCreator.ConvertOdeIntResultsToDictionary([x,u], answer)
        self.assertEqual(7, toDict[x][0], msg="x=0 value")
        self.assertEqual(8, toDict[u][0], msg="u=0 value")
        self.assertAlmostEqual(120, toDict[x][1], places=3, msg="x=1 value") 
        self.assertAlmostEqual(40.5, toDict[u][1], places=3,msg="u=1 value")

        toDict = scipyCreator.ConvertEitherIntegratorResultsToDictionary([x,u], answer)
        self.assertEqual(7, toDict[x][0], msg="x=0 value from either")
        self.assertEqual(8, toDict[u][0], msg="u=0 value value from either")
        self.assertAlmostEqual(120, toDict[x][1], places=3, msg="x=1 value value from either") 
        self.assertAlmostEqual(40.5, toDict[u][1], places=3,msg="u=1 value value from either")

        initialState = scipyCreator.GetInitialStateFromIntegratorResults(answer)
        self.assertAlmostEqual(7, initialState[0], places=3, msg="x initial value value from either") 
        self.assertAlmostEqual(8, initialState[1], places=3,msg="u initial value value from either")

        finalState = scipyCreator.GetFinalStateFromIntegratorResults(answer)
        self.assertAlmostEqual(120, finalState[0], places=3, msg="x final value value from either") 
        self.assertAlmostEqual(40.5, finalState[1], places=3,msg="u final value value from either")

    def testSolveIvpHelperFunctions(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        u = sy.Function('u')(t)
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        xDot = 2*u*a
        uDot = 5*t*b
        callback = scipyCreator.CreateSimpleCallbackForSolveIvp(t, [x,u], {x:xDot, u:uDot}, {a:3}, [b])
        answer = solve_ivp(callback, [0.0, 1.0], [7,8], args=(13,))
        toDict = scipyCreator.ConvertSolveIvptResultsToDictionary([x,u], answer)
        self.assertEqual(7, toDict[x][0], msg="x=0 value")
        self.assertEqual(8, toDict[u][0], msg="u=0 value")
        self.assertAlmostEqual(120, toDict[x][-1], places=3, msg="x=1 value") 
        self.assertAlmostEqual(40.5, toDict[u][-1], places=3,msg="u=1 value")

        toDict = scipyCreator.ConvertEitherIntegratorResultsToDictionary([x,u], answer)
        self.assertEqual(7, toDict[x][0], msg="x=0 value from either")
        self.assertEqual(8, toDict[u][0], msg="u=0 value value from either")
        self.assertAlmostEqual(120, toDict[x][-1], places=3, msg="x=1 value value from either") 
        self.assertAlmostEqual(40.5, toDict[u][-1], places=3,msg="u=1 value value from either")

        initialState = scipyCreator.GetInitialStateFromIntegratorResults(answer)
        self.assertAlmostEqual(7, initialState[0], places=3, msg="x initial value value from either") 
        self.assertAlmostEqual(8, initialState[1], places=3,msg="u initial value value from either")

        finalState = scipyCreator.GetFinalStateFromIntegratorResults(answer)
        self.assertAlmostEqual(120, finalState[0], places=3, msg="x final value value from either") 
        self.assertAlmostEqual(40.5, finalState[1], places=3,msg="u final value value from either")

    def testSolveIvpHelperFunctionsDenseOutput(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        u = sy.Function('u')(t)
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        xDot = 2*u*a
        uDot = 5*t*b
        callback = scipyCreator.CreateSimpleCallbackForSolveIvp(t, [x,u], {x:xDot, u:uDot}, {a:3}, [b])
        answer = solve_ivp(callback, [0.0, 1.0], [7,8], args=(13,), dense_output=True)
        toDict = scipyCreator.ConvertSolveIvptResultsToDictionary([x,u], answer)
        self.assertEqual(7, toDict[x][0], msg="x=0 value")
        self.assertEqual(8, toDict[u][0], msg="u=0 value")
        self.assertAlmostEqual(120, toDict[x][-1], places=3, msg="x=1 value") 
        self.assertAlmostEqual(40.5, toDict[u][-1], places=3,msg="u=1 value")

        toDict = scipyCreator.ConvertEitherIntegratorResultsToDictionary([x,u], answer)
        self.assertEqual(7, toDict[x][0], msg="x=0 value from either")
        self.assertEqual(8, toDict[u][0], msg="u=0 value value from either")
        self.assertAlmostEqual(120, toDict[x][-1], places=3, msg="x=1 value value from either") 
        self.assertAlmostEqual(40.5, toDict[u][-1], places=3,msg="u=1 value value from either")

        initialState = scipyCreator.GetInitialStateFromIntegratorResults(answer)
        self.assertAlmostEqual(7, initialState[0], places=3, msg="x initial value value from either") 
        self.assertAlmostEqual(8, initialState[1], places=3,msg="u initial value value from either")

        finalState = scipyCreator.GetFinalStateFromIntegratorResults(answer)
        self.assertAlmostEqual(120, finalState[0], places=3, msg="x final value value from either") 
        self.assertAlmostEqual(40.5, finalState[1], places=3,msg="u final value value from either")        