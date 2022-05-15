import unittest
import sympy as sy
from scipy.integrate import odeint
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

    def testConvertOdeIntResultsToDictionary(self) :
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

