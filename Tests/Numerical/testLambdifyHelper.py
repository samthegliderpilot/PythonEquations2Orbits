import unittest
import sympy as sy
from pyeq2orb.Numerical.LambdifyHelpers import LambdifyHelper

class testLambdifyHelper(unittest.TestCase) :
    def testEmptyCreation(self) :
        helper = LambdifyHelper(None, None, None)
        self.assertEqual([], helper.ExpressionsToLambdify, msg="expression")
        self.assertEqual([], helper.LambdifyArguments, msg="State")
        self.assertEqual({}, helper.SubstitutionDictionary, msg="substitution dict")

    def testDefaultStateEverythingSet(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        y = sy.Function('y')(t)
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        c = sy.Symbol('c')
        d = sy.Symbol('d')
        dx = 2*a*c*t
        dy = -3*b*d*sy.sin(x)

        helper = LambdifyHelper([x,y], [dx, dy], {c:5, d:8})
        self.assertEqual(2, len(helper.LambdifyArguments), msg="2 sv's")
        self.assertEqual(2, len(helper.ExpressionsToLambdify), msg="2 eq's")
        self.assertEqual(2, len(helper.SubstitutionDictionary), msg="2 sub's")
        
        self.assertEqual(x, helper.LambdifyArguments[0], msg="x state")
        self.assertEqual(y, helper.LambdifyArguments[1], msg="y state")
        self.assertEqual(dx, helper.ExpressionsToLambdify[0], msg="dx expr")
        self.assertEqual(dy, helper.ExpressionsToLambdify[1], msg="dy expr")
        self.assertEqual(5, helper.SubstitutionDictionary[c], msg="c subs")
        self.assertEqual(8, helper.SubstitutionDictionary[d], msg="d subs")

    def testCreatingOdeIntCallback(self) :
        t = 1#sy.Symbol('t')
        x = sy.Function('x')(t)
        y = sy.Function('y')(t)
        a = 21#sy.Symbol('a')
        b = 22#sy.Symbol('b')
        c = sy.Symbol('c')
        d = sy.Symbol('d')
        dx = 2*a*c*t
        dy = -3*b*d*sy.sin(x)

        cVal = 5
        dVal = 8
        helper = LambdifyHelper([x,y], [dx, dy], {c:cVal, d:dVal})

        ivpCallback= helper.Lambdify()
        aVal = 21
        bVal = 22
        valuesToEvalAt = {t:1, x:5, y:1,a:aVal, b:bVal, c:cVal, d:dVal}
        dxAtT1 = dx.subs(valuesToEvalAt)
        dyAtT1 = dy.subs(valuesToEvalAt)

        actualVals = ivpCallback(5, 1)
        self.assertEqual(float(dxAtT1), actualVals[0], msg="dx")        
        self.assertEqual(float(dyAtT1), actualVals[1], msg="dy")      

    def testCreateLambdifiedExpressions(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        u = sy.Function('u')(t)
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        xDot = 2*u*a
        uDot = 5*2*b
        callback = LambdifyHelper.CreateLambdifiedExpressions([x,u], [xDot, uDot], {a:3, b:13})
        answer = callback(7,8) 
        self.assertEqual(48, answer[0], msg="x dot val")
        self.assertEqual(130, answer[1], msg="u dot val")              