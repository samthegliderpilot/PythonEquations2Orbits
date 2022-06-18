import unittest
import sympy as sy
from PythonOptimizationWithNlp.Numerical.LambdifyModule import LambdifyHelper

class testLambdifyHelper(unittest.TestCase) :
    def testEmptyCreation(self) :
        helper = LambdifyHelper(None, None, None, None, None)
        self.assertEqual([], helper.ExpressionsToLambdify, msg="expression")
        self.assertEqual(None, helper.Time, msg="time")
        self.assertEqual([], helper.OtherArguments, msg="other args")
        self.assertEqual([], helper.StateVariableListOrdered, msg="State")
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

        helper = LambdifyHelper(t, [x,y], [dx, dy], [a,b], {c:5, d:8})
        self.assertEqual(2, len(helper.StateVariableListOrdered), msg="2 sv's")
        self.assertEqual(2, len(helper.ExpressionsToLambdify), msg="2 eq's")
        self.assertEqual(2, len(helper.SubstitutionDictionary), msg="2 sub's")
        self.assertEqual(2, len(helper.OtherArguments), msg="2 other's")
        
        self.assertEqual(x, helper.StateVariableListOrdered[0], msg="x state")
        self.assertEqual(y, helper.StateVariableListOrdered[1], msg="y state")
        self.assertEqual(dx, helper.ExpressionsToLambdify[0], msg="dx expr")
        self.assertEqual(dy, helper.ExpressionsToLambdify[1], msg="dy expr")
        self.assertEqual(a, helper.OtherArguments[0], msg="a other")
        self.assertEqual(b, helper.OtherArguments[1], msg="b other")
        self.assertEqual(5, helper.SubstitutionDictionary[c], msg="c subs")
        self.assertEqual(8, helper.SubstitutionDictionary[d], msg="d subs")
        
        expectedDefaultState = [t, [x, y], [a,b]]
        self.assertListEqual(expectedDefaultState, helper.CreateDefaultState(), msg="default state")

    def testCreatingSolveIvpCallback(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        y = sy.Function('y')(t)
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        c = sy.Symbol('c')
        d = sy.Symbol('d')
        dx = 2*a*c*t
        dy = -3*b*d*sy.sin(x)

        cVal = 5
        dVal = 8
        helper = LambdifyHelper(t, [x,y], [dx, dy], [a,b], {c:cVal, d:dVal})

        ivpCallback= helper.CreateSimpleCallbackForSolveIvp()
        aVal = 21
        bVal = 22
        valuesToEvalAt = {t:1, x:5, y:1,a:aVal, b:bVal, c:cVal, d:dVal}
        dxAtT1 = dx.subs(valuesToEvalAt)
        dyAtT1 = dy.subs(valuesToEvalAt)

        actualVals = ivpCallback(1, [5, 1], aVal, bVal)
        self.assertEqual(float(dxAtT1), actualVals[0], msg="dx")        
        self.assertEqual(float(dyAtT1), actualVals[1], msg="dy")


    def testCreatingSolveIvpCallbackNoOtherArgs(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        y = sy.Function('y')(t)
        c = sy.Symbol('c')
        d = sy.Symbol('d')
        dx = 2*c*t
        dy = -3*d*sy.sin(x)

        cVal = 5
        dVal = 8
        helper = LambdifyHelper(t, [x,y], [dx, dy], None, {c:cVal, d:dVal})

        ivpCallback= helper.CreateSimpleCallbackForSolveIvp()
        valuesToEvalAt = {t:1, x:5, y:1, c:cVal, d:dVal}
        dxAtT1 = dx.subs(valuesToEvalAt)
        dyAtT1 = dy.subs(valuesToEvalAt)

        actualVals = ivpCallback(1, [5, 1])
        self.assertEqual(float(dxAtT1), actualVals[0], msg="dx")        
        self.assertEqual(float(dyAtT1), actualVals[1], msg="dy")

    def testCreatingOdeIntCallback(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        y = sy.Function('y')(t)
        a = sy.Symbol('a')
        b = sy.Symbol('b')
        c = sy.Symbol('c')
        d = sy.Symbol('d')
        dx = 2*a*c*t
        dy = -3*b*d*sy.sin(x)

        cVal = 5
        dVal = 8
        helper = LambdifyHelper(t, [x,y], [dx, dy], [a,b], {c:cVal, d:dVal})

        ivpCallback= helper.CreateSimpleCallbackForOdeint()
        aVal = 21
        bVal = 22
        valuesToEvalAt = {t:1, x:5, y:1,a:aVal, b:bVal, c:cVal, d:dVal}
        dxAtT1 = dx.subs(valuesToEvalAt)
        dyAtT1 = dy.subs(valuesToEvalAt)

        actualVals = ivpCallback([5, 1], 1, aVal, bVal)
        self.assertEqual(float(dxAtT1), actualVals[0], msg="dx")        
        self.assertEqual(float(dyAtT1), actualVals[1], msg="dy")

    def testCreatingOdeIntCallbackNoOtherArgs(self) :
        t = sy.Symbol('t')
        x = sy.Function('x')(t)
        y = sy.Function('y')(t)
        c = sy.Symbol('c')
        d = sy.Symbol('d')
        dx = 2*c*t
        dy = -3*d*sy.sin(x)

        cVal = 5
        dVal = 8
        helper = LambdifyHelper(t, [x,y], [dx, dy], None, {c:cVal, d:dVal})

        ivpCallback= helper.CreateSimpleCallbackForOdeint()
        valuesToEvalAt = {t:1, x:5, y:1, c:cVal, d:dVal}
        dxAtT1 = dx.subs(valuesToEvalAt)
        dyAtT1 = dy.subs(valuesToEvalAt)

        actualVals = ivpCallback([5, 1], 1)
        self.assertEqual(float(dxAtT1), actualVals[0], msg="dx")        
        self.assertEqual(float(dyAtT1), actualVals[1], msg="dy")        