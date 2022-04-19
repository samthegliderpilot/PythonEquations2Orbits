import sympy as sy
from PythonOptimizationWithNlp.Symbolics.ScalingHelper import ScalingHelper
import unittest

class testScalingHelper(unittest.TestCase) :

    def testscaleExpressionByFinalTime(self) :
        t = sy.Symbol('t')
        tf = sy.Symbol('t_f')
        tau = sy.Symbol('tau')

        someExpression = 5 * sy.sin(t)

        scaledExpression = ScalingHelper.scaleExpressionsByFinalTime(someExpression, t, tf, tau)
        self.assertEqual(0, (scaledExpression - 5.0*tf*sy.sin(tau)))

    def testscaleExpressionsByFinalTime(self) :
        t = sy.Symbol('t')
        tf = sy.Symbol('t_f')
        tau = sy.Symbol('tau')

        someExpression = 5 * sy.sin(t)
        someOtherExpression = 3.0 *sy.exp(2.0*sy.cos(t))

        scaledExpression = ScalingHelper.scaleExpressionsByFinalTime([someExpression, someOtherExpression], t, tf, tau)
        self.assertEqual(0, (scaledExpression[0] - 5.0*tf*sy.sin(tau)))
        self.assertEqual(0, (scaledExpression[1] - tf* 3.0 *sy.exp(2.0*sy.cos(tau))))

    # def testScaleDerivativeByFinalTime(self) :
    #     t = sy.Symbol('t')
    #     tau = sy.Symbol('tau')
    #     tf = sy.Symbol('t_f')
    #     x = sy.Function('x')(t)
    #     y = 0.5*x*x
    #     dydt = sy.diff(y, t).doit()
    #     scaled = ScalingHelper.simpleScale(dydt, t, tf, tau).doit()
    #     self.assertEqual(0, (scaled - x/tf)) 

    # def testSimpleScale(self) :
    #     t = sy.Symbol('t')
    #     tf = sy.Symbol('t_f')
    #     tau = sy.Symbol('tau')

    #     someExpression = 5 * sy.sin(t)
    #     someOtherExpression = 3.0 *sy.exp(2.0*sy.cos(t))

    #     scaledExpression = ScalingHelper.simpleScale([someExpression, someOtherExpression], t, tf, tau)
    #     self.assertEqual(0, (scaledExpression[0] - 5.0*tf*sy.sin(tau)))
    #     self.assertEqual(0, (scaledExpression[1] - tf* 3.0 *sy.exp(2.0*sy.cos(tau))))
