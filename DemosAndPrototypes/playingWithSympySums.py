#%%
import sympy as sy
import scipyPaperPrinter as jh
from IPython.display import display
import math
i, j, k, l = sy.symbols('i, j, k, l', cls=sy.Dummy)
A = sy.IndexedBase('A')
x = sy.Symbol('x', real=True)
N = sy.symbols('N', integer=True, positive=True)
expr = ((-1)**i/sy.factorial(2*i+1))*x**(2*i+1)
sum = sy.Sum(expr, (i, 0, N))

display(sum)
sumCallback = sy.lambdify([N, x], sum)
sympyLambdifiedValue = float(sumCallback(4, 0.5))

mathPyValue = math.sin(0.5)
display(sy.Eq(sy.Symbol('sin(0.5)_{mathpy}', real=True), mathPyValue))

display(sy.Eq(sy.Symbol('sin(0.5)_{approx}', real=True), sympyLambdifiedValue))
display(sy.Eq(sy.Symbol('diff_{lmd_vs_mathpy}'), mathPyValue-sympyLambdifiedValue))

# manual sum
callable = sy.lambdify([N, x, i], expr)
manual_sum = 0
for iVal in [0,1,2,3,4]:
    manual_sum = manual_sum + callable(5, 0.5, iVal)
display(sy.Eq(sy.Symbol('sin(0.5)_{backwards}', real=True), manual_sum))
display(sy.Eq(sy.Symbol('diff_{manual_vs_mathpy}'), mathPyValue-manual_sum))
#%%
import numpy as np

expr2 = (1.0/(10.0**i))
sum2 = sy.Sum(expr2, (i, 0, N))
sumCallable2 = sy.lambdify([N, x], sum2)
sympyLambdifiedValue2 = sumCallable2(np.longlong(18), np.float128(0.5))
print(sympyLambdifiedValue2)

callable2 = sy.lambdify([N, x, i], expr2)
manual_sum = np.float128(0.0)
for iVal in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
    thisTerm = np.float128(callable2(15, np.float128(0.5), iVal))
    print(thisTerm)
    manual_sum = manual_sum + thisTerm
print(manual_sum)
