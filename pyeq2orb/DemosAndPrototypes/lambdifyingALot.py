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

# I was worried that having other custom functions in a lambdified expression like 
# this would be hard.
def someOtherFunction2(x):
    print("hi") 
    return x+5

y =sy.Symbol('y')
x = sy.Symbol('x')
aofx = sy.Function('someOtherFunction')(x)
myExp = y+2
fullExp = myExp + aofx
display(fullExp)

map = {'someOtherFunction' : someOtherFunction2}

lmded = sy.lambdify([x,y], fullExp, map)

display(lmded(3,4))