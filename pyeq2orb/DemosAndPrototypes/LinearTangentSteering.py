#%%
import __init__
import sympy as sy
import math
import os
import sys
from IPython.display import display
from collections import OrderedDict
sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)
sy.init_printing()
import scipyPaperPrinter as jh
import numpy as np


t = sy.Symbol('t', real=True)
tf = sy.Symbol('t_f', real=True)
y1 = sy.Function('y_1', real=True)(t)
y2 = sy.Function('y_2', real=True)(t)
y3 = sy.Function('y_3', real=True)(t)
y4 = sy.Function('y_4', real=True)(t)

a = sy.Symbol('a', real=True, constant=True)
u = sy.Function('u', real=True)(t)

y1Dot = y3
y2Dot = y4
y3Dot = a*sy.cos(u)
y4Dot = a*sy.sin(u)

y = [y1,y2,y3,y4]
dydx = [y1Dot, y2Dot, y3Dot, y4Dot]



n = 3
h = n/tf

nsv = 4
nu = 1
# standard approach

# def createGuessVector(nVariables, steps, nControls) :
#     return np.zeros(nVariables*steps+1+nControls*steps)

def createSymbolicXVector(n) :
    x=[0] * (n*5+1)#sy.Matrix.zeros(n*5+1, 1)
    x[0] = tf
    for i in range(0, n) :
        x[i*5+1] = sy.Symbol('y_{1_{' + str(i) + '}}')
        x[i*5+2] =sy.Symbol('y_{2_{' + str(i) + '}}')
        x[i*5+3]=sy.Symbol('y_{3_{' + str(i) + '}}')
        x[i*5+4]=sy.Symbol('y_{4_{' + str(i) + '}}')
        x[i*5+5]=sy.Symbol('u_{' + str(i) + '}')
    display(x)
    return x

xS = createSymbolicXVector(n)
display(xS)
#x = createGuessVector(n, nsv, nu)
c1 = sy.Function('C_1')(*xS)
c2 = sy.Function('C_2')(*xS)
c3 = sy.Function('C_3')(*xS)
c4 = sy.Function('C_4')(*xS)
cs = [c1, c2, c3, c4]



def createTrapizodalEquations(steps) :
    eqs = []
    tfV = xS[0]
    for step in range(0, (steps-1)) :
        colThisStart = int(step%5)*5+1
        xFull = {tf: xS[0],  y[0]: xS[colThisStart],  y[1] : xS[colThisStart+1],  y[2]: xS[colThisStart+2],  y[3] : xS[colThisStart+3],  u: xS[colThisStart+4]}
        for i in range(0, 4) :
            dy = dydx[i]
            colNextStart =colThisStart+5
            xNextFull = {tf: xS[0],  y[0]: xS[colNextStart],  y[1] : xS[colNextStart+1],  y[2]: xS[colNextStart+2],  y[3] : xS[colNextStart+3],  u: xS[colNextStart+4]}
            eq1 = xS[(step+1)*5 + 1+i]-xS[step*5+1+i] - steps/tfV * (dy.subs(xFull)+dy.subs(xNextFull))
            eqs.append(eq1)
    return eqs

display(createTrapizodalEquations(n))

def trapizodalMatrixCreation(x, steps) :
    lenx = len(x)
    colCount = lenx
    rowCount = (steps-1)*4
    theMatrix = sy.Matrix.zeros(rowCount, colCount)
    districtrizedEquations = createTrapizodalEquations(steps) 
    for r in range(0, 4*(steps-1)) :
        for c in range(0, lenx) : 
            try :
                theMatrix[r, c] = districtrizedEquations[r].diff(x[c]).doit()
            except:
                display(theMatrix)
                raise
    return theMatrix

display(trapizodalMatrixCreation(xS, n))
