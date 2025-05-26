#%%
# convergence of fourth order finite differences
# for various N, setup a grid in [-pi, pi] and function u(x)
import math
import scipy.sparse as sparse
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as linalg

import sympy as sy
def performErrorAnalysis(expr, x, NVec, title  :str):

    uCallback = sy.lambdify(x, expr, modules="numpy")
    duCallback = sy.lambdify(x, expr.diff(x), modules='numpy')
    
    plt.subplot(position=[.1, .4, .8, .5])
    for n in NVec:
        h = 2*math.pi/n
        x = np.linspace(-1*math.pi+h, math.pi, n)
        u = uCallback(x)
        du = duCallback(x)
        e =np.ones(n)
        D1 = sparse.diags_array([(2.0/3.0)*e, [2.0/3.0]], offsets=[1, -(n-1)], shape =(n,n))
        D2 = sparse.diags_array([(1.0/12.0)*e, [1.0/12.0, 1.0/12.0]], offsets=[2, -(n-2)],shape =(n,n))
        DTemp = D1-D2
        D = (DTemp-DTemp.transpose())/h
        error = linalg.norm(D@u-du, ord=np.inf)
        plt.loglog(n,error,'.',markersize=10)
        # no hold needed
    plt.grid(visible=True, which="both", ls="--", color='0.65')
    plt.xlabel("N", loc="center")
    plt.ylabel("error")
    plt.title('Convergence of fourth-order finite differences: ' + title)
    plt.semilogy(NVec,np.float_power(NVec, -4),'--')
    plt.text(105,5e-8,r'$N^{-4}$',fontsize=12)
    plt.show()

x = sy.Symbol('x', real=True)
#%%
NVec = [2**x for x in range(3,13)]    
expr = sy.exp(sy.sin(x))
performErrorAnalysis(expr, x, NVec, "P1")

#%%
NVec = [2**x for x in range(3,16)] 
expr = sy.exp(sy.sin(x))   
performErrorAnalysis(expr, x, NVec, "1.4")

#%%
NVec = [2**x for x in range(3,13)]    
expr = sy.exp(sy.sin(x)**2)
performErrorAnalysis(expr, x, NVec, "1.5a")

#%%
NVec = [2**x for x in range(3,13)]    
expr = sy.exp(sy.sin(x)*sy.Abs(sy.sin(x)))
performErrorAnalysis(expr, x, NVec, "1.5b")

#%%
from IPython.display import display
import sympy as sy

x, x0 = sy.symbols("x,x_0", real=True)
f = sy.Function("f", real=True)(x)
def factorial(n):
    if n <= 0:
        return 1
    else:
        return n*factorial(n-1)
def taylor(function : sy.Function, x, x0,n):
    i = 0
    p = function
    while i <= n-1:
        p = p + function.diff(x, i).subs(x,x0)/(factorial(i))*(x-x0)**i
        i += 1
    return p

display(taylor(f, x, 0, 2))
#display(f(x).series(x, x0=x0, n=3))