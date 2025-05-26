#%%
# convergence of fourth order finite differences
# for various N, setup a grid in [-pi, pi] and function u(x)
import math
from scipy import linalg
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as nplinalg

plt.subplot(position=[.1, .4, .8, .5])
for n in range(2, 102, 2):
    h = 2*math.pi/n
    x = np.linspace(-1*math.pi+h, math.pi, n)
    u = np.exp(np.sin(x))
    du = np.cos(x)*u
    column = np.array([0,*[(.5*((-1)**(x)))*(1/math.tan(x*h/2)) for x in range(1,n)]])
    print(column)
    D = linalg.toeplitz(column, -1*column)
    print(repr(D))
    error = nplinalg.norm(D@u-du, ord=np.inf)
    plt.loglog(n,error,'.',markersize=10)
    # no hold needed
plt.grid(visible=True, which="both", ls="--", color='0.65')
plt.xlabel("N", loc="center")
plt.ylabel("error")
plt.title('Convergence of spectral differentiation')
plt.show()
