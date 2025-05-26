#%%
# convergence of fourth order finite differences
# for various N, setup a grid in [-pi, pi] and function u(x)
import math
import scipy.sparse as sparse
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as linalg

NVec = [2**x for x in range(3,13)]
plt.subplot(position=[.1, .4, .8, .5])
for n in NVec:
    h = 2*math.pi/n
    x = np.linspace(-1*math.pi+h, math.pi, n)
    u = np.exp(np.sin(x))
    du = np.cos(x)*u
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
plt.title('Convergence of fourth-order finite differences')
plt.semilogy(NVec,np.float_power(NVec, -4),'--')
plt.text(105,5e-8,r'$N^{-4}$',fontsize=12)
plt.show()