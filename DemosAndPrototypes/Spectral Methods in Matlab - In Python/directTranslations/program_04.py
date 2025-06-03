#%%
# p4.m - periodic spectral differentiation
from matplotlib import pyplot as plt
import math
import numpy as np
import numpy.linalg as linalg
import scipy.linalg as splinalg
import scipy.sparse as sparse

# Set up grid and differentiation matrix:
pi = math.pi
N = 24
h = 2*pi/N
x = h*np.array(range(1,N+1))
column = np.array([0,*[(.5*((-1)**(x)))*(1/math.tan(x*h/2)) for x in range(1,N)]])
D = splinalg.toeplitz(column,-1*column)

# Differentiation of a hat function:
v = [max(0,1-abs(x2-pi)/2) for x2 in x]
plt.subplot(3,2,1)
plt.plot(x,v,'.-',markersize=10)
plt.axis([0, 2*pi, -.5, 1.5])
plt.grid(visible= True, ls="--", color='0.65')
plt.title = 'function'
plt.subplot(3,2,2), 
plt.plot(x,D@v,'.-',markersize=10)
plt.axis([0, 2*pi, -1, 1])
plt.grid(visible= True, axis='both')
plt.title = 'spectral derivative'

# Differentiation of exp(sin(x)):
v = np.exp(np.sin(x))
vprime = np.cos(x)*v
plt.subplot(3,2,3)
plt.plot(x,v,'.-',markersize=10)
plt.axis([0, 2*pi, 0, 3])
plt.grid(visible= True, ls="--", color='0.65')
plt.subplot(3,2,4)
plt.plot(x,D@v,'.-',markersize=10)
plt.axis([0, 2*pi, -2, 2])
plt.grid(visible= True, axis='both')
error = linalg.norm(D@v-vprime, ord=np.inf)
plt.text(1.2,1.4,'max error = ' + '{:0.5e}'.format(error))