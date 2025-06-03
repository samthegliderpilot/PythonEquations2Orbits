#%%
# p4.m - periodic spectral differentiation
from matplotlib import pyplot as plt
import math
import numpy as np
import numpy.linalg as linalg
import scipy.linalg as splinalg
import scipy.fft as fft

# Set up grid and differentiation matrix:
pi = math.pi
N = 24
h = 2*pi/N
x = h*np.array(range(1,N+1))
column = np.array([0,*[(.5*((-1)**(x)))*(1/math.tan(x*h/2)) for x in range(1,N)]])
D = splinalg.toeplitz(column,-1*column)

# Differentiation of a hat function:
v = [max(0,1-abs(x2-pi)/2) for x2 in x]
v_hat = fft.fft(v)
w_hat_arr = np.array([*list(range(0, int(N/2))), 0, *list(range(int(-(N-1)/2), 0,))])
w_hat = 1j*w_hat_arr*v_hat
w = np.real(fft.ifft(w_hat))
plt.subplot(3,2,1)
plt.plot(x,v,'.-',markersize=10)
plt.axis([0, 2*pi, -.5, 1.5])
plt.grid(visible= True, axis='both', ls="--", color='0.65')
plt.title = 'function'
plt.subplot(3,2,2), 
plt.plot(x,w,'.-',markersize=10)
plt.axis([0, 2*pi, -1, 1])
plt.grid(visible= True, axis='both', ls="--", color='0.65')
plt.title = 'spectral derivative'

# Differentiation of exp(sin(x)):
v = np.exp(np.sin(x))
vprime = np.cos(x)*v

v_hat = fft.fft(v)
w_hat = 1j*w_hat_arr*v_hat
w = np.real(fft.ifft(w_hat))
plt.subplot(3,2,3)
plt.plot(x,v,'.-',markersize=10)
plt.axis([0, 2*pi, 0, 3])
plt.grid(visible= True, axis='both', ls="--", color='0.65')
plt.subplot(3,2,4)
plt.plot(x,w,'.-',markersize=10)
plt.axis([0, 2*pi, -2, 2])
plt.grid(visible= True, axis='both', ls="--", color='0.65')
error = linalg.norm(w-vprime, ord=np.inf)
plt.text(1.2,1.4,'max error = ' + '{:0.5e}'.format(error))