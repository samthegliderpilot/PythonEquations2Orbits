#%%
# convergence of fourth order finite differences
# for various N, setup a grid in [-pi, pi] and function u(x)
import math
import scipy.sparse as sparse
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as linalg
# p3.m - band-limited interpolation

h = 1
xmax = 10
x = np.array(range(-xmax,xmax,h))            # computational grid
xx = -xmax+np.array(range(h/20,h/10,xmax+h/20))       # plotting grid
for n in range(1,3+1):
    plt.subplot(4,1,n)
    if n== 1:
        v = (x==0);               # delta function
    elif n== 2:
        v = (abs(x)<=3);          # square wave
    elif 3:
        v = max(0.0,1-abs(x)/3);    # hat function
    
    plt.plot(x,v,'.','markersize',14), 
    plt.grid=True
    p = np.zeros(len(xx))
    for i in range(0, len(x)):
        p = p + v(i)*np.sin(math.pi*(xx-x[i])/h)/(math.pi*(xx-x[i]/h))

plt.line(xx,p)
#plt.axis([-xmax xmax -.5 1.5])
#plt.set(gca,'xtick',[])
#plt.set(gca,'ytick',[0 1])
