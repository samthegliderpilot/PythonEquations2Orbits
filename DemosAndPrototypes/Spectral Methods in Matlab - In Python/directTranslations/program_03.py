#%%
# convergence of fourth order finite differences
# for various N, setup a grid in [-pi, pi] and function u(x)
import math
from matplotlib import pyplot as plt
import numpy as np
# p3.m - band-limited interpolation

h = 1
xmax = 10
x = np.array(range(-xmax,xmax+1,h))      # computational grid
xx = np.linspace(-10.05, 10.05, num=202) # plotting grid

for n in range(1,3+1):
    plt.subplot(4,1,n)
    if n== 1:
        v = (x==0);               # delta function
    elif n== 2:
        v = (abs(x)<=3);          # square wave
    elif n==3:
        v = [max(0.0,1-abs(x2)/3) for x2 in x];    # hat function   
    plt.plot(x,v,'.',markersize =14)
    plt.grid(visible= True, axis="y", ls="--", color='0.65')
    p = np.zeros(len(xx))
    for i in range(0, len(x)):
        p = p + v[i]*np.sin(math.pi*(xx-x[i])/h)/(math.pi*(xx-x[i]/h))
    plt.plot(xx,p)
    plt.axis = [-xmax, xmax, -.5, 1.5]
    plt.xtick = []
    plt.ytick = [0.0, 1.0]