#%%
import __init__ #type: ignore
import sympy as sy
import math
import os
import sys
from IPython.display import display
from collections import OrderedDict
#sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0]))) # need to import 2 directories up (so pyeq2orb is a subfolder)
sy.init_printing()
import scipyPaperPrinter as jh #type: ignore
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from pyeq2orb.Coordinates.CartesianModule import MotionCartesian
from pyeq2orb.Coordinates.QuaternionModule import MotionQuaternion



from sympy import symbols
import sympy.physics.mechanics as syMech
n=5
q = syMech.dynamicsymbols('q:' + str(n+1), real=True) # general coordinates
u = syMech.dynamicsymbols('u:' + str(n+1), real=True) # generalized speeds
f = syMech.dynamicsymbols('f', real=True) # force applied

m = sy.symbols('m:' + str(n+1), real=True) #mass of each bob
l = sy.symbols('l:'+str(n), real=True) # length of each link
g,t = sy.symbols('g,t', real=True)

I = syMech.ReferenceFrame('I') # inertial frame
O = syMech.Point('O') # origin
O.set_vel(I, 0) # origin does not move

P0 = syMech.Point('P0')
P0.set_pos(O, q[0]*I.x) # set position of P0
P0.set_vel(I, u[0]*I.x) # set velocity of P0
Pa0 = syMech.Particle('Pa0', P0, m[0]) # a Particle at P0

