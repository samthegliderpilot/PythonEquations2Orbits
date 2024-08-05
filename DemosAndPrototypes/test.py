#%%
from pyomo.dae import ContinuousSet, DerivativeVar #type: ignore
from pyomo.environ import ConcreteModel, TransformationFactory, Var, NonNegativeReals, Constraint, SolverFactory, Objective, cos, sin, minimize, NonNegativeReals #type: ignore
import numpy as np
import matplotlib.pyplot as plt#type: ignore

# Define parameters of the problem
h = 185.2e3 # meters, final altitude (100 nmi circular orbit)
Vc = 1.627e3 # m/s, Circular speed at 100 nmi
g_accel = 1.62 # m/sˆ2, gravitational acceleration of Moon
Thrust2Weight = 3 # Thrust to Weight ratio for Ascent Vehicle, in lunar G's
F = Thrust2Weight * g_accel

model = ConcreteModel("rocket")
model.T = Var(domain=NonNegativeReals)
model.t = ContinuousSet(bounds=(0, 1))
model.x = Var(model.t, domain=NonNegativeReals)
model.y = Var(model.t, domain=NonNegativeReals)
model.xdot = DerivativeVar(model.x, wrt=model.t, domain=NonNegativeReals)
model.xdoubledot = DerivativeVar(model.xdot, wrt=model.t)
model.ydot = DerivativeVar(model.y, wrt=model.t, domain=NonNegativeReals)
model.ydoubledot = DerivativeVar(model.ydot, wrt=model.t)
model.alpha = Var(model.t, bounds=(-np.pi, np.pi))

# Dynamics
model.xode = Constraint(model.t, rule=lambda m, t: m.xdoubledot[t] == (F*cos(m.alpha[t]))*m.T**2)
model.yode = Constraint(model.t, rule=lambda m, t: m.ydoubledot[t] == (F*sin(m.alpha[t]) - g_accel)*m.T**2)

# Boundary conditions at the initial time
model.x[0].fix(0)
model.y[0].fix(0)
model.xdot[0].fix(0)
model.ydot[0].fix(0)

# Boundary conditions at the final time
model.y[1].fix(h)
# Since our velocity state is dx/dtau instead of dx/dt, we must set it to be the final value of dx/dt (which is Vc)
# multiplied by the final time. Technically ydot needs this as well but fortunately ydotfinal is 0.
model.xdottf = Constraint(model.t, rule=lambda m, t: Constraint.Skip if t != m.t.last() else m.xdot[t] == Vc*m.T)
model.ydot[1].fix(0)


discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(model, wrt=model.t, nfe=30, ncp=6)
# reduce_collocation_points comes from "pyomo.dae: A Modeling and Automatic Discretization Framework
# for Optimization with Differential and Algebraic Equations"
discretizer.reduce_collocation_points(model, var=model.alpha, ncp=1, contset=model.t)

model.obj = Objective(expr=model.T, sense=minimize)

solver = SolverFactory('cyipopt')
results = solver.solve(model)

tf_direct = model.T()
tdirect = [t*tf_direct for t in model.t]
alphadirect = [model.alpha[t]() for t in model.t]
xdirect = [model.x[t]()/1000 for t in model.t]
ydirect = [model.y[t]()/1000 for t in model.t]
xdotdirect = [model.xdot[t]()/tf_direct/1000 for t in model.t]
ydotdirect = [model.ydot[t]()/tf_direct/1000 for t in model.t]
xdoubledotdirect = [model.xdoubledot[t]() for t in model.t]
ydoubledotdirect = [model.ydoubledot[t]() for t in model.t]

fig = plt.figure()
fig.suptitle('Optimal Ascent from Flat Moon via Direct Optimization')

ax = fig.add_subplot(321)
ax.plot(tdirect, xdirect)
ax.set_title('x, km')
ax.set_xlabel('Time, sec')
ax.grid()

ax = fig.add_subplot(322)
ax.plot(tdirect, ydirect)
ax.set_title('y, km')
ax.set_xlabel('Time, sec')
ax.grid()

ax = fig.add_subplot(323)
ax.plot(tdirect, xdotdirect)
ax.set_title(r'$V_x$, km/s')
ax.set_xlabel('Time, sec')
ax.grid()

ax = fig.add_subplot(324)
ax.plot(tdirect, ydotdirect)
ax.set_title(r'$V_y$, km/s')
ax.set_xlabel('Time, sec')
ax.grid()

ax = fig.add_subplot(325)
ax.plot(tdirect, np.rad2deg(alphadirect), 'b+')
ax.set_title(r'$\alpha$, deg')
ax.set_xlabel('Time, sec')
ax.grid()

plt.tight_layout()
plt.show()