#%%
#type: ignore
import sympy as sy


t = sy.Symbol('t')
t2 = sy.Symbol('t')

print(str(t==t))
somedict = {}
somedict[t] = "5"

someExp = t*t+5
someExp2 = t2*t2+5

print(str(someExp==someExp2))

somedict[someExp] = 6
print(str(somedict[someExp2]))