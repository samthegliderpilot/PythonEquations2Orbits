# This is a set of helper function to display pretty printed equations and markdown in a Jupyter (or Jupyter-like window like in VS Code).
# This is just for outputting purposes and I don't plan on adding tests or thorough documentation to this (for now).
from IPython.display import  Latex, display, Markdown
from pytest import mark
import sympy as sy
from sympy.printing.latex import LatexPrinter
from sympy.core import evaluate

defaultCleanEquations = True
silent = False
syFunctions = ['cos', 'sin', 'tan', 'exp', 'log', 're', 'im', 'Abs']
tStr = "t"
t0Str = sy.Symbol("t_0", real=True)
tfStr = sy.Symbol("t_f", real=True)

t = sy.symbols(tStr)

def isRunningJupyter():
    import __main__ as main
    return not hasattr(main, '__file__')

def printMarkdown(markdown : str) -> None :
    if (not silent):
        if(isRunningJupyter()) :
            display(Markdown(markdown))
        else :
            print(markdown)

def clean(equ) :
    if(equ is sy.Matrix) :
        for row in equ.sizeof(0) :
            for col  in equ.sizeof(1) :
                clean(equ[row, col])
    else:            
        for val in equ.atoms(sy.Function):
            
            dt=sy.Derivative(val, t)
            ddt=sy.Derivative(dt,t)

            # skip built in functions (add to list above)            
            clsStr = str(type(val))
            if(clsStr in syFunctions):
                continue
            
            if(hasattr(val, "name")) :
                newStr = val.name
                if t0Str in val.args :
                    newStr = newStr + "{_0}"
                elif tfStr in val.args :
                    newStr = newStr + "{_f}"
                elif t in val.args :
                    newStr = newStr# + "(t)"
            else :
                newStr = str(val)
            newDtStr = r'\dot{' +newStr +"}"
            newDDtStr = r'\ddot{' + newStr +"}"

            # newDtStr = newDtStr.replace('}_{', '_')
            # newDtStr = newDtStr.replace('}_0', '_0}')
            # newDtStr = newDtStr.replace('}_f', '_f}')
            # newDDtStr = newDDtStr.replace('}_{', '_')
            # newDDtStr = newDDtStr.replace('}_0', '_0}')    
            # newDDtStr = newDDtStr.replace('}_f', '_f}')    

            equ=equ.subs(ddt, sy.Symbol(newDDtStr))
            equ=equ.subs(dt, sy.Symbol(newDtStr))
            equ=equ.subs(val, sy.Symbol(newStr))
        return equ

def showEquation(lhs, rhs=None, cleanEqu=defaultCleanEquations) :    
    def shouldIClean(side) :
        return (isinstance(side, sy.Function) or 
                isinstance(side, sy.Derivative) or 
                isinstance(side, sy.Add) or 
                isinstance(side, sy.Mul) or 
                isinstance(side, sy.MutableDenseMatrix) or 
                isinstance(side, sy.Matrix) or 
                isinstance(side, sy.ImmutableMatrix))

    # this isn't working yet...
    #def round_expr(expr, num_digits):
    #    return expr.replace({n : round(n, num_digits) for n in expr.atoms(sy.Number)})

    realLhs = lhs
    realRhs = rhs
    if(isinstance(lhs, sy.Eq)) :
        realLhs = lhs.lhs
        realRhs = lhs.rhs
    if(isinstance(lhs, str)) :
        if(isinstance(rhs, sy.Matrix) or 
           isinstance(rhs, sy.ImmutableMatrix)):
            realLhs = sy.MatrixSymbol(lhs, 
                                   rhs.shape[0], 
                                   rhs.shape[1])            
        else:
            realLhs = sy.symbols(lhs)
    if(isinstance(rhs, str)) :
        if(isinstance(lhs, sy.Matrix) or 
           isinstance(lhs, sy.ImmutableMatrix)):
            realRhs = sy.MatrixSymbol(rhs, 
                                   lhs.shape[0], 
                                   lhs.shape[1])
        else:
            realRhs = sy.symbols(rhs)

    # if(isinstance(realLhs, SymbolWithValue) and unit == None) :
    #     unit = realLhs.Unit
        
    if(cleanEqu and shouldIClean(realRhs)) : 
        realRhs = clean(realRhs)
    if(cleanEqu and shouldIClean(realLhs)) : 
        realLhs = clean(realLhs)

    if(not silent) :
        if(isinstance(realLhs, sy.Eq) and realRhs == None) :
            display(realLhs) 
        else :
            display(sy.Eq(realLhs, realRhs))
