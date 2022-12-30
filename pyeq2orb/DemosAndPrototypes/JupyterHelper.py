# This is a set of helper function to display pretty printed equations and markdown in a Jupyter (or Jupyter-like window like in VS Code).
# This is just for outputting purposes and I don't plan on adding tests or thorough documentation to this (for now).
from IPython.display import  Latex, display, Markdown
import sympy as sy
from sympy.printing.latex import LatexPrinter
from sympy.core import evaluate
import sys
from typing import List
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

#https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode
def isInInteractiveMode() :
    return hasattr(sys, 'ps1')

def printMarkdown(markdown : str) -> None :
    if (not silent):
        if(isInInteractiveMode()) :
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



# import subprocess
# subprocess.run('p2j ModifiedEquinoctialElementsExplination.py -o')
# #subprocess.run("jupyter nbconvert --execute ModifiedEquinoctialElementsExplination.ipynb")
# subprocess.run("jupyter nbconvert --execute --to pdf ModifiedEquinoctialElementsExplination.ipynb")
# #next convert markdown to ms word
# #conversionCommand = 'pandoc -s ModifiedEquinoctialElementsExplination.md -t docx -o ModifiedEquinoctialElementsExplination.docx --filter pandoc-citeproc --bibliography="sources.bib" --csl="apa.csl"'
# #subprocess.run(conversionCommand)
import subprocess
class ReportGeneratorFromPythonFileWithCells :
    def __init__(self, directory, pythonFileName, outputFileName) :
        self.directory = directory
        self.pythonFileName = pythonFileName
        self.pythonFilePath = join(self.directory, self.pythonFileName)
        self.outputFileName = outputFileName
        self.outputFilePath = join(self.directory, self.outputFileName)

    @property
    def baseFilePathAsIpynb(self) -> str :
        return join(self.directory, self.pythonFilePath.replace(".py", ".ipynb"))

    @property
    def baseFileNameAsIpynb(self) -> str :
        return self.pythonFileName.replace(".py", ".ipynb")

    @property
    def baseFilePathAsMarkdown(self) -> str :
        return join(self.directory, self.pythonFilePath.replace(".py", ".md"))

    @property
    def baseFileNameAsMarkdown(self) -> str :
        return self.pythonFileName.replace(".py", ".md")

    def WritePdfDirectlyFromJupyter(self) :
        markerFileName = self.baseFileNameAsIpynb
        with FileMarker(self.directory, markerFileName) as fm:
            if not fm.fileAlreadyExists: 
                subprocess.run('p2j ' + self.pythonFileName + ' -o', cwd = self.directory)
                subprocess.run("jupyter nbconvert --execute --to pdf --no-input " + self.baseFileNameAsIpynb, cwd = self.directory)
            #jupyter nbconvert --execute --to pdf --no-input
    # def WritePdfWithMarkdownInTheMiddile(self) :
    #     pass

    # def _executeCommandsToMakePdf(self, commands : List[str]) :
    #     # first, convert to jupyter
    #     # then the ipynb must get executed BUT, if this command is at the end of the ipynb, we can't let it run the same 
    #     # way over again, otherwise it will execute the command again...
        
    #     subprocess.run('p2j ModifiedEquinoctialElementsExplination.py -o')
    #     subprocess.run("jupyter nbconvert --execute --to pdf ModifiedEquinoctialElementsExplination.md")


from os import listdir, unlink, remove
from os.path import isfile, join, basename
import os
import glob

class FileScope :
    def __init__(self, directory : str, localNewFilesToKeep : List[str] = []) :
        self.localNewFilesToKeep = localNewFilesToKeep
        self.directory = directory
        return self

    def __enter__(self) :
        self.filesInDirectory = []
        for filename in glob.iglob(self.directory, recursive=True) :
            self.filesInDirectory.append(filename)
        return self

    def __exit__(self, exc_type, exc_value, tb) :
        itemsAtEnd = listdir(self.directory)
        for item in itemsAtEnd :
            fileNameOfItem = basename(item)
            if not item in self.filesInDirectory and not fileNameOfItem in self.localNewFilesToKeep :
                if isfile(item) :
                    remove(item)
import uuid
class FileMarker :
    def __init__(self, directory, fileName = None) :
        self.directory = directory
        if fileName == None :
            fileName = str(uuid.uuid4())
        self.fileName =join(self.directory, fileName)
        self.fileAlreadyExists = isfile(self.fileName)
    
    def __enter__(self) :        
        if not self.fileAlreadyExists :
            f = open(self.fileName, 'w')
            f.close()
        return self

    def __exit__(self, exc_type, exc_value, tb) :
        if not self.fileAlreadyExists and isfile(self.fileName) :
            remove(self.fileName)

    
