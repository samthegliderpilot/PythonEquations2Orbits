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
syFunctions = ['cos', 'sin', 'tan', 'exp', 'log', 're', 'im', 'Abs'] # this list might need to grow
tStr = "t"
t0Str = sy.Symbol("t_0", real=True)
tfStr = sy.Symbol("t_f", real=True)

t = sy.symbols(tStr)

def isRunningJupyter():
    """
    Returns True if this code is being run in a Jupyter notebook. False otherwise.
    """
    import __main__ as main
    return not hasattr(__builtins__, '__IPYTHON__')

#https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode
def isInInteractiveMode() :
    """
    Returns True if being run in interactive mode (Jupyter, VS Code Interactive...).  
    False otherwise. Note that there are multiple ways to figure this out, but according 
    to a stack-overflow link, this is the best way to do it.
    """
    return hasattr(sys, 'ps1')

def printMarkdown(markdown : str) -> None :
    """
    Prints the passed in string as a Markdown string when run in an interactive mode.
    Otherwise just prints the string.
    """
    if (not silent):
        if(isInInteractiveMode()) :
            display(Markdown(markdown))
        else :
            print(markdown)

def clean(equ) :
    """
    Cleans the passed in sympy expression.  This will remove the (t) that is common in many expressions I've worked with.
    """
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

def showEquation(lhsOrEquation, rhs=None, cleanEqu=defaultCleanEquations) :    
    """
    Shows the equation.  The first item is a sympy equation and no rhs will be given.  It can also be a string or number but the rhs 
    must be specified in that case.
    """
    def shouldIClean(side) :
        return (isinstance(side, sy.Function) or 
                isinstance(side, sy.Derivative) or 
                isinstance(side, sy.Add) or 
                isinstance(side, sy.Mul) or 
                isinstance(side, sy.MutableDenseMatrix) or 
                isinstance(side, sy.Matrix) or 
                isinstance(side, sy.ImmutableMatrix))

    realLhs = lhsOrEquation
    realRhs = rhs
    if(isinstance(lhsOrEquation, sy.Eq)) :
        realLhs = lhsOrEquation.lhs
        realRhs = lhsOrEquation.rhs
    if(isinstance(lhsOrEquation, str)) :
        if(isinstance(rhs, sy.Matrix) or 
           isinstance(rhs, sy.ImmutableMatrix)):
            realLhs = sy.MatrixSymbol(lhsOrEquation, 
                                   rhs.shape[0], 
                                   rhs.shape[1])            
        else:
            realLhs = sy.symbols(lhsOrEquation)
    if(isinstance(rhs, str)) :
        if(isinstance(lhsOrEquation, sy.Matrix) or 
           isinstance(lhsOrEquation, sy.ImmutableMatrix)):
            realRhs = sy.MatrixSymbol(rhs, 
                                   lhsOrEquation.shape[0], 
                                   lhsOrEquation.shape[1])
        else:
            realRhs = sy.symbols(rhs)
       
    if(cleanEqu and shouldIClean(realRhs)) : 
        realRhs = clean(realRhs)
    if(cleanEqu and shouldIClean(realLhs)) : 
        realLhs = clean(realLhs)

    if(not silent) :
        if(isinstance(realLhs, sy.Eq) and realRhs == None) :
            display(realLhs) 
        else :
            display(sy.Eq(realLhs, realRhs))

import subprocess
from os import listdir, unlink, remove, walk, rmdir
from os.path import isfile, join, basename, dirname, splitext, realpath

class ReportGeneratorFromPythonFileWithCells :      
    @staticmethod
    def WriteIpynbToDesiredFormatWithPandoc(pythonFilePath, outputFilePath = None, extension = "pdf", sources = None, csl=None, keepDirectoryClean = True) :
        if outputFilePath != None :
            extension = splitext(pythonFilePath)[1]
        else :
            outputFilePath = pythonFilePath.replace(".py", "."+extension)
        ipynbFile = pythonFilePath.replace(".py", ".ipynb")
        mdFileName = pythonFilePath.replace(".py", ".md")
        directory = dirname(pythonFilePath)
        
        if sources == None or csl == None:
            files = [f for f in listdir(directory) if isfile(join(directory, f))]
            if sources == None :
                for file in files :
                    if file.endswith("bib") :
                        sources = file
                        break

            if csl == None :
                for file in files :
                    if file.endswith("csl") :
                        csl = file
                        break

        
        with CleanDirectoryScope(directory, [basename(outputFilePath)], keepDirectoryClean) :
            if not ScopeIfFileDoesNotExist.isFileControlledByScope(ipynbFile) :
                ReportGeneratorFromPythonFileWithCells.ConvertPythonToJupyter(pythonFilePath, directory)
                jupyterCommand = "jupyter nbconvert --execute --to markdown --no-input " + ipynbFile
                ReportGeneratorFromPythonFileWithCells.runCommandPrintingOutput(jupyterCommand)
                ReportGeneratorFromPythonFileWithCells.RemoveSinglePercentLinesFromFile(mdFileName)
                pandocCommand = "pandoc " + mdFileName + " -s -N -o " + outputFilePath +" --citeproc --bibliography=" + join(directory, sources) + " --csl=" + join(directory, csl)
                ReportGeneratorFromPythonFileWithCells.runCommandPrintingOutput(pandocCommand, directory)
                if not isfile(outputFilePath) :
                    raise Exception("File was not created sucessfully")

    @staticmethod
    def RemoveSinglePercentLinesFromFile(filePath) :
        # note that the p2j routine leaves in a % for each #%% cell in the original py file
        # this pair-of-loops here removes those lines
        with open(filePath, "r") as f:
            lines = f.readlines()
        with open(filePath, "w") as f:
            for line in lines:
                if line.strip("\n") != "%":
                    f.write(line)

    @staticmethod
    def ConvertPythonToJupyter(pythonFileToConvert, workingDirectory = None):
        # I would like to someday replace p2j since it does a few things that are frustrating
        # in my workflow (turning comment lines into markdown is the big one)
        if workingDirectory == None :
            workingDirectory = dirname(pythonFileToConvert)
        ReportGeneratorFromPythonFileWithCells.runCommandPrintingOutput('p2j ' + pythonFileToConvert + ' -o', workingDirectory)

    @staticmethod
    def runCommandPrintingOutput(command, workingDirectory = None) :
        if workingDirectory == None :
            workingDirectory = dirname(dirname(realpath(__file__)))
        result = subprocess.run(command, capture_output=True, text=True, cwd=workingDirectory)
        if(len(result.stderr.strip()) > 0):
            print(result.stderr)
            print(result.stdout)            

class CleanDirectoryScope :
    """
    A scope that will record the contents of a directory upon entry, and 
    on exit will delete all new files and new, empty, directories.  When working with a process 
    where a bunch of extra files might clutter up an otherwise well manicured directory, 
    this is a easy way to keep that directory clean.
    """
    def __init__(self, directory : str, localNewFilesToKeep : List[str] = [], keepDirectoryClean = True) :
        for file in localNewFilesToKeep :
            self.localNewFilesToKeep = join(directory, file)
        self.directory = directory
        self.keepDirectoryClean = keepDirectoryClean

    def __enter__(self) :
        self.filesInDirectory, self.directories = self.getFilesAndDirectoriesInDirectory()
        return self

    def __exit__(self, exc_type, exc_value, tb) :        
        if not self.keepDirectoryClean :
            return
        filesAtEnd, directoriesAtEnd = self.getFilesAndDirectoriesInDirectory()
        for file in filesAtEnd :
            if not file in self.filesInDirectory and not file in self.localNewFilesToKeep :
                if isfile(file) :
                    remove(file)
                    
        for dir in directoriesAtEnd :
            if not dir in self.directories and len(listdir(dir)) == 0: # since we already removed files, only delete dir if empty
                rmdir(dir)

    def getFilesAndDirectoriesInDirectory(self) -> List[List[str]] :
        files = []
        directories = []
        for (dirpath, dirnames, filenames) in walk(self.directory) :
            for dir in dirnames :
                directories.append(join(dirpath, dir))
            for file in filenames :
                files.append(join(dirpath, file))
        return [files, directories]

import uuid
class ScopeIfFileDoesNotExist :
    """
    Creates a scope based on the existence of a file.  What happened is when I want a 
    script to make a PDF of itself, the script might need to get rerun, which means 
    the 'create pdf' part of the code runs again.  
    """
    # however, with the need to know the exact file location, this is less useful than 
    # it was
    scopedFiles = []

    def __init__(self, directory, fileName = None) :
        self.directory = directory
        if fileName == None :
            fileName = str(uuid.uuid4())
        self.fileName =join(self.directory, fileName)
    
    def __enter__(self) :        
        if isfile(self.fileName) :
            with open(self.fileName, 'w') :
                pass
        ScopeIfFileDoesNotExist.scopedFiles.append(self.fileName)
        return self

    def __exit__(self, exc_type, exc_value, tb) :
        ScopeIfFileDoesNotExist.scopedFiles.remove(self.fileName)
        if not self.fileAlreadyExists and isfile(self.fileName) :
            remove(self.fileName)
            
    @staticmethod
    def isFileControlledByScope(filepath) :
        return filepath in ScopeIfFileDoesNotExist.scopedFiles


    
