# Python Equations To Orbits

A library to explore non-linear programming for small-to-medium scale optimization problems with a focus on orbital trajectories. After a lot of studying and false starts, I decided that it shouldn't be THAT hard to do create a simple transcription and optimization routine for an optimal control problem. Additional, after finding Matthew Kelly's very helpful paper [An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation](https://epubs.siam.org/doi/pdf/10.1137/16M1062569) and going through Longuski's, Guzman's and Prussing's textbook Optimal Control with Aerospace Applications, I sat down and started to create this project.

What's more, during my masters degree I became very comfortable with using sympy and taking those equations and automatically turning them into code that can be run numerically with significantly more performance than just substituting values into the symbolic expressions.  One frustration that I have with many other packages is how opaque they are to the math you are trying to work with, but this let's me work with the math symbolically when it makes sense to, and convert them to evaluating expressions at the right time.  Again, that helps with understanding the math and process.

The goal with this project is to create an easy-to-understand set of types to assist with solving optimal control problems for indirect and direct methods. Performance is nice, but it is a secondary goal to being able to understand what is going on and to be able to stop the debugger at key points to inspect various aspects of the NLP being solved. Preserving that transparency and keeping the code as simple and pure-python as possible means that this will never be as performant as GPOPS or other commercial offerings.  The goal is not to solve the problem for you, but make it easy for you to solve the problem using python tools that you are hopefully already familiar with.

This is clearly a work in progress, interfaces will almost certainly change and there is no real verification or validation.  Some of the tests are pretty weak and I have a list of things to refactor after rushing to get this good enough for a release by my personal deadline.

I have tried to make a virtual environment, however it is still something I'm figuring out.  If you want to make the environment from scratch... the way I've been setting up my environments has been with these conda/pip commands.  Note that in all cases, I have a C++ compiler on my computer (Visual Studio 2022 Community is installed on Windows, gcc on Linux).  Cmake can find it, and it is probably required for setting up pyomo all the way:

```
conda create --name py310OptFun python=3.10
conda activate py310OptFun
conda install -c conda-forge sympy numpy pandas scipy plotly matplotlib jupyter pygmo pyomo cmake pytest plotly p2j pandoc networkx openpyxl pint pymysql pyodbc pyro4 xlrd cyipopt pandoc mypy vispy spiceypy setuptools mypy pylint czml3 dill
pyomo download-extensions
conda install -c conda-forge pynumero_libraries
pyomo build-extensions
python setup.py dependencies --extra optional 
```

That last command is adding additional dependencies for pyomo, however it doesn't seem to work and probably isn't needed.  Note that the command installing pynumero_libraries may only be required for Windows.

I find that for unit tests to properly work in VS Code, I need to in the conda console, activate the environment and then start code from that console.

Note that I am trying to make this library strongly typed with MyPy.  

Note that my conversion of Jupyter notebooks requires LaTeX of some sort to be installed (on Windows, I'm using MiKTeX)

### Near-term future work
This is clearly a lot for me to learn about techniques for NLP techniques for optimal control problems.  In the near term I am adding the following:
- More complicated problems
- More manual transcription of problems 
- Supporting using pygmo optimizers and parallel routines
- Sparse matrix implementations for problems

But I want to do things that interest me, and I want to spend more time on research and aerospace stuff.  So my real TODO is:
- Mean Element Propagation with low-thrust (or acceleration)
- 6 element at time indirect optimization
- Q-Law and other techniques like that
- Guess costate values from Q-Law like solution
- CisLunar Toolbox
 - Derive analytical earth-and-moon 3BP equations

