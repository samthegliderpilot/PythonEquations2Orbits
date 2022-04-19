# Python Optimization with Non-Linear Programming

A library to explore non-linear programming for small-to-medium scale optimization problems.  After a lot of studying and false starts, I finally decided that it shouldn't be THAT hard to do create a simple transcription and optimization routine for an optimal control problem.  Additional, after finding Matthew Kelly's very helpful paper [An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation](https://epubs.siam.org/doi/pdf/10.1137/16M1062569) I sat down and started to create this project.

The goal with this project is to create an easy-to-understand set of types to assist with solving optimal control problems with direct methods. Performance is nice, but it is a secondary goal to being able to understand what is going on and to be able to stop the debugger at key points to inspect various aspects of the NLP being solved. Preserving that transparency and keeping the code as simple and pure-python as possible means that this will never be as performant as GPOPS or other commercial offerings.

This is clearly a work in progress, interfaces will almost certainly change and there is no real verification or validation (but there is a thorough set of automated tests).

### Near-term future work
This is clearly a lot for me to learn about techniques for NLP techniques for optimal control problems.  In the near term I am adding the following:
- More integration schemes than just the trapezoidal rule
- Mesh refinement
- More complicated problems
- Supporting using pygmo optimizers and parallel routines
- Sparse matrix implementations for problems

### About Me

I am a principle systems engineer working on planning missions and flying satellites for a US aerospace company.  Most of the time I'm writing software in C# and VB.net to assist with maneuver planning for those satellite missions (alas, it is more the software around the numerical algorithms than the astronautics itself, but on a good day I get to actually do some math).  Fluent in C#, decent at Python and VB.net, curious about C++, and I will begrudgingly use MATLAB if I have to.
