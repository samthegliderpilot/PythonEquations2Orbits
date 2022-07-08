# Python Equations To Orbits

A library to explore non-linear programming for small-to-medium scale optimization problems with a focus on orbital trajectories.  After a lot of studying and false starts, I decided that it shouldn't be THAT hard to do create a simple transcription and optimization routine for an optimal control problem. Additional, after finding Matthew Kelly's very helpful paper [An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation](https://epubs.siam.org/doi/pdf/10.1137/16M1062569) and going through Longuski's, Guzman's and Prussing's textbook Optimal Control with Aerospace Applications, I sat down and started to create this project.

What's more, during my masters degree I became very comfortable with using sympy and taking those equations and automatically turning them into code that can be run numerically with significantly more performance than just substituting values into the symbolic expressions.  One frustration that I have with many other packages is how opaque they are to the math you are trying to work with, but this let's me work with the math symbolically when it makes sense to, and convert them to evaluating expressions at the right time.  Again, that helps with understanding the math and process.

The goal with this project is to create an easy-to-understand set of types to assist with solving optimal control problems for indirect and direct methods. Performance is nice, but it is a secondary goal to being able to understand what is going on and to be able to stop the debugger at key points to inspect various aspects of the NLP being solved. Preserving that transparency and keeping the code as simple and pure-python as possible means that this will never be as performant as GPOPS or other commercial offerings.  The goal is not to solve the problem for you, but make it easy for you to solve the problem using python tools that you are hopefully already familiar with.

This is clearly a work in progress, interfaces will almost certainly change and there is no real verification or validation.  Some of the tests are pretty weak and I have a list of things to refactor after rushing to get this good enough for a release by my personal deadline.

### Near-term future work
This is clearly a lot for me to learn about techniques for NLP techniques for optimal control problems.  In the near term I am adding the following:
- More complicated problems
- More manual transcription of problems 
- Supporting using pygmo optimizers and parallel routines
- Sparse matrix implementations for problems
