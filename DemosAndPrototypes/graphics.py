import sys

from vispy import scene #type: ignore
from vispy.visuals.transforms import STTransform #type: ignore

canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',
                           size=(800, 600), show=True)

view = canvas.central_widget.add_view()
view.camera = 'arcball'

sphere1 = scene.visuals.Sphere(radius=1, method='latitude', parent=view.scene,
                               edge_color='black')

sphere2 = scene.visuals.Sphere(radius=1, method='ico', parent=view.scene,
                               edge_color='black')

sphere3 = scene.visuals.Sphere(radius=1, rows=10, cols=10, depth=10,
                               method='cube', parent=view.scene,
                               edge_color='black')

sphere1.transform = STTransform(translate=[-2.5, 0, 0])
sphere3.transform = STTransform(translate=[2.5, 0, 0])

view.camera.set_range(x=[-3, 3]) #type: ignore

if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas.app.run(True)





# import pyqtgraph.opengl as gl
# from pyqtgraph.Qt import QtWidgets
# import numpy as np
# import sys

# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     w = gl.GLViewWidget()

#     xx = 0
#     yx = 0
#     zx = 0

#     xy = 1
#     yy = 0
#     zy = 0

#     XDot = (xx, yx, zx)
#     YDot = (xy, yy, zy)

#     xyx = np.array([XDot, YDot])
#     sh1 = gl.GLLinePlotItem(pos=xyz, width=2, antialias=False, color='red')
#     w.addItem(sh1)
#     w.show()
#     app.exec()

# import numpy as np 
# import pyqtgraph as pg
# from pyqtgraph.Qt import *
# import pyqtgraph.opengl as gl
# import sys


# class Simulation(): #create a class with all the following data
#     def __init__(self):
#         self.app = QtGui.QApplication(sys.argv)
#         self.window = gl.GLViewWidget()     
#         self.window.setGeometry(100, 100, 800, 600) #set the geometry of the window(x padding, y padding, dimension x, dimension y)
#         self.window.setWindowTitle("I am going to draw a line") #set the title
#         self.window.show()  #show the window

        
#         global points_list
#         points_list = [] #create an empty list
#         self.draw() #call the draw function

#     def draw(self):
#         point1 = (0, 0, 0)  #specify the (x, y, z) values of the first point in a tuple
#         point2 = (5, 6, 8)  #specify the (x, y, z) values of the second point in a tuple

#         points_list.append(point1) #add the point1 tuple to the points_list
#         points_list.append(point2) #add the point2 tuple to the points_list
#         print(points_list)
#         points_array = np.array(points_list) #convert the list to an array
#         drawing_variable = gl.GLLinePlotItem(pos = points_array, width = 1, antialias = True)   #make a variable to store drawing data(specify the points, set anti-aliasing)
#         self.window.addItem(drawing_variable) #draw the item
#         drawing_variable2 = gl.GLLinePlotItem(pos = xyz, width = 1, antialias = True)   #make a variable to store drawing data(specify the points, set anti-aliasing)
#         self.window.addItem(drawing_variable2) #draw the item

#     def start(self):
#         QtGui.QApplication.instance().exec_() #run the window properly

# Simulation().start()
# 
# 

# plot = scene.Line(xyz, parent=view.scene, color=[1.0, 0.0, 0.0, 1.0], width=3, antialias=True)
# view.camera.set_range(x=[9000000, 3])
# if __name__ == '__main__' and sys.flags.interactive == 0:
#     canvas.app.run()
# canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',
#                            size=(800, 600), show=True)

# view = canvas.central_widget.add_view()
# view.camera = 'arcball'

# sphere1 = scene.visuals.Sphere(radius=1, method='latitude', parent=view.scene,
#                                edge_color='black')

# sphere2 = scene.visuals.Sphere(radius=1, method='ico', parent=view.scene,
#                                edge_color='black')

# sphere3 = scene.visuals.Sphere(radius=1, rows=10, cols=10, depth=10,
#                                method='cube', parent=view.scene,
#                                edge_color='black')

# sphere1.transform = STTransform(translate=[-2.5, 0, 0])
# sphere3.transform = STTransform(translate=[2.5, 0, 0])

# view.camera.set_range(x=[-3, 3])

# if __name__ == '__main__' and sys.flags.interactive == 0:
#     canvas.app.run(True)
#     


#%%
import sympy as sy
from IPython.display import display

t = sy.Symbol('t', real=True)
lmdy = sy.Function(r'\\lambda_{y}', real=True)(t)
lmdx = sy.Function(r'\\lambda_{x}', real=True)(t)
x = [lmdy, lmdx]
expr = sy.asin(sy.atan2(lmdy, lmdx))

syntaxSafe = [sy.Symbol('lmdy', real=True), sy.Symbol('lmdx', real=True)]
subsDict=  dict(zip(x, syntaxSafe))
exprSafe = expr.subs(subsDict, deep=True).doit()
display(exprSafe)