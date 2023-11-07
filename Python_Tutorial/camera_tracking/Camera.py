import os, sys, copy
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Dict, Callable

sys.path.append(os.path.join(os.getcwd(), "Camera"))

class Camera:
    def __init__(self, starting_point_x, starting_point_y, width, height, x_vec, y_vec, lavd_field):
        self.x = starting_point_x
        self.y = starting_point_y
        self.width = width
        self.height = height
        self.x_vec = x_vec
        self.y_vec = y_vec
        self.lavd_field = lavd_field

    @property
    def left(self):
        return self.x - self.width/2

    @property
    def right(self):
        return self.x + self.width/2

    @property
    def top(self):
        return self.y + self.height/2

    @property
    def bottom(self):
        return self.y - self.height/2
    
    def getFrame(self):
        
        # find the indices corresponding to the camera box
        x_ind = np.argwhere(np.logical_and(self.x_vec>=self.left,   self.x_vec<=self.right))
        y_ind = np.argwhere(np.logical_and(self.y_vec>=self.bottom, self.y_vec<=self.top))
        
        mesh          = np.meshgrid(self.x_vec[:,x_ind[:,1]], self.y_vec[:,y_ind[:,1]])
        camera_mesh_x = mesh[0]
        camera_mesh_y = mesh[1]
        
        mesh = np.meshgrid(x_ind[:,1], y_ind[:,1])
        
        camera_mesh_lavd = self.lavd_field[mesh[1], mesh[0]]

        return camera_mesh_x, camera_mesh_y, camera_mesh_lavd

    def find_maximum(self, camera_mesh_x, camera_mesh_y, camera_mesh_lavd):
        # for i in range(self.left, self.right):
        #     for j in range(self.top, self.bottom):
        #         # do some sort of maximum search
        j = np.unravel_index(camera_mesh_lavd.argmax(), camera_mesh_lavd.shape)
        x_max  = camera_mesh_x[j]
        y_max  = camera_mesh_y[j]
        lavd_max = camera_mesh_lavd[j]
        
        return x_max, y_max, lavd_max
    

    def move_camera(self, x_max, y_max):
        
        if x_max - self.width/2 <= np.argmin(self.x_vec):

            self.x = self.x
            
        elif x_max + self.width/2 >= np.argmax(self.x_vec):

            self.x = self.x
        
        elif y_max - self.height/2 <= np.argmin(self.y_vec):

            self.y = self.y
        
        elif y_max + self.height/2 <= np.argmin(self.y_vec):
            
            self.y = self.y
            
        else:

            self.x = x_max
            self.y = y_max
            
        return
        

"""

class Camera:

    def __init__(self, x, y, xMin, xMax, yMin, yMax, mesh, lavd_field):
        
        
        self.x          = x
        self.y          = y
        self.xMin       = xMin
        self.xMax       = xMax
        self.yMin       = yMin
        self.yMax       = yMax
        self.mesh       = mesh
        self.lavd_field = lavd_field
        
        
    # Obtains the camera frame (specified by the user)
    def getFrame(lavd_field, x, y, xMin, xMax, yMin, yMax):
        
        xind1 = find(x >= xMin)
        xind2 = find(x<= xMax)
        xind = intersect(xind1, xind2)
        yind1 = find(y >= yMin)
        yind2 = find(y <= yMax)
        yind = intersect(yind1,yind2)
    
        xF = x(inds)
        yF = y(inds)
        F = lavd_field(inds)
        
        return xF, yF, F


    # Calculates the point of maximum
    def findMax(lavd_field, mesh):
        
        j      = np.unravel_index(lavd_field.argmax(), lavd_field.shape)
        x_mesh = mesh[0]
        x_max  = x_mesh[j]
        y_mesh = mesh[1]
        y_max  = y_mesh[j]
        
        lavd_max = lavd_field(j)
        
        return x_max, y_max
        
    
    # Centers the camera at the maximum point
    def moveCamera(xP, yP, ):
        
        
    
        return 

"""