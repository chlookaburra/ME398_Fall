#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:17:30 2023

@author: chloe
"""

import os, sys, copy
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Dict, Callable

sys.path.append(os.path.join(os.getcwd(), "SparseCamera"))

class SparseCamera:
    def __init__(self, starting_point_x, starting_point_y, width, height, x_vec, y_vec, lavd_field):
        self.x = starting_point_x
        self.y = starting_point_y
        self.width = width
        self.height = height
        self.x_vec = x_vec
        self.y_vec = y_vec
        self.lavd_field = lavd_field
        
        self.data = np.concatenate((x_vec, y_vec), axis=0)

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
        
        indices = ((self.data[0,:] >= self.left) & (self.data[0,:] <= self.right) &
                         (self.data[1,:] >= self.bottom) & (self.data[1,:] <= self.top))
        
        camera_x = self.data[0,indices]
        camera_y = self.data[1,indices]
        
        camera_lavd = self.lavd_field[:, indices]

        return camera_x, camera_y, camera_lavd

    def find_maximum(self, camera_mesh_x, camera_mesh_y, camera_mesh_lavd):
        
        camera_mesh_lavd = camera_mesh_lavd.squeeze()
        
        j        = np.unravel_index(camera_mesh_lavd.argmax(), camera_mesh_lavd.shape)
        
        x_max    = camera_mesh_x[j]
        y_max    = camera_mesh_y[j]
        lavd_max = camera_mesh_lavd[j]
        
        return x_max, y_max, lavd_max
    

    def move_camera(self, x_max, y_max):
        
        self.x = x_max
        self.y = y_max
            
        return

