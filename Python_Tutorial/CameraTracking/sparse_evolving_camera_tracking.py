#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:27:27 2023

@author: chloe
"""

# Import defaults
import os, sys, copy
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import scipy.io as sio

from scipy.integrate import odeint
from typing import Dict, Callable

# Predefined functions 
from SparseCamera import *

#bkwd_time_1, 2, 3 should be from 30 to 0 with dt=-0.1

# Time parameters
ti = 0
tf = 40
dt = 0.1    # time increment
#t0 = 0      # initial time
#t1 = 20     # final time

# Gyre parameters
A       = 0.1
epsilon = 0.1
omega   = 2*np.pi/10

# Camera parameters
# right side
#starting_point_x = 1.5
#starting_point_y = 0.3

# left side
#starting_point_x = 0.3
#starting_point_y = 0.7

starting_point_x = 1.125
starting_point_y = 0.2

# big camera
#width            = 0.4
#height           = 0.2

# small camera
width            = 0.2
height           = 0.1

saveFigs = 'yes'
saveVars = 'yes'


def findMax(lavd_field, x_mesh, y_mesh, xlow, xhigh):
    
    # Use boolean indexing to find points within bounds
    within_bounds = ((x_mesh[0,:] >= xlow) & (x_mesh[0,:] <= xhigh))    
    
    x_mesh = x_mesh[0,within_bounds]
    y_mesh = y_mesh[0,within_bounds]
    
    j         = np.unravel_index(lavd_field[0,within_bounds].argmax(), lavd_field[0,within_bounds].shape)
    
    x_max     = x_mesh[j]
    y_max     = y_mesh[j]
    field_max = lavd_field[0,j]
    
    return x_max, y_max, field_max


def createMesh(x,y):
    "Create a grid for the x and y values to plot the field of interest"
    
    # Then, make the mesh grid and flatten it to get a single vector of positions.  
    mesh   = np.meshgrid(x, y)
    x_mesh = mesh[0]
    y_mesh = mesh[1]
    
    return x_mesh, y_mesh

def main(A, omega, epsilon, ti, tf, dt, width, height):
    
    # initialize lavd_max
    conv = 1
    
    convs = [1]
    
    iter_num = 0
    
    interval = 201 #75
    
    # bounds for the max search of the entire field
    xlow  = 1
    xhigh = 2

    for frame in np.arange(0,interval,1):
        
        # Load all the variables
        #load_file = '../GeometricLCS/saveSparse/fwd_time_1/fields_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_frame'+str(frame)+'.mat'
        load_file = '../GeometricLCS/saveSparse/fwd_time_1000/fields_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_frame'+str(frame)+'.mat'
        f = sio.loadmat(load_file)
        
        x_mesh = f['gyre_x']
        y_mesh = f['gyre_y']
        lavd_field = f['lavd_field']
        
        #x_mesh2, y_mesh2 = createMesh(x[0,201:400],y)
        lavd_x_max, lavd_y_max, field_max = findMax(lavd_field, x_mesh, y_mesh, xlow, xhigh)
        
        # Plot the figures
        fig, ax = plt.subplots(1, 1) #constrained_layout=True)

        # LAVD figure
        ax.axis('scaled')
        ax.set_xlim([0,2])
        ax.set_ylim([0,1])
        
        sc = ax.scatter(x_mesh, y_mesh, s=15, c=lavd_field) 
        
        sc.set_clim([0, np.nanmax(lavd_field)])
        
        j = np.unravel_index(lavd_field.argmax(), lavd_field.shape)
        #t0 = 0
        #t1 = -30
        ax.set_title('LAVD at $A$='+str(A)+', $\epsilon$='+str(epsilon)+', $\omega$=$\pi$/5, $t$='+str(round(ti+frame*dt,2))+' to '+str(round(frame*dt+ti+interval*dt-dt,2))+', $dt$='+str(dt))
        
        ax.plot( lavd_x_max, lavd_y_max, marker="d", markersize=14, markeredgecolor="cyan", markerfacecolor="cyan" )
        
        if iter_num == 0:
            camera = SparseCamera(starting_point_x, starting_point_y, width, height, x_mesh, y_mesh, lavd_field)
        else:
            camera = SparseCamera(camera.x, camera.y, width, height, x_mesh, y_mesh, lavd_field)
            
        # Plot the camera center
        ax.plot(camera.x, camera.y, marker="x", markersize=16, markeredgecolor="red", markerfacecolor="red")
        
        # Get the bottom left corner
        rect_left   = camera.left
        rect_bottom = camera.bottom
        
        # Plot the camera center and rectangle
        ax.add_patch( Rectangle((rect_left, rect_bottom),
                            width, height,
                            fc ='none',
                            ec ='r',
                            lw = 2) )
        
        # Record the LAVD field
        camera_mesh_x, camera_mesh_y, camera_mesh_lavd = camera.getFrame()
        
        x_max, y_max, lavd_max = camera.find_maximum(camera_mesh_x, camera_mesh_y, camera_mesh_lavd)
        
        camera.move_camera(x_max, y_max)

        # Plot the maximum value found within the camera
        ax.plot(camera.x, camera.y, marker="P", markersize=10, markeredgecolor="darkgreen", markerfacecolor="darkgreen")

        # Save figure
        if saveFigs == 'yes':
            plt.savefig('./sparse/track_plots/fwd_time_1000/camera_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_frame'+str(round(frame,2))+'.png', format='png')
        
        plt.show()
        
        conv = abs(lavd_max - field_max) / field_max
        
        convs.append( conv )
        
        #if iter_num > 0:
        #    if convs[iter_num] == convs[iter_num-1]:
        #        break
        
        if saveVars == 'yes':
            save_y0_file = './sparse/track_info/fwd_time_1000/camera_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_frame'+str(round(frame,2))+'.mat'
            sio.savemat(save_y0_file, {'camera_x':camera.x, 'camera_y':camera.y, 'x_mesh':camera_mesh_x, 'y_mesh':camera_mesh_y, 'lavd_mesh':camera_mesh_lavd})

        iter_num += 1
        
    
if __name__ == "__main__":
    
    main(A, omega, epsilon, ti, tf, dt, width, height)