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
from Camera import *

# Time parameters
t0 = 0      # initial time
t1 = 20     # final time
dt = 0.1    # time increment

# Gyre parameters
A       = 0.1
epsilon = 0.1
omega   = 2*np.pi/10

# Camera parameters
#starting_point_x = 1.5
#starting_point_y = 0.3
#starting_point_x = 1.125
#starting_point_y = 0.2
starting_point_x = 1.05
starting_point_y = 0.95
width            = 0.4 #0.2
height           = 0.2 #0.1

def findMax(field_interest, mesh):
    
    j      = np.unravel_index(field_interest.argmax(), field_interest.shape)    
    x_mesh = mesh[0]
    x_max  = x_mesh[j]
    y_mesh = mesh[1]
    y_max  = y_mesh[j]
    
    field_max = field_interest[j]
    
    return x_max, y_max, field_max


def createMesh(x,y):
    "Create a grid for the x and y values to plot the field of interest"
    
    # Then, make the mesh grid and flatten it to get a single vector of positions.  
    mesh   = np.meshgrid(x, y)
    x_mesh = mesh[0]
    y_mesh = mesh[1]
    
    return x_mesh, y_mesh

def main(A, omega, epsilon, t0, t1, dt, width, height):

    # Load all the variables
    load_file = '/Users/chloe/Documents/Stanford/ME398_Fall/Python_Tutorial/saveExplore/paramexplore/fields_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(round(omega,2))+'_t'+str(t1)+'.mat'
    f = sio.loadmat(load_file)
    x = f['x']
    y = f['y']
    lavd_field = f['lavd_field']
    
    # Create a mesh for the x and y values
    x_mesh, y_mesh = createMesh(x,y)
    lavd_x_max, lavd_y_max, field_max = findMax(lavd_field[:, 0:200], [x_mesh, y_mesh])
    #x_mesh2, y_mesh2 = createMesh(x[0:200],y)
    #lavd_x_max, lavd_y_max, field_max = findMax(lavd_field[:,0:200], [x_mesh2, y_mesh2])
    
    
    #print('x_vec:'+str(np.shape(x)))
    #print('y_vec:'+str(np.shape(y)))
    #print('lavd:'+str(np.shape(lavd_field)))
    
    # initialize lavd_max
    conv = 1
    
    convs = [1]
    
    iter_num = 0

    while conv >= 1e-6:
        
        # Plot the figures
        fig, ax = plt.subplots(1, 1) #constrained_layout=True)
    
        # LAVD figure
        ax.axis('scaled')
        ax.set_xlim([0,2])
        ax.set_ylim([0,1])
        h1 = ax.pcolormesh(x_mesh, y_mesh, lavd_field)
        j = np.unravel_index(lavd_field.argmax(), lavd_field.shape)
        ax.set_title('LAVD at $A$='+str(A)+', $\epsilon$='+str(epsilon)+', $\omega$=$\pi$/5, $t$='+str(t0)+' to '+str(t1)+', $dt$='+str(dt)+', Iteration='+str(iter_num))
        im_ratio = x.shape[0]/x.shape[1]
        #cbar1 = fig.colorbar(h1, fraction=0.046*im_ratio)
        
        ax.plot( lavd_x_max, lavd_y_max, marker="d", markersize=14, markeredgecolor="cyan", markerfacecolor="cyan" )
        
        if iter_num == 0:
            camera = Camera(starting_point_x, starting_point_y, width, height, x, y, lavd_field)
        else:
            camera = Camera(camera.x, camera.y, width, height, x, y, lavd_field)
            
        # Plot the camera center
        ax.plot(camera.x, camera.y, marker="x", markersize=16, markeredgecolor="red", markerfacecolor="red")
        
        # Get the bottom left corner
        rect_left = camera.left
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
    
    
        ax.legend(['Max. Point in LAVD Field', 'Center of Camera', 'Camera Frame', 'Max. Point in Camera Frame'], bbox_to_anchor=(1.04, 1), loc="upper left")
    
        plt.show()
        
        conv = abs(lavd_max - field_max) / field_max
        
        print(conv)
        
        convs.append( conv )
        
        if iter_num > 0:
            if convs[iter_num] == convs[iter_num-1]:
                break
        
        iter_num += 1
        
    
if __name__ == "__main__":
    
    main(A, omega, epsilon, t0, t1, dt, width, height)