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
ti = 0
tf = 20
dt = 0.1    # time increment
#t0 = 0      # initial time
#t1 = 20     # final time

# Gyre parameters
A       = 0.1
epsilon = 0.1
omega   = 2*np.pi/10

# Camera parameters
# right side
starting_point_x = 1.5
starting_point_y = 0.3

# left side
#starting_point_x = 0.3
#starting_point_y = 0.7

#starting_point_x = 1.125
#starting_point_y = 0.2

# big camera
width            = 0.4
height           = 0.2

# small camera
#width            = 0.2
#height           = 0.1

saveFigs = 'yes'
saveVars = 'yes'


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

def main(A, omega, epsilon, ti, tf, dt, width, height):
    
    # initialize lavd_max
    conv = 1
    
    convs = [1]
    
    iter_num = 0

    for t_add in np.arange(ti, tf+dt, dt): #np.arange(1.1, 20, 0.1):
        
        #t0 = ti - t_add
        #t1 = tf - t_add
        
        t0 = ti + t_add
        t1 = tf + t_add
        
        # Load all the variables
        load_file = '../saveExplore/fwd_time/fields_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t0_'+str(round(t0,2))+'_t'+str(round(t1,2))+'.mat'
        f = sio.loadmat(load_file)
        x = f['x']
        y = f['y']
        lavd_field = f['lavd_field']
        #lavd_field = -lavd_field
        
        # Create a mesh for the x and y values
        x_mesh, y_mesh = createMesh(x,y)
        #lavd_x_max, lavd_y_max, field_max = findMax(lavd_field, [x_mesh, y_mesh])
        #x_mesh2, y_mesh2 = createMesh(x,y)
        
        x_mesh2, y_mesh2 = createMesh(x[:,201:400],y)
        #lavd_x_max, lavd_y_max, field_max = findMax(lavd_field[:,0:200], [x_mesh2, y_mesh2])
        
        #x_mesh2, y_mesh2 = createMesh(x[0,201:400],y)
        lavd_x_max, lavd_y_max, field_max = findMax(lavd_field[:,201:400], [x_mesh2, y_mesh2])
        
        print('x_vec:'+str(np.shape(x)))
        print('y_vec:'+str(np.shape(y)))
        print('lavd:'+str(np.shape(lavd_field)))
        
        # Plot the figures
        fig, ax = plt.subplots(1, 1) #constrained_layout=True)

        # LAVD figure
        ax.axis('scaled')
        ax.set_xlim([0,2])
        ax.set_ylim([0,1])
        h1 = ax.pcolormesh(x_mesh, y_mesh, lavd_field)
        j = np.unravel_index(lavd_field.argmax(), lavd_field.shape)
        ax.set_title('LAVD at $A$='+str(A)+', $\epsilon$='+str(epsilon)+', $\omega$=$\pi$/5, $t$='+str(round(t0,2))+' to '+str(round(t1,2))+', $dt$='+str(dt))
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
        ax.plot(camera.x, camera.y, marker="o", markersize=8, markeredgecolor="darkgreen", markerfacecolor="darkgreen")
    
        # Save figure
        if saveFigs == 'yes':
            plt.savefig('../rotation_prez/evolving/fwd_test1/camera_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t0_'+str(round(t0,2))+'_t'+str(round(t1,2))+'.png', format='png')
            #plt.savefig('./track_plots/bkwd_time_2/test1/camera_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t0_'+str(round(t0,2))+'_t'+str(round(t1,2))+'.png', format='png')
        
        plt.show()
        
        conv = abs(lavd_max - field_max) / field_max
        
        convs.append( conv )
        
        #if iter_num > 0:
        #    if convs[iter_num] == convs[iter_num-1]:
        #        break
        
        if saveVars == 'yes':
            #save_y0_file = './track_info/bkwd_time_2/test1/camera_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t0_'+str(round(t0,2))+'_t'+str(round(t1,2))+'.mat'
            save_y0_file = '../rotation_prez/evolving/fwd_test1/camera_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t0_'+str(round(t0,2))+'_t'+str(round(t1,2))+'.mat'
            sio.savemat(save_y0_file, {'camera_x':camera.x, 'camera_y':camera.y, 'x_mesh':camera_mesh_x, 'y_mesh':camera_mesh_y, 'lavd_mesh':camera_mesh_lavd})

    
        iter_num += 1
        
    
if __name__ == "__main__":
    
    main(A, omega, epsilon, ti, tf, dt, width, height)