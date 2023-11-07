#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Oct 11 22:48:59 2023

@author: chloe
"""

"""
Creating an animation of the double gyre
"""

# General imports
import os, sys, copy
import numpy as np
from matplotlib import pyplot as plt

# Animation imports
import matplotlib.animation as animation
import matplotlib as mpl
from IPython.display import HTML

# Import the functions from the Flows folder
from Flows.Flows import Flow

# FTLE Imports
import GeometricLCS.Geometric as geom

# Saving Imports
import scipy.io as sio

# Initialize parameters for the domain
numRows = 200
#dx      = 1/numRows # the way that the dx is defined
numCols = 200
x0      = 0 # initial x-value
xf      = 2 # final x-value
y0      = 0 # initial y-value
yf      = 1 # final y-value
"""
x0      = 0 # initial x-value
xf      = 2 # final x-value
y0      = 0 # initial y-value
yf      = 1 # final y-value
"""

# Save variables and figs
saveVars = 'no'
saveFigs = 'no'

#  Initialize
t0 = 0      # initial time
t1 = -20     # final time
dt = -0.1    # time increment
# t1 = -20     # final time
# dt = -0.1    # time increment
"""
t0 = 0      # initial time
t1 = 20      # final time
dt = 0.1    # time increment
"""

# Specify parameters for the double gyre
# A       = 0.1
# epsilon = 0.1
# omega   = 2*np.pi/10

def findMin(field_interest, mesh):
    
    j      = np.unravel_index(field_interest.argmin(), field_interest.shape)
    x_mesh = mesh[0]
    x_min  = x_mesh[j]
    y_mesh = mesh[1]
    y_min  = y_mesh[j]
    
    return x_min, y_min

def findMax(field_interest, mesh):
    
    j      = np.unravel_index(field_interest.argmax(), field_interest.shape)
    x_mesh = mesh[0]
    x_max  = x_mesh[j]
    y_mesh = mesh[1]
    y_max  = y_mesh[j]
    
    return x_max, y_max


def createVectors(x0, xf, y0, yf, numRows, numCols):
    "Define the initial conditions for our integrations."
    "We will define them on a grid for now."
    
    # Specify the flow domain using dim[0] = x axis and dim[1] = y axis
    domain  = np.array([[x0, xf],[y0, yf]])
    dx      = 1/numRows
    dy      = 1/numCols
    
    # Now, make vectors associated with each axis.
    x_vec = np.arange(domain[0,0],domain[0,1]+dx,dx)     # 50 columns
    y_vec = np.arange(domain[1,0],domain[1,1]+dy,dy)     # 25 rows
    
    # Then, make the mesh grid and flatten it to get a single vector of positions.  
    mesh = np.meshgrid(x_vec, y_vec)
    x = mesh[0].reshape(-1,1)
    y = mesh[1].reshape(-1,1)
    initial_conditions = np.append(x, y, axis=1)
    return mesh, initial_conditions


def main(x0, xf, y0, yf, t0, t1, dt, numRows, numCols):
    
    # Initialize a flow object
    function_name = "Gyre"
    
    gyre = Flow()

    # originally numRows = 15
    mesh, initial_conditions = createVectors(x0, xf, y0, yf, 15, 15)

    time_vector = np.arange(t0,t1+dt,dt)
    
    A       = 0.1
    epsilon = 0.1
    omega   = 2*np.pi/10

    '''We now need to specify the flow that the Flow object will operate on.'''
    parameters = {  # These are specified as defaults as well. 
        "A": A,
        "epsilon": epsilon,
        "omega": omega
    }
    gyre.predefined_function(function_name, initial_conditions, time_vector, parameters=parameters)

    # Integrate the particles over the defined time vector
    gyre.integrate_trajectories()
    n_particles, n_times, dim = np.shape(gyre.states)
    
    print(n_particles)
    print(n_times)
    print(dim)

    "Create animation"
    
    # mpl.use('Qt5Agg') # to get GUI up
    
    mpl.rcParams['animation.embed_limit'] = 100 # Mb

    # Create a new figure
    fig, ax = plt.subplots(1,1)

    # Set the x and y limits to be the same
    ax.axis('scaled')
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 1])

    # initialize the plot
    l, = ax.plot([],[],'o', markerfacecolor='blue', markersize=3)

    title = ax.text(0.5,1.1, "", transform=ax.transAxes, ha="center")
    
    # title = ax.text(0.5,1.1, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
    #            transform=ax.transAxes, ha="center")

    # Loop through each trajectory in self.states and plot it on the same axis
    def update(frame):
        x_vals = gyre.states[:,frame,0]
        y_vals = gyre.states[:,frame,1]
        l.set_data(x_vals, y_vals)
        title.set_text(u"$A$={}, $\epsilon$={}, $\omega$={}, Time: {}".format(A, epsilon, float(round(omega,3)), float(round(dt*frame,2))) )

    ani = animation.FuncAnimation(fig, update, frames=n_times, interval=dt*1000)
    
    #ani.save('test.mp4', writer = 'ffmpeg', fps = 3) 
    ani_name = 'A'+str(A)+'_epsilon'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'.mp4'
    ani.save(ani_name, writer='ffmpeg', fps=20)

    plt.close()
    
    # Save figure
    if saveFigs == 'yes':
        plt.savefig('./saveExplore/FIG_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(t1)+'.png', format='png')
        
    
    if saveVars == 'yes':
        save_y0_file = './saveExplore/fields_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(t1)+'.mat'
        #sio.savemat(save_y0_file, {'x':mesh[0], 'y':mesh[1], 'ftle_field':ftle_field, 'lavd_field':lavd_field})
    return


if __name__ == "__main__":
    
    main(x0, xf, y0, yf, t0, t1, dt, numRows, numCols)
    