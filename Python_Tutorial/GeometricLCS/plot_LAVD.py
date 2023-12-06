# -*- coding: utf-8 -*-
"""
Performing a parameter exploration of the double gyre
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

# Save variables and figs
saveVars = 'no'
saveFigs = 'no'

#  Initialize
t0 = 19.9      # initial time
t1 = 20      # final time
dt = 0.1    # time increment


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
    return x_vec, y_vec, mesh, initial_conditions


def main(x0, xf, y0, yf, t0, t1, dt, numRows, numCols):
    
    # Initialize a flow object
    function_name = "Gyre"
    
    gyre = Flow()

    # originally numRows = 15
    x_vec, y_vec, mesh, initial_conditions = createVectors(x0, xf, y0, yf, numRows, numCols)

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
    gyre.predefined_function(function_name, initial_conditions, time_vector,
                             parameters=parameters, include_gradv=True)
    # Integrate the particles over the defined time vector
    gyre.integrate_trajectories()
    n_particles, n_times, dim = np.shape(gyre.states)

    # Plot the figures
    fig, ax = plt.subplots(1, 1)
    
    # First, integrate the trajectories.
    print("Integrating Trajectories...")
    gyre.integrate_trajectories()
    
    # Computing the jacobian at each mesh position
    print("Computing Jacobians...")
    J_array = geom.computeJacobian(gyre.states, mesh)
    
    # Computing the LAVD field.
    print("Computing LAVD...")
    lavd_field = geom.computeLAVD(gyre.states, gyre.time_vector, mesh, gyre.gradv_function)
    
    # LAVD figure
    ax.axis('scaled')
    ax.set_xlim([0,2])
    ax.set_ylim([0,1])
    h1 = ax.pcolormesh(mesh[0], mesh[1], lavd_field)
    j = np.unravel_index(lavd_field.argmin(), lavd_field.shape)
    x_max, y_max = findMax(lavd_field, mesh)
    ax.set_title('LAVD at $A$='+str(A)+', $\epsilon$='+str(epsilon)+', $\omega$=$\pi$/5, $t$='+str(t0)+' to '+str(t1)+', $dt$='+str(dt))
    im_ratio = mesh[0].shape[0]/mesh[0].shape[1]
    cbar1 = fig.colorbar(h1, fraction=0.046*im_ratio)
    
    # Save figure
    if saveFigs == 'yes':
        plt.savefig('./saveExplore/FIG_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(t1)+'.png', format='png')
    
    plt.show()
    
    if saveVars == 'yes':
        save_y0_file = './saveExplore/fields_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(t1)+'.mat'
        sio.savemat(save_y0_file, {'x':x_vec, 'y':y_vec, 'lavd_field':lavd_field})
        
    return


if __name__ == "__main__":
    
    main(x0, xf, y0, yf, t0, t1, dt, numRows, numCols)
    