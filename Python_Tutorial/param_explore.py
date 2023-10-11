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
"""
x0      = 0 # initial x-value
xf      = 2 # final x-value
y0      = 0 # initial y-value
yf      = 1 # final y-value
"""

# Save variables and figs
saveVars = 'yes'
saveFigs = 'yes'

#  Initialize
t0 = 0      # initial time
t1 = 20      # final time
dt = 0.1    # time increment

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
    
    #A = 0.1
    epsilon = 0.1
    omega   = 2*np.pi/10
    
    for A in np.arange(0.1, 5, 5): # #np.arange(np.pi/5, 2*np.pi, 2*np.pi/5)

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
    
        # Plot the figures
        fig, ax = plt.subplots(3, 1) #constrained_layout=True)
        
        # Set the x and y limits to be the same
        ax[0].axis('scaled')
        ax[0].set_xlim([0, 2])
        ax[0].set_ylim([0, 1])
        ax[0].xaxis.set_ticks([])
        ax[0].set_title('A='+str(A)+', $\epsilon$='+str(epsilon)+', $\omega$=$\pi$/5')
        
        # Loop through each trajectory in self.states and plot it on the same axis
        for i in range(0, n_particles, 20):
            x_vals = gyre.states[i,:,0]
            y_vals = gyre.states[i,:,1]
            ax[0].plot(x_vals, y_vals)
            
        if saveVars == 'yes':
            save_y0_file = './saveExplore/A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(t1)+'.mat'
            sio.savemat(save_y0_file, {'x_vals':x_vals, 'y_vals':y_vals, 'dx':1/numRows})
        
        # Show the plot
        # plt.show()

        "Computing Geometric LCS"
        # Reinitialize the flow object using Gyre once again
        gyre = Flow()
        
        # Now, make a denser grid of particles
        mesh, initial_conditions = createVectors(x0, xf, y0, yf, numRows, numCols)
        
        # Next, we need to make a time vector
        #t0 = 0      # initial time
        #t1 = 12      # final time
        #dt = 0.1    # time increment <- # For standard FTLE we will only need the first and last time, but 
                                        # it will be helpful when computing LAVD to have increments.
        time_vector = np.arange(t0,t1+dt,dt)
    
        '''We now need to specify the flow that the Flow object will operate on.'''
        parameters = {  # These are specified as defaults as well. 
            "A": A,
            "epsilon": epsilon,
            "omega": omega
        }
        gyre.predefined_function(function_name, initial_conditions, time_vector,
                                 parameters=parameters, include_gradv=True)  # <- including gradv for LAVD
    
        "Finite-Time Lyapunov Exponent (FTLE)"
        
        # First, integrate the trajectories.
        print("Integrating Trajectories...")
        gyre.integrate_trajectories()
        
        # Computing the jacobian at each mesh position
        print("Computing Jacobians...")
        J_array = geom.computeJacobian(gyre.states, mesh)
        
        # J_array stores a d-by-d array at each particle mesh location.
        print(np.shape(J_array))
        
        # Computing the FTLE field.
        print("Computing FTLE...")
        ftle_field = geom.computeFTLE(J_array, t1-t0)
        
        # Computing the LAVD field.
        print("Computing LAVD...")
        lavd_field = geom.computeLAVD(gyre.states, gyre.time_vector, mesh, gyre.gradv_function)
        
        # FTLE figure
        ax[1].axis('scaled')
        ax[1].set_xlim([0,2])
        ax[1].set_ylim([0,1])
        i = np.unravel_index(ftle_field.argmin(), ftle_field.shape)
        h0 = ax[1].pcolormesh(mesh[0], mesh[1], ftle_field)
        x_min, y_min = findMin(ftle_field, mesh)
        x_max, y_max = findMax(ftle_field, mesh)
        ax[1].set_title('FTLE')
        ax[1].xaxis.set_ticks([])
        ax[1].plot(x_min,y_min, marker="o", markersize=8, markeredgecolor="red", markerfacecolor="red")
        ax[1].plot(x_max, y_max, marker="o", markersize=8, markeredgecolor="hotpink", markerfacecolor="cyan")
        im_ratio = mesh[0].shape[0]/mesh[0].shape[1]
        cbar0 = fig.colorbar(h0, fraction=0.046*im_ratio)
        
        # LAVD figure
        ax[2].axis('scaled')
        ax[2].set_xlim([0,2])
        ax[2].set_ylim([0,1])
        h1 = ax[2].pcolormesh(mesh[0], mesh[1], lavd_field)
        j = np.unravel_index(lavd_field.argmin(), lavd_field.shape)
        x_min, y_min = findMin(lavd_field, mesh)
        x_max, y_max = findMax(lavd_field, mesh)
        ax[2].set_title('LAVD')
        ax[2].plot(x_min, y_min, marker="o", markersize=8, markeredgecolor="red", markerfacecolor="red")
        ax[2].plot(x_max, y_max, marker="o", markersize=8, markeredgecolor="hotpink", markerfacecolor="cyan")
        im_ratio = mesh[0].shape[0]/mesh[0].shape[1]
        cbar1 = fig.colorbar(h1, fraction=0.046*im_ratio)
        
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.subplots_adjust(left=0.54)
        plt.subplots_adjust(left=0.58, hspace=0.35)

        
        "Create animation"
        
#        mpl.rcParams['animation.embed_limit'] = 100 # Mb
        
#        # Create a new figure
#        fig, ax = plt.subplots(1,1)
        
#        # Set the x and y limits to be the same
#        ax.axis('scaled')
#        ax.set_xlim([0, 2])
#        ax.set_ylim([0, 1])
    
#        # initialize the plot
#        l, = ax.plot([],[],'o', markerfacecolor='blue', markersize=3)
    
#        # Loop through each trajectory in self.states and plot it on the same axis
#        def update(frame):
#            x_vals = gyre.states[:,frame,0]
#            y_vals = gyre.states[:,frame,1]
#            l.set_data(x_vals, y_vals)
        
#        ani = animation.FuncAnimation(fig, update, frames=n_times, interval=dt*1000)
#        HTML(ani.to_jshtml())
        
        
        #for i in np.arange(0,3,1):
        #    ax[i].set_anchor('W')
        
        # Save figure
        if saveFigs == 'yes':
            plt.savefig('./saveExplore/FIG_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(t1)+'.png', format='png')
        
        plt.show()
        
        if saveVars == 'yes':
            save_y0_file = './saveExplore/fields_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(t1)+'.mat'
            sio.savemat(save_y0_file, {'x':mesh[0], 'y':mesh[1], 'ftle_field':ftle_field, 'lavd_field':lavd_field})
            
    return


if __name__ == "__main__":
    
    main(x0, xf, y0, yf, t0, t1, dt, numRows, numCols)
    