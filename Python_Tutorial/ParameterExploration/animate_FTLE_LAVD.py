#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:50:12 2023

@author: chloe
"""

# General imports
import os, sys, copy
import numpy as np
from matplotlib import pyplot as plt

# Animation imports
import matplotlib.animation as animation
import matplotlib as mpl
from IPython.display import HTML
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

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
t1 = 20     # final time
dt = 0.1    # time increment
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

    "Computing Geometric LCS"
    # Reinitialize the flow object using Gyre once again
    gyre = Flow()
    
    # Now, make a denser grid of particles
    mesh, initial_conditions = createVectors(x0, xf, y0, yf, numRows, numCols)
    
    time_vector = np.arange(t0, t1, dt)
    tot_time = np.arange(t0, t1, dt)
    
    # time_vector = np.arange(t0,dt+dt,dt)
    
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
                             parameters=parameters, include_gradv=True)  # <- including gradv for LAVD
    
    # "Finite-Time Lyapunov Exponent (FTLE)"
    
    # First, integrate the trajectories.
    print("Integrating Trajectories...")
    gyre.integrate_trajectories()

    n_particles, n_times, dim = np.shape(gyre.states)

    "Create animation"    
    mpl.rcParams['animation.embed_limit'] = 100 # Mb
    
    # Create a new figure
    fig, ax = plt.subplots(2,1)
    title = ax[0].text(0.5,2.1, "", transform=ax[1].transAxes, ha="center")
    
    # # initialize the max/min points
    # minPoint, = ax[0].plot([],[],'o', markerfacecolor='red', markersize=8)
    # maxPoint, = ax[0].plot([],[],'o', markerfacecolor='cyan', markeredgecolor='hotpink', markersize=8)
    
    minPoint2, = ax[1].plot([],[],'o', markerfacecolor='red', markersize=8)
    maxPoint2, = ax[1].plot([],[],'o', markerfacecolor='cyan', markeredgecolor='hotpink', markersize=8)
    
    # print("Computing Jacobians...")
    # J_array = geom.computeJacobian(gyre.states[:,0:1,:], mesh)
    # # J_array stores a d-by-d array at each particle mesh location.
    # print(np.shape(J_array))
    
    # # Computing the FTLE field.
    # print("Computing FTLE...")
    # ftle_field = geom.computeFTLE(J_array, t1-t0)
    
    # Computing the LAVD field.
    print("Computing LAVD...")
    gyre.predefined_function(function_name, initial_conditions, time_vector,
                             parameters=parameters, include_gradv=True)  # <- including gradv for LAVD
    lavd_field = geom.computeLAVD(gyre.states, gyre.time_vector, mesh, gyre.gradv_function)
    
    # FTLE settings
    # ax[0].axis('scaled')
    # ax[0].set_xlim([0,2])
    # ax[0].set_ylim([0,1])
    # title.set_text(u"FTLE at $A$={}, $\epsilon$={}, $\omega$={}, Time: {}".format(A, epsilon, float(round(omega,3)), float(round(dt*0,2))) )
    # i = np.unravel_index(ftle_field.argmin(), ftle_field.shape)
    # h0 = ax[0].pcolormesh(mesh[0], mesh[1], ftle_field)
    # x_min, y_min = findMin(ftle_field, mesh)
    # x_max, y_max = findMax(ftle_field, mesh)
    # #l.set_title('FTLE')
    # ax[0].xaxis.set_ticks([])
    # minPoint.set_data(x_min, y_min)
    # maxPoint.set_data(x_max, y_max)
    
    # Set up the colorbar
    # divider = make_axes_locatable(ax[0])
    # cax = divider.append_axes("right", \
    #                            size = "5%",\
    #                           pad = .1) # J. Lang
    # im_ratio = mesh[0].shape[0]/mesh[0].shape[1]
    # cb0 = fig.colorbar(h0, cax=cax, fraction=im_ratio*0.046)
    
    # LAVD settings
    ax[1].axis('scaled')
    ax[1].set_xlim([0,2])
    ax[1].set_ylim([0,1])
    h1 = ax[1].pcolormesh(mesh[0], mesh[1], lavd_field)
    j = np.unravel_index(lavd_field.argmin(), lavd_field.shape)
    x_min, y_min = findMin(lavd_field, mesh)
    x_max, y_max = findMax(lavd_field, mesh)
    ax[1].set_title('LAVD')
    ax[1].plot(x_min, y_min, marker="o", markersize=8, markeredgecolor="red", markerfacecolor="red")
    ax[1].plot(x_max, y_max, marker="o", markersize=8, markeredgecolor="hotpink", markerfacecolor="cyan")
    im_ratio = mesh[0].shape[0]/mesh[0].shape[1]
    #cb1 = fig.colorbar(h1, cax=cax, fraction=im_ratio*0.046)
    minPoint2.set_data(x_min, y_min)
    maxPoint2.set_data(x_max, y_max)
    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", \
                               size = "5%",\
                              pad = .1) # J. Lang
    im_ratio = mesh[0].shape[0]/mesh[0].shape[1]
    cb1 = fig.colorbar(h1, cax=cax, fraction=im_ratio*0.046)
    
    
    # trying colorbar
    # gs = GridSpec(1,2,width_ratios = [0.9, 0.05])
    # ax = fig.add_subplot(gs[0])
    # cbar_ax = fig.add_subplot(gs[1])

    # Loop through each trajectory in self.states and plot it on the same axis
    def update(frame):
        
        # time_vector = np.arange(t0,dt*frame+dt,dt)
        # gyre.predefined_function(function_name, initial_conditions, time_vector,
        #                          parameters=parameters, include_gradv=True)  # <- including gradv for LAVD
        
        # print("Computing Jacobians...")
        # print(frame)
        # J_array = geom.computeJacobian(gyre.states[:,0:frame+1,:], mesh)
        # # J_array stores a d-by-d array at each particle mesh location.
        # print(np.shape(J_array))
        
        # # Computing the FTLE field.
        # print("Computing FTLE...")
        # ftle_field = geom.computeFTLE(J_array, t1-t0)
        title.set_text(u"LAVD at $A$={}, $\epsilon$={}, $\omega$={}, Time: {}".format(A, epsilon, float(round(omega,3)), float(round(dt*frame,2))) )
        
        #print('frame='+str(frame))
        time_vector = np.arange(t0+dt*frame, round(dt*(frame+len(tot_time)),2), dt)
        #print('time_vector='+str(time_vector))
        #print('time vector length='+str(len(time_vector)))
        gyre.predefined_function(function_name, initial_conditions, time_vector,
                                 parameters=parameters, include_gradv=True)  # <- including gradv for LAVD
        # Computing the LAVD field.
        print("Computing LAVD...")
        print("frame="+str(frame))
        #print('gyre size='+str(np.shape(gyre.states)))
        # [:,frame:frame+len(time_vector),:]
        lavd_field = geom.computeLAVD(gyre.states, gyre.time_vector, mesh, gyre.gradv_function)
        
        # ax[0].axis('scaled')
        # ax[0].set_xlim([0,2])
        # ax[0].set_ylim([0,1])
        # i = np.unravel_index(ftle_field.argmin(), ftle_field.shape)
        # h0 = ax[0].pcolormesh(mesh[0], mesh[1], ftle_field)
        # x_min, y_min = findMin(ftle_field, mesh)
        # x_max, y_max = findMax(ftle_field, mesh)
        # ax[0].xaxis.set_ticks([])
        # minPoint.set_data(x_min, y_min)
        # maxPoint.set_data(x_max, y_max)
        # title.set_text(u"FTLE at $A$={}, $\epsilon$={}, $\omega$={}, Time: {}".format(A, epsilon, float(round(omega,3)), float(round(dt))) )
        # #ax[0].set_title('FTLE')
        
        # im_ratio = mesh[0].shape[0]/mesh[0].shape[1]
        # cb0.update_normal
        
        ax[1].axis('scaled')
        ax[1].set_xlim([0,2])
        ax[1].set_ylim([0,1])
        h1 = ax[1].pcolormesh(mesh[0], mesh[1], lavd_field)
        j = np.unravel_index(lavd_field.argmin(), lavd_field.shape)
        x_min, y_min = findMin(lavd_field, mesh)
        x_max, y_max = findMax(lavd_field, mesh)
        ax[1].set_title('LAVD')
        ax[1].plot(x_min, y_min, marker="o", markersize=8, markeredgecolor="red", markerfacecolor="red")
        ax[1].plot(x_max, y_max, marker="o", markersize=8, markeredgecolor="hotpink", markerfacecolor="cyan")
        im_ratio = mesh[0].shape[0]/mesh[0].shape[1]
        minPoint2.set_data(x_min, y_min)
        maxPoint2.set_data(x_max, y_max)
        cb1.update_normal

    ani = animation.FuncAnimation(fig, update, frames=n_times, interval=dt*1000)
    
    #ani.save('test.mp4', writer = 'ffmpeg', fps = 3) 
    ani_name = 'LAVD_A'+str(A)+'_epsilon'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(t1)+'.mp4'
    ani.save(ani_name, writer='ffmpeg', fps=10)

    plt.close()
    
    # Save figure
    if saveFigs == 'yes':
        plt.savefig('./saveExplore/FIG_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(t1)+'.png', format='png')
        
    
    if saveVars == 'yes':
        save_y0_file = './saveExplore/LAVD_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(t1)+'.mat'
        #sio.savemat(save_y0_file, {'x':mesh[0], 'y':mesh[1], 'ftle_field':ftle_field, 'lavd_field':lavd_field})
    return


if __name__ == "__main__":
    
    main(x0, xf, y0, yf, t0, t1, dt, numRows, numCols)