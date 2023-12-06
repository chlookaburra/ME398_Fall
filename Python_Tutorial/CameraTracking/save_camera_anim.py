#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:23:16 2023

@author: chloe
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

nframes = 201

A       = 0.1
epsilon = 0.1
omega   = 2*np.pi/10

# STATIC GRID TRACKING
# =============================================================================
# def animate(frame):
#     
#     # sparse grid
#     im = plt.imread('./static/test2/frame'+str(round(frame,2))+'.png', format='png')
#     plt.imshow(im)
# 
# anim = FuncAnimation(plt.gcf(), animate, frames=nframes, interval=(2000.0/nframes))
# 
# ani_name = './static/test2/fwd_LAVD_A'+str(A)+'_epsilon'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'.mp4'
# anim.save(ani_name, writer='ffmpeg', fps=10)
# =============================================================================

# DENSE GRID TRACKING
#=============================================================================
def animate(i):
    
    t0 = ti + dt*i #bkwd time
    t1 = tf + dt*i #bkwd time
    
    #t0 = ti - dt*i #bkwd time
    #t1 = tf - dt*i #bkwd time
    
    im = plt.imread('../rotation_prez/evolving/fwd_test1/camera_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t0_'+str(round(t0,2))+'_t'+str(round(t1,2))+'.png', format='png')

    #im = plt.imread('./track_plots/bkwd_time_2/test3/camera_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t0_'+str(round(t0,2))+'_t'+str(round(t1,2))+'.png', format='png')
    plt.imshow(im)
 
# fwd time 
ti = 0
tf = 20
dt = 0.1

# bkwd time
#ti = 0
#tf = -20
#dt = -0.1

anim = FuncAnimation(plt.gcf(), animate, frames=nframes, 
                  interval=(2000.0/nframes))


#ani_name = './track_plots/bkwd_time_2/test3/bkwd_LAVD_A'+str(A)+'_epsilon'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(tf)+'.mp4'
ani_name = '../rotation_prez/evolving/fwd_test1/fwd_LAVD_A'+str(A)+'_epsilon'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_t'+str(tf)+'.mp4'

anim.save(ani_name, writer='ffmpeg', fps=10)
#=============================================================================


# SPARSE TRACKING
# =============================================================================
# def animate(frame):
#     
#     # sparse grid
#     im = plt.imread('./sparse/track_plots/fwd_time_50/camera_A'+str(A)+'_eps'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'_frame'+str(round(frame,2))+'.png', format='png')
#     plt.imshow(im)
# 
# anim = FuncAnimation(plt.gcf(), animate, frames=nframes, 
#                  interval=(2000.0/nframes))
# 
# 
# ani_name = './sparse/track_plots/fwd_time_50/bkwd_LAVD_A'+str(A)+'_epsilon'+str(epsilon)+'_omega'+str(float(round(omega,2)))+'.mp4'
# anim.save(ani_name, writer='ffmpeg', fps=10)
# 
# =============================================================================
