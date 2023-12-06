# ME398 Fall Rotation
Source code for "Towards a Predictive Framework for Tracking Lagrangian Vortices"

## Outline

There were two main parts to this rotation project:

1. Parameter exploration <br />
2. Camera tracking algorithm <br />
    a. Static frame with dense data <br />
    b. Evolving frame with dense data <br />
    c. Evolving frame with sparse data <br />

## Parameter Exploration

To gain physical intuition of the double gyre problem, parameters $A, \epsilon, \omega$ were explored via various figures and animations.

1. Run ```./Python_Tutorial/ParameterExploration/param_explore.py``` <br />
    a. Alters parameters used in the double gyre problem <br />
    b. To replicate the parameter subplots in the presentation, run ```./Python_Tutorial/ParameterExploration/plot_params.m```
2. Run ```./Python_Tutorial/ParameterExploration/animate_particles.py``` <br />
    a. Animates a grid of particles in the double gyre < br/>
3. Run ```./Python_Tutorial/ParameterExploration/animate_FTLE.py``` <br />
    a. Animates the FTLE field over time <br />
4. Run ```./Python_Tutorial/ParameterExploration/animate_FTLE_LAVD.py``` <br />
    a. Animates both the FTLE field and LAVD field over time <br />

## Camera Tracking Algorithm

### Static frame with dense data

1. Save the LAVD field of interest in ```./Python_Tutorial/GeometricLCS/save_LAVD.py``` <br />
2. Track the vortex core in ```./Python_Tutorial/CameraTracking/static_camera_tracking.py``` <br />
    a. To save data, set ```saveVars = 'yes'``` and to save figures, ```saveFigs = 'yes'``` <br />
3. Uncomment the corresponding section and save animation using ```./Python_Tutorial/CameraTracking/save_camera_anim.py``` <br />

### Evolving frame with dense data

1. Save the LAVD field of interest in ```./Python_Tutorial/GeometricLCS/save_LAVD.py``` <br />
2. Track the vortex core in ```./Python_Tutorial/CameraTracking/dense_evolving_camera_tracking.py``` <br />
    a. To save data, set ```saveVars = 'yes'``` and to save figures, ```saveFigs = 'yes'``` <br />
3. Uncomment the corresponding section and save animation using ```./Python_Tutorial/CameraTracking/save_camera_anim.py``` <br />

### Evolving frame with sparse data

1. Save the LAVD field of interest in ```./Python_Tutorial/SparseLCS/save_LAVD_sparse.py``` <br />
2. Track the vortex core in ```./Python_Tutorial/CameraTracking/sparse_evolving_camera_tracking.py``` <br />
    a. To save data, set ```saveVars = 'yes'``` and to save figures, ```saveFigs = 'yes'``` <br />
3. Uncomment the corresponding section and save animation using ```./Python_Tutorial/CameraTracking/save_camera_anim.py``` <br />
