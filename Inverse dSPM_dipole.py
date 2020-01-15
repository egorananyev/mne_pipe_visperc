# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:48:05 2020

@author: Aaron Ang
"""
import matplotlib.pyplot as plt

import os
import numpy as np
import mne

from mne.forward import read_forward_solution
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)

from mne.minimum_norm import apply_inverse_raw, apply_inverse_epochs

from mayavi import mlab

## If you need help just ask the machine
#get_ipython().run_line_magic('pinfo', 'mne.pick_types')

## Specify path to data
data_path = os.path.expanduser("D:\\Ubuntu_Programs\\meeg\\ds000117-practical\\")  

## Specify path to MEG raw data file
raw_fname = os.path.join(data_path,
    'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif')

## Read raw data file and assign it as a variable
raw = mne.io.read_raw_fif(raw_fname, preload=False)

evoked_fname = raw_fname.replace('_meg.fif', '-ave.fif')
evoked = mne.read_evokeds(evoked_fname, baseline=(None, 0), proj=True)

epochs_fname = raw_fname.replace('_meg.fif', '-epo.fif')
epochs = mne.read_epochs(epochs_fname)

trans_fname = os.path.join(data_path,
    'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01-trans.fif')
bem_fname = os.path.join(data_path,
    'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01-bem.fif')

## Read the forward solution and compute the inverse operator
fwd_fname = os.path.join(data_path,
    'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01-meg-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.convert_forward_solution(fwd, surf_ori=True)

subjects_dir = os.path.join(data_path, 'derivatives\\freesurfer-reconall')

## ============================================================================
## Section 1:
## To initiate your desired function, indicate '1' in the respective variables.
##
## Note that %matplotlib qt (or inline) won't work on Spyder.
## In the case of Spyder, you'll need to set Tools -> references -> IPython Console -> Graphics -> Backend: Automatic.
## to get plots to be generated in a sepaate window.
##
## It is also important to set Tools -> Preferences -> Run -> General Settings -> Remove all variables before execution
## to make sure that all stored memory has been purged from previous runs to prevent memory overload.
##
## Comments with *** indicates that these variables should always be set to 1. Otherwise, data/process will turn
## wonky.
## ============================================================================
operations_to_apply = dict(
        
                    ### Inverse Solution ###
                    ### -------------------------------------
                    ## 
                    PlotHeadSensorAlignment = 0,
                    
                    ComputeContrast = 1,                # *** Need to define conditions you wish to read in Section 2, part 1
                    PlotContrast = 0,                   # Needs ComputeContrast = 1.
                    ComputeNoiseCov = 1,                # *** Plot Noise covarience. If noise is independent, should see diagonals in plots. Spatial whitening plot requires ComputeContrast = 1 to work.
                    PlotNoiseCov = 0,                   # Needs ComputeNoiseCov = 1.
                    
                    FitDipole = 0,                      # Using sequential (time-varying position and orientation) fit
                    FitSelectDipole = 0,                # Use a subselection of channels. Define in Section 2 part 2.
                                                        # Use this method to improve Goodness of Fit, should FitDipole fit poorly.
                    
                    ComputeAndApplyInverse = 1,         # *** Requires the above two Computes = 1 to work. Need to define InverseMethod in Section 2, part 2.
                    PlotSourceTimeCourse = 0,           # UNSTABLE FOR THE TIME BEING. Shows brain but script cannot shut down after closing image. Saving image doesn't work too.
                    
                    MorphDataToAvgBrain = 0,            # Define the exact time you wish to generate brain image in Section 2 part 3. (type stc and look at the 2nd value of data shape - the max time slice available)
                    ComputeSourceMorph = 1              # UNSTABLE FOR THE TIME BEING. Need to define time range in Section 2 part 4.
                    )


## ============================================================================
## Section 2:
## Here, you get a sandbox for editing the data in various ways that you require
## ============================================================================
subject = 'sub-01' 

## Part 1                   
epochs.pick_types(meg=True, eeg=False) # Pick Epoch based on sensor types
SensorType = 'mag'

evoked_cond1 = epochs['face'].average() # Pick conditions to generate average evoked
evoked_cond2 = epochs['scrambled'].average()   

ContrastMinTime = None
ContrastMaxTime = 0.2                

## Part 2                    
# Select channels for dipole fit
SelectedChns = 'Left'

# Restrict forward solution as necessary for MEG
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)    

InverseMethod = 'dSPM' 

## Part 3
# Time Index of the specific slice you need for plotting of data on inflated brain      
TimeIndexNum = 120  

## Part 3
start, stop = raw.time_as_index([0, 20])  # read the first 20s of data (at 1100 Hz, it's 22k samples)       
                    
                    
## ============================================================================
## Operations commands
## ============================================================================

if operations_to_apply['PlotHeadSensorAlignment']:
    bem = mne.bem.read_bem_solution(bem_fname)

    info = mne.io.read_info(epochs_fname)
    fig = mne.viz.plot_alignment(info, trans_fname, subject=subject, dig=True,
                                 subjects_dir=subjects_dir, bem=bem, verbose=True);
                           
if operations_to_apply['ComputeContrast']:
    contrast = mne.combine_evoked([evoked_cond1, evoked_cond2], [0.5, -0.5])
    contrast.crop(ContrastMinTime, ContrastMaxTime)  ## could specify -0.2 instead of None
    
if operations_to_apply['FitDipole']:
    cov = mne.compute_covariance(epochs, rank='info')

    # Fit a dipole using a sequential (time-varying position and orientation) fit
    contrast_crop = contrast.copy().crop(0.05, 0.06)
    dip, residual = mne.fit_dipole(contrast_crop, cov, bem_fname,
                                   trans_fname)
    print(dip)
    
    # Look at our result
    print(dip.gof)
    
    dip.plot_locations(subject=subject, trans=trans_fname,
                       subjects_dir=subjects_dir, mode='orthoview');

if operations_to_apply['FitSelectDipole']:
    selection = mne.read_selection(SelectedChns, info=contrast.info)
    
    # Fit a dipole using a sequential (time-varying position and orientation) fit
    dip, residual =     mne.fit_dipole(contrast_crop.copy().pick_channels(selection),
                       cov, bem_fname, trans_fname)
    print(dip)
    
    print(dip.gof)
    
    dip.plot_locations(subject=subject, trans=trans_fname,
                       subjects_dir=subjects_dir, mode='orthoview');


if operations_to_apply['PlotContrast']:
    contrast.plot();
    
    contrast.plot_topomap(times=np.linspace(0.1, 0.2, 5), ch_type=str(SensorType));    
    
if operations_to_apply['ComputeNoiseCov']:
    noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0,  ## although tmax/min are set by default ...
                                   method=['shrunk', 'empirical'],  ## ...it's best to specify explicitly for speed
                                   rank='info')
    print(noise_cov.data.shape)
    
if operations_to_apply['PlotNoiseCov']:
    mne.viz.plot_cov(noise_cov, epochs.info)
    contrast.plot_white(noise_cov);


if operations_to_apply['ComputeAndApplyInverse']:    
    # make an M/EEG, MEG-only, and EEG-only inverse operator
    info = contrast.info
    inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                             loose=0.2,  ## default value for surface-oriented source space; used as weight for tangential source var
                                             depth=0.8)  ## also default for surface-based methods
    
    method = str(InverseMethod)
    snr = 3.
    lambda2 = 1. / snr ** 2
    stc = apply_inverse(contrast, inverse_operator, lambda2,
                        method=method, pick_ori=None)  ## pick_ori=None is default
    print(stc)
    
    # The ``stc`` (Source Time Courses) are defined on a source space formed by 8188 candidate
    # locations and for a duration spanning 121 time points. (Or 211 in our case.)    
    stc.save('fixed_ori')  # save the STC to disk

if operations_to_apply['PlotSourceTimeCourse']:
    brain = stc.plot(surface='inflated', hemi='both', subjects_dir=subjects_dir,
                  time_viewer=True)
    brain.set_data_time_index(TimeIndexNum)  ## 120, or this works with 211 anyway
    brain.scale_data_colormap(fmin=8, fmid=12, fmax=15, transparent=True)
    brain.show_view('ventral')
    brain.save_image('dspm.jpg')
    
    mlab.close()
    #Image(filename='dspm.jpg', width=600)

if operations_to_apply['MorphDataToAvgBrain']:
    mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
    
    morph = mne.compute_source_morph(stc, subject_from=subject,
                                     subject_to='fsaverage',  ## this is a "template" surface image
                                     subjects_dir=subjects_dir)
    stc_fsaverage = morph.apply(stc)
    
    stc_fsaverage.save('fsaverage_dspm')
    
    brain_fsaverage = stc_fsaverage.plot(surface='inflated', hemi='lh',
                                         subjects_dir=subjects_dir)
    brain_fsaverage.set_data_time_index(TimeIndexNum) #45
    brain_fsaverage.scale_data_colormap(fmin=8, fmid=12, fmax=15, transparent=True)
    brain_fsaverage.show_view('lateral')
    brain_fsaverage.save_image('dspm_fsaverage.jpg')

    # mlab.close()
    # Image(filename='dspm_fsaverage.jpg', width=600)

if operations_to_apply['ComputeSourceMorph']:
    
    label = mne.compute_source_morph(stc, subject_from=subject,
                                     subject_to='fsaverage',
                                     subjects_dir=subjects_dir)
    stc = apply_inverse_raw(raw, inverse_operator, lambda2, method, label,
                            start, stop)