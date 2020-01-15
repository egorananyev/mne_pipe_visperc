# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:57:20 2020

@author: Aaron Ang
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch

from IPython.display import Image
from mayavi import mlab

from nilearn import plotting

## We set the log-level to 'warning' so the output is less verbose
mne.set_log_level('warning')

## If you need help just ask the machine
#get_ipython().run_line_magic('pinfo', 'mne.pick_types')

## Specify path to data
data_path = os.path.expanduser("D:\\Ubuntu_Programs\\meeg\\ds000117-practical\\")  

## Specify path to MEG raw data file
raw_fname = os.path.join(data_path,
    'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif')

## Read raw data file and assign it as a variable
raw = mne.io.read_raw_fif(raw_fname, preload=False)

subjects_dir = os.path.join(data_path, 'derivatives\\freesurfer-reconall\\')

trans_fname = os.path.join(data_path,
    'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01-trans.fif')

bem_fname = os.path.join(data_path,
    'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01-bem.fif')

fwd_fname = os.path.join(data_path,
    'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01-meg-fwd.fif')

## Read Epochs file generated from 'Raw to Epochs_Evoked'
## We're reading the epochs that contain the projections that haven't yet been applied.
epochs_fname = raw_fname.replace('_meg.fif', '-epo.fif')

epochs = mne.read_epochs(epochs_fname)  # preload=True by default


# epochs.resample(200., npad='auto')  # resample to reduce computation time
epochs.apply_proj()  # does this mean *all* projections are applied?! which projections are stored & how can we tell?

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
ProcessMode = 2 # To reduce memory load of script. Mode = 1 only runs commands for ### Time Frequency ###.
                # Mode = 2 runs commands for ### Forward Modelling ###.

operations_to_apply = dict(
        
                    ### Time Frequency ###
                    ### -------------------------------------
                    ## Frequency Analysis
                    PlotPSD = 0,                    # Power Spectral Density. Need to specify freq range in Section 2 part 1.
                    PlotPSDMultiTaper = 0,          # Need to specify which freq range and sensor group type in Section 2 - Part 1
                    PlotPSDWelch = 0,               # As above.
                    ComputeTRF = 0,                 # Compute Time Frequency Representation. Need to define Freq and Time range in Section 2 part 1.
                    
                    ## Power
                    PlotPowerTopo = 0,              # Power plot over entire brain.
                    PlotPowerSensor = 0,            # Power plot for a specific sensor. Need to define in Section 2 part 2, SensorID.
                    PlotFreqBandTopo = 0,           # Power plot for different frequency bands. Need to define time range in Section 2 part 2.
                    PlotTimeFreqAcrossSensors = 0,  # A joint plot showing both the aggregated TFR across channels and topomaps at specific times and frequencies to obtain
                                                    # a quick overview regarding oscillatory effects across time and space.
                                                    # Need to specifiy specific time and freq for plot in Section 2 part 3.
                                                    
                    ## Inter Trial Coherence
                    PlotITC = 0,
                    
                    
# ## Compute and visualize BEM surfaces
#
# Symmetric boundary element method (symmetric BEM) uses three realistic layers (scalp, inner skull, outer skull). The goal of this forward solution is mostly for EEG users, to provide more accurate results than the spherical models.
# Computing the BEM surfaces requires FreeSurfer and makes use of either of the two following command line tools:
# 
# [mne watershed_bem](http://martinos.org/mne/dev/generated/commands.html#mne-watershed-bem)
# [mne flash_bem](http://martinos.org/mne/dev/generated/commands.html#mne-flash-bem)
# 
# or directly by calling the functions
# 
# https://mne.tools/stable/generated/mne.bem.make_watershed_bem.html
# https://mne.tools/stable/generated/mne.bem.make_flash_bem.html

                    ### Forward Modelling ###
                    ### -----------------------------
                    ## Structural
                    PlotT1 = 0,
                    PlotT1Slices = 0,               # Need to specify subject and which orientation slice in Section 2 part 4
                    
                    ## Co-registration
                    CoReg = 0,                      # Launch GUI to do co-registration. Need to specify subject in Section 2 part 4
                                                    # In order to generate the trans.fif file, load the digitization source (raw data: _meg.fif_) and press either Fit Fid. Then you can save using the bottom-right button.
                    CheckCoReg = 0,
                    PlotSourceSpace = 1,            # *** This needs to be set to 1 for computing of forward model to work.
                    
                    ## Forward Modelling
                    ComputeForwardSolution = 1,     # *** Leadfield matrix requires forward solution.
                    LeadfieldMatrix = 1,            # *** Need to specify sensor type in Section 2 part 5. Should always be set to 1 for Plot3DModel to work.
                    Plot3DModel = 1                 # Need to specify sensor type, view type in Section 2 part 5
                    )

## Path to T1 structural
t1_fname = os.path.join(data_path, 'derivatives\\freesurfer-reconall\\sub-01\\mri\\T1.mgz')

## ============================================================================
## Section 2:
## Here, you get a sandbox for editing the data in various ways that you require
## ============================================================================

## Part 1
baseline_mode = 'logratio'
baseline = (None, 0)

MinFreq = 2.
MaxFreq = 40.
meg = 'grad'

MinTime = -0.1
MaxTime = 1.6

## Part 2
SensorID = 82

FreqBMinTime = 0.05
FreqBMaxTime = 0.15

## Part 3
TimeFreq1 = (0.25,2.) #in seconds and Hz
TimeFreq2 = (1.,11.)

## Part 4
subject = 'sub-01'
sliceType = 'coronal'

## Part 5
SensorType = 'mag'
BrainView = 'lat'

## ============================================================================
## Operations commands
## ============================================================================
if ProcessMode == 1:
    if operations_to_apply['PlotPSD']:
        epochs.plot_psd(fmin=MinFreq, fmax=MaxFreq, average=False, bandwidth=2);
        ## Normalized = True - relative power, defined as the power in a given band divided by the total power
        epochs.plot_psd_topomap(ch_type='eeg', normalize=True, cmap='viridis');
        epochs.plot_psd_topomap(ch_type='mag', normalize=True, cmap='viridis');
        epochs.plot_psd_topomap(ch_type='grad', normalize=True, cmap='viridis')
    
    if operations_to_apply['PlotPSDMultiTaper']:
        psds, freqs = psd_multitaper(epochs, fmin=MinFreq, fmax=MaxFreq, n_jobs=1, bandwidth=2)
        print('PSD Shape (No. of Epochs, No. of Channels, No. of Freq):')
        print(psds.shape)
        
        psds_ave = np.mean(10. * np.log10(psds), axis=0)  # use dB and average over epochs
        picks_grad = mne.pick_types(epochs.info, meg=str(meg), eeg=False)  ## fixed to "grad" (was "mag")
        
        f, ax = plt.subplots()
        ax.plot(freqs, psds_ave[picks_grad].T, color='k', alpha=0.15)
        ax.set(title='Multitaper PSD' + '('+ str(meg) + ')', xlabel='Frequency (Hz)',
               ylabel='Power Spectral Density (dB)')
        plt.show()
    
    if operations_to_apply['PlotPSDWelch']:
        psds_welch, freqs_welch = psd_welch(epochs, fmin=MinFreq, fmax=MaxFreq, n_jobs=1, average='median')
        print('PSD Shape (No. of Epochs, No. of Channels, No. of Freq):')
        print(psds_welch.shape)
        
        psds_ave_welch = np.mean(10. * np.log10(psds_welch), axis=0)  # use dB and average over epochs
        picks_grad = mne.pick_types(epochs.info, meg=str(meg), eeg=False) 
    
        f, ax = plt.subplots()
        ax.plot(freqs_welch, psds_ave_welch[picks_grad].T, color='k', alpha=0.15)
        ax.set(title='Welch PSD' + '('+ str(meg) + ')', xlabel='Frequency (Hz)',
               ylabel='Power Spectral Density (dB)')
        plt.show()
    
    if operations_to_apply['ComputeTRF']:    
        # define frequencies of interest (log-spaced)
        freqs = np.logspace(*np.log10([MinFreq, MaxFreq]), num=20)
        n_cycles = freqs / 2.  # different number of cycle per frequency
        power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                return_itc=True, decim=3, n_jobs=1)
        
        power.crop(MinTime, MaxTime)  # crop the time to remove edge artifacts
        itc.crop(MinTime, MaxTime)  # crop the time to remove edge artifacts
        
        print(power,itc)
        
    if operations_to_apply['PlotPowerTopo']:
        power.plot_topo(baseline=baseline, mode=baseline_mode, title='Average power');
        
    if operations_to_apply['PlotPowerSensor']:
        power.plot([SensorID], baseline=baseline, mode=baseline_mode, title=power.ch_names[SensorID]);
    
    if operations_to_apply['PlotFreqBandTopo']:
        fig, axis = plt.subplots(1, 3, figsize=(7, 4))
        power.plot_topomap(ch_type='grad', tmin=FreqBMinTime, tmax=FreqBMaxTime, fmin=4, fmax=7,
                           baseline=baseline, mode=baseline_mode, axes=axis[0],
                           title='Theta', show=False, contours=1)
        power.plot_topomap(ch_type='grad', tmin=FreqBMinTime, tmax=FreqBMaxTime, fmin=8, fmax=12,
                           baseline=baseline, mode=baseline_mode, axes=axis[1],
                           title='Alpha', show=False, contours=1)
        power.plot_topomap(ch_type='grad', tmin=FreqBMinTime, tmax=FreqBMaxTime, fmin=15, fmax=30,
                           baseline=baseline, mode=baseline_mode, axes=axis[2],
                           title='Beta', show=False, contours=1)
        mne.viz.tight_layout()
        plt.show()
        
    if operations_to_apply['PlotTimeFreqAcrossSensors']:
        power.plot_joint(baseline=baseline, mode='mean', tmin=None, tmax=None,
                     timefreqs=[TimeFreq1, TimeFreq2])  # time (in s), freq (in Hz)
        
    if operations_to_apply['PlotITC']:
        itc.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1., cmap='Reds');

elif ProcessMode == 2:
    if operations_to_apply['PlotT1']:
        plotting.plot_anat(t1_fname);
        plt.show()
        
    if operations_to_apply['PlotT1Slices']:
        mne.viz.plot_bem(subject=str(subject), subjects_dir=subjects_dir,
                     orientation=str(sliceType));
    
    if operations_to_apply['CoReg']:                     
        mne.gui.coregistration(subject=str(subject), subjects_dir=subjects_dir, inst=raw_fname);
        
    if operations_to_apply['CheckCoReg']: 
        info = mne.io.read_info(raw_fname)
        fig = mne.viz.plot_alignment(info, trans_fname, subject=subject, dig=True,
                                     subjects_dir=subjects_dir, verbose=True);
                                     
    if operations_to_apply['PlotSourceSpace']:
        mne.set_log_level('WARNING')
        src = mne.setup_source_space(subject, spacing='oct6',
                                     subjects_dir=subjects_dir,
                                     add_dist=False)
        
        print(src)
        # ``src`` contains two parts, one for the left hemisphere (4098 locations) and one for the right hemisphere (4098 locations).
        info = mne.io.read_info(raw_fname)
        fig = mne.viz.plot_alignment(info, trans_fname, subject=subject, dig=False, src=src,
                                     subjects_dir=subjects_dir, verbose=True, meg=False,
                                     eeg=False);
        
        mne.viz.plot_alignment(info, trans_fname, subject=subject,
                               src=src, subjects_dir=subjects_dir, dig=True,
                               surfaces=['head-dense', 'white'], coord_frame='meg')
        
    if operations_to_apply['ComputeForwardSolution']:    
        conductivity = (0.3,)  # for single layer (MEG)
        # conductivity = (0.3, 0.006, 0.3)  # for three layers (EEG)
        model = mne.make_bem_model(subject=subject, ico=4,
                                   conductivity=conductivity,
                                   subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        
        mne.bem.write_bem_solution(bem_fname, bem)
        
        fwd = mne.make_forward_solution(raw_fname, trans=trans_fname,
                                        src=src, bem=bem,
                                        meg=True, # include MEG channels
                                        eeg=False, # exclude EEG channels
                                        mindist=5.0, # ignore sources <= 5mm from inner skull
                                        n_jobs=1) # number of jobs to run in parallel
        
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
        
    if operations_to_apply['LeadfieldMatrix']:
        fwd = mne.convert_forward_solution(fwd, surf_ori=True)
        leadfield = fwd['sol']['data']
        print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)  ## three times the number of vertices -- x, y, z
        
        
        # Compute sensitivity maps for gradiometers
        sens_map = mne.sensitivity_map(fwd, ch_type=str(SensorType), mode='fixed')
        
        
        # # Show gain matrix a.k.a. leadfield matrix with sensitivy map    
        picks = mne.pick_types(fwd['info'], meg=str(SensorType), eeg=False)
        
        im = plt.imshow(leadfield[picks, :500], origin='lower', aspect='auto', cmap='RdBu_r')
        plt.xlabel('sources')
        plt.ylabel('sensors')
        plt.title('Lead field matrix for Gradiometers', fontsize=14)
        plt.colorbar(cmap='RdBu_r')
        
        
        plt.figure()
        plt.hist(sens_map.data.ravel(), bins=20, label='Gradiometers',
                  color='c')
        plt.legend()
        plt.title('Normal orientation sensitivity')
        plt.xlabel('sensitivity')
        plt.ylabel('count');
        
    if operations_to_apply['Plot3DModel']:
        clim = dict(kind='percent', lims=(0.0, 50, 99), smoothing_steps=3)  # let's see single dipoles
        brain = sens_map.plot(subject=subject, time_label=str(SensorType) + ' sensitivity',
                              subjects_dir=subjects_dir, clim=clim, smoothing_steps=8);
        view = str(BrainView)
        brain.show_view(view)
        brain.save_image('sensitivity_map_' + str(SensorType) + '_%s.jpg' % view)
        Image(filename='sensitivity_map_' + str(SensorType) + '_%s.jpg' % view, width=400)