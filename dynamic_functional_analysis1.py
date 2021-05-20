#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:35:23 2021

@author: cjrichier
"""

import nibabel as nib
import nilearn as nl
import os
import numpy as np
import sys
import platform
import getpass
import pandas as pd
import datetime as dt
from nilearn.connectome import ConnectivityMeasure
from nilearn.connectome import sym_matrix_to_vec

sep = os.path.sep
sys_name = platform.system() 

session_n = 19
session = str(session_n).zfill(3)
user = getpass.getuser()
'''Set some directories'''
if user == 'kyle':
    clouddirr = f'C:{sep}Users{sep}kyle{sep}OneDrive - University of Illinois - Urbana{sep}MyConnectome{sep}'
    derivdirr = f'S:{sep}MyConnectome{sep}ds000031_R1.0.4{sep}derivatives{sep}'
    subjectdirr = f'{derivdirr}fmriprep_1.0.0{sep}fmriprep{sep}sub-01{sep}'
    sessiondirr = f'{subjectdirr}{session}{sep}func{sep}'#Do this in runtme
    #Confounds = f'{subjectdirr}' #unused
    Behavior = f'{derivdirr}fmriprep_1.0.0{sep}fmriprep{sep}trackingdata.txt'
else:
    subjectdirr = "/Volumes/Byrgenwerth/Datasets/My Connectome/master/sub-01/"
    #Make sessiondir in runtime
    #Confounds = '/Volumes/Byrgenwerth/Datasets/My Connectome/master/sub-01/ses-019/sub-01_ses-019_task-rest_run-001_motion-scrubbing-mask.csv' Unused
    Behavior = r'/Volumes/Byrgenwerth/Datasets/My Connectome/master/Behavior Data/Behavior.txt'

try:
    os.mkdir('{clouddirr}ses-{session}')
    os.mkdir('{clouddirr}ses-{session}{sep}func')
except:
    print('directory already exists')


session_list = []
for f in os.listdir(subjectdirr):
    if 'ses-' in f:
        session_list.append(f)


def load_timeseries(session, task):
    """Load timeseries data for a single run
    
    Args:
        session (int): 0-based session ID to load
        task (str): name of possible tasks to retrieve. either rest or nback
        
    Returns
        ts: A file of the subject's data'

    """
    if task == 'rest':
        bold_path = f"{subjectdirr}{session}{sep}func{sep}"
        bold_file = f"sub-01_{session}_task-rest_run-001_bold_space-T1w_preproc.nii.gz"
        
    elif task == 'nback': #unused
        bold_path = f"{subjectdirr}{session}{sep}func{sep}"
        bold_file = f"sub-01_{session}_task-nback_run-001_bold_space-T1w_preproc.nii.gz"
    print(f"{bold_path}{bold_file}")
    ts = nib.load(f"{bold_path}{bold_file}")
    return ts



      
def load_all_timeseries(session_list, task):
    '''Load in the entirety of the subject's data for one type of scan 
    into a list to be manipulated'''
    '''
    session_list = []
    with os.scandir(subjectdirr) as listOfEntries:
        for entry in listOfEntries:
        # print all entries that are files
            if 'ses' in entry.name:
                session_list.append(entry.name)
    '''
    '''load in the imaging data'''
    total_timeseries = []
    count_bad = 0
    for session in session_list:
        try:
            total_timeseries.append(load_timeseries(session, task))
            print("Loaded session", str(session))
        except:
            print("Session", str(session), "Doesn't exist. But that's fine.")
            count_bad+=1
            pass
    if count_bad>5:
        print('... Or is it?')
    return total_timeseries
    
if False:

    #time_series = masker.fit_transform(img_19, confounds=Confounds)
                                    

    timeseries_list = []        
    for ts in rest_list:
        timeseries_list.append(masker.fit_transform(ts))

    '''make the data a dictionary so that 
    it can have info about subject as well'''
    rest_data_dict = dict(zip(session_list, timeseries_list))
                                    
                                    
    from nilearn.connectome import ConnectivityMeasure
    correlation_measure = ConnectivityMeasure(kind='correlation')
    #correlation_matrix = correlation_measure.fit_transform([img_19])[0]            
            
    np.any(np.isnan(timeseries_list))
    missing = np.argwhere(np.isnan(timeseries_list))
        
    '''calculate the FC'''
    N_PARCELS = 48
    N_SCANS = len(session_list)
    fc_matrices = np.zeros((N_SCANS, N_PARCELS, N_PARCELS))   
    for subject, ts in enumerate(rest_data_dict.values()):
        fc_matrices[subject] = correlation_measure.fit_transform([ts])[1] 
    '''now we have an object, FC matrices, with every subjects' FC matrix'''


    for ts in rest_data_dict.values():
        correlation_measure.fit_transform([ts])[1]


    # Plot the correlation matrix
    import numpy as np
    from nilearn import plotting
    # Make a large figure
    # Mask the main diagonal for visualization:
    np.fill_diagonal(correlation_matrix, 0)
    # The labels we have start with the background (0), hence we skip the
    # first label
    # matrices are ordered for block-like representation
    plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                        vmax=0.8, vmin=-0.8, reorder=True)                  
                                    




    behavior_data = pd.read_csv(Behavior, header = None, delimiter = "/t")

    panas = behavior_data.iloc[:,np.r_[0, 29:90]]

    panas[0].replace('sub', 'ses')
    panas[0] = panas[0].str.replace(r'/D', '')

    #panas.columns = panas.iloc[0,]
    #panas.drop(rows=0, inplace=True)

    for value in panas[0]:
        if value.startswith('sub'):
            panas[value] = value[-3:]

def loadfiles(session_list):
    output = {}
    for session in session_list:
        sessiondirr = f'{subjectdirr}{session}{sep}func{sep}'
        timeseries = load_timeseries(session, 'rest')
        confound = f'{sessiondirr}sub-01_{session}_task-rest_run-001_bold_confounds.tsv'
        output[session] = {
            'session':session,
            'timeseries':timeseries,
            'confound':confound
            }
    return output

def categorical_thirds(z_score):
    if z_score<=-.43091:
        return -1
    elif z_score<-.43091 and z_score<.43091:
        return 0
    elif z_score>=.43091:
        return 1

if __name__ == '__main__':
    total_start_time = dt.datetime.now()
    rest_list = load_all_timeseries(session_list[:10],'rest')
    from nilearn import datasets
    dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    #mni_brain_mask = datasets.load_mni152_brain_mask()
    #mni_brain_mask_template = datasets.load_mni152_template()
    atlas_filename = dataset.maps
    labels = dataset.labels

    from nilearn import plotting
    plotting.plot_roi(atlas_filename)

    from nilearn.input_data import NiftiLabelsMasker
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
    #works to here
    timeseries_list = []        
    for ts in rest_list:
        timeseries_list.append(masker.fit_transform(ts))

    '''make the data a dictionary so that 
    it can have info about subject as well'''
    rest_data_dict = dict(zip(session_list, timeseries_list))
    np.save('S:\\rest_data_dict.npy', rest_data_dict)