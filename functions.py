#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 22:21:25 2021

@author: abdulwahab

FILE CONTAINING FUNCTIONS CREATED THROUGHOUT THE PROJECT.
"""


import pandas as pd
import numpy as np
import nibabel
import copy

def features_averaged(featuresL,featuresR,labelL,labelR):
    
    outputL = copy.deepcopy(featuresL)
    outputR = copy.deepcopy(featuresR)
    
    label_names_L = labelL.labeltable.get_labels_as_dict()
    label_names_R = labelR.labeltable.get_labels_as_dict()
    
        # FOR LEFT HEMI.
    for i in np.arange(1,len(label_names_L)):
        for j in range(featuresL.numDA):
            
            # Variable to give location of each label.
            label_pos = np.where(labelL.darrays[0].data == i )[0] 
            
            # Setting mean value of each feature for the corresponding feature in all labels
            temp_mean_L = np.mean(featuresL.darrays[j].data[label_pos ])
            #np.mean(abs(featuresL.darrays[j].data[label_pos]))
            outputL.darrays[j].data[label_pos ] = temp_mean_L
            
    # NOW FOR RIGHT HEMI.
    
    for i in np.arange(1,len(label_names_R)):
        for j in range(featuresR.numDA):
            
            # Variable to give location of each label.
            label_pos = np.where(labelR.darrays[0].data == i )[0] 
            
            # Setting mean value of each feature for the corresponding feature in all labels
            temp_mean_R = np.mean(featuresR.darrays[j].data[label_pos ])
            #np.mean(abs(featuresR.darrays[j].data[label_pos]))
            outputR.darrays[j].data[label_pos ] = temp_mean_R

    return outputL , outputR


def features_std(featuresL,featuresR,labelL,labelR):
    
    outputL = copy.deepcopy(featuresL)
    outputR = copy.deepcopy(featuresR)
    
    label_names_L = labelL.labeltable.get_labels_as_dict()
    label_names_R = labelR.labeltable.get_labels_as_dict()
    
        # FOR LEFT HEMI.
    for i in np.arange(1,len(label_names_L)):
        for j in range(featuresL.numDA):
            
            # Variable to give location of each label.
            label_pos = np.where(labelL.darrays[0].data == i )[0] 
            
            # Setting std value of each feature for the corresponding feature in all labels
            temp_std_L = np.std(featuresL.darrays[j].data[label_pos])
            outputL.darrays[j].data[label_pos ] = temp_std_L
            
    # NOW FOR RIGHT HEMI.
    
    for i in np.arange(1,len(label_names_R)):
        for j in range(featuresR.numDA):
            
            # Variable to give location of each label.
            label_pos = np.where(labelR.darrays[0].data == i )[0] 
            
            # Setting std value of each feature for the corresponding feature in all labels
            temp_std_R = np.std(featuresR.darrays[j].data[label_pos])
            outputR.darrays[j].data[label_pos ] = temp_std_R

    return outputL , outputR
            
# FUNCTION TO MAKE THE FEATURE MATRICES OF THE DATA.
def X_maker(sheet_num):
    
    # Sheet_num: represents whether to load in training or testing data.
    # colL : column number of column containing features of L 
    # colR : column number of column containing features of R
    
    # OUTPUTS X_TRAIN, X_TEST, OR X_VALID DEPENDING ON SHEET NUM.
        
    # file_name =  # path to file + file name
    file_name = '/Users/abdulwahab/Y3 Stuff/Project/DATA/DATASETS.xlsx'
    df = pd.read_excel(io=file_name, sheet_name = sheet_num  )
    # Sheet_num: represents whether to load in training, testing or validation data.
    data = df.to_numpy()
    
    Feature_filenames_L = data[:,0]
    Feature_filenames_R = data[:,1]
    
    Labels_filenames_L = data[:,2]
    Labels_filenames_R = data[:,3]
    
    
    if(sheet_num == 0):
         # TRAINING DATA PATH
         # path_features = '/Users/abdulwahab/Y3 Stuff/Project/Trial folder/Training data/Meaned features/'
         # path_features = '/Users/abdulwahab/OneDrive - King\'s College London/My files/HCP_PARCELLATION/TRAININGDATA/featuresets/'
         # path_labels = '/Users/abdulwahab/OneDrive - King\'s College London/My files/HCP_PARCELLATION/TRAININGDATA/classifiedlabels/'
         
         path_features = '/Users/abdulwahab/Y3 Stuff/Project/DATA/ALL featuresets/'
         path_labels = '/Users/abdulwahab/Y3 Stuff/Project/DATA/ALL labels/'

         
    if(sheet_num == 1):
         # TESTING DATA PATH
         # path_features = '/Users/abdulwahab/OneDrive - King\'s College London/My files/HCP_PARCELLATION/TESTINGDATA/featuresets/'
         # path_features = '/Users/abdulwahab/Y3 Stuff/Project/Trial folder/Testing data/Meaned features/'
         # path_labels = '/Users/abdulwahab/OneDrive - King\'s College London/My files/HCP_PARCELLATION/TESTINGDATA/classifiedlabels/'
         
         path_features = '/Users/abdulwahab/Y3 Stuff/Project/DATA/ALL featuresets/'
         path_labels = '/Users/abdulwahab/Y3 Stuff/Project/DATA/ALL labels/'
         
    if(sheet_num == 2):
          # VALIDATION DATA PATH
          path_features = '/Users/abdulwahab/Y3 Stuff/Project/DATA/ALL featuresets/'
          path_labels = '/Users/abdulwahab/Y3 Stuff/Project/DATA/ALL labels/'
       
    
    
    # 210 samples in rows and 360x112 in columns
    # L_l1_f1 / R_l1_f1 /L_l1_f2 / R_l1_f2
    
    X_out = np.zeros([180*112*2])
        
    for m in range(len(Feature_filenames_L)):
        
        # LOADING UP.
        featuresL = nibabel.load(path_features+Feature_filenames_L[m])
        featuresR = nibabel.load(path_features+Feature_filenames_R[m])
        
        labelL = nibabel.load(path_labels+Labels_filenames_L[m])
        labelR = nibabel.load(path_labels+Labels_filenames_R[m])
        
        label_names_L = labelL.labeltable.get_labels_as_dict()
        label_names_R = labelR.labeltable.get_labels_as_dict()
        np.array([0])

        
        if (len(label_names_L) == 181):    
            row = [] 
            for i in np.arange(1,len(label_names_L)):
                for j in range(featuresL.numDA):
                        
                    # Variable to give location of each label.
                    label_pos_L = np.where(labelL.darrays[0].data == i )[0]
                    
                    # fj_li_L = featuresL.darrays[j].data[label_pos_L]                 
                    # val_L = fj_li_L[0]
                    
                    temp_mean_L = np.mean(featuresL.darrays[j].data[label_pos_L])
                    # temp_mean_L = np.mean(abs(featuresL.darrays[j].data[label_pos_L]))
        
                    label_pos_R = np.where(labelR.darrays[0].data == i )[0]
                    
                    temp_mean_R = np.mean(featuresR.darrays[j].data[label_pos_R])
                    # temp_mean_R = np.mean(abs(featuresR.darrays[j].data[label_pos_R]))
        
                    # fj_li_R = featuresR.darrays[j].data[label_pos_R] 
                    # val_R = fj_li_R[0]
        
                    row.append(temp_mean_L)
                    row.append(temp_mean_R)
                    # row.append(val_L)
                    # row.append(val_R)
                
        if (len(label_names_L) == 361):   
            row = [] 
            for i in np.arange(1,len(label_names_L)/2):
                for j in range(featuresL.numDA):
                        
                    # Variable to give location of each label.
                    label_pos_L = np.where(labelL.darrays[0].data == (180+i) )[0]
                    
                    # fj_li_L = featuresL.darrays[j].data[label_pos_L]                 
                    # val_L = fj_li_L[0]
                    
                    temp_mean_L = np.mean(featuresL.darrays[j].data[label_pos_L])
                    # temp_mean_L = np.mean(abs(featuresL.darrays[j].data[label_pos_L]))
        
                    label_pos_R = np.where(labelR.darrays[0].data == i )[0]
                    
                    temp_mean_R = np.mean(featuresR.darrays[j].data[label_pos_R])
                    # temp_mean_R = np.mean(abs(featuresR.darrays[j].data[label_pos_R]))
        
                    # fj_li_R = featuresR.darrays[j].data[label_pos_R] 
                    # val_R = fj_li_R[0]
        
                    row.append(temp_mean_L)
                    row.append(temp_mean_R)
                    # row.append(val_L)
                    # row.append(val_R)


        X_out = np.vstack([X_out,row])
                          
        print('Percent Completed:' ,(m*100)/len(Feature_filenames_L))
    
    X_out = np.delete(X_out, (0), axis=0)
    
    return X_out
                
def sex_extractor(sheet_num):
    
    # Sheet_num: represents whether to load in training or testing data.
        
    info_file_name = '/Users/abdulwahab/Y3 Stuff/Project/HCP_s1200_unrestricted (1).csv'
    df = pd.read_csv(info_file_name)
    
    info_data = df.to_numpy()
    
    file_name = '/Users/abdulwahab/Y3 Stuff/Project/features_filename.xlsx'
    
    df = pd.read_excel(io = file_name, sheet_name = sheet_num)
    data  = df.to_numpy()
    
    Feature_filenames_L = data[:,0]
    
    # ALL the cases
    cases = info_data[:,0]
    # Converting cases from string to int
    for k in range(len(cases)):
          int(cases[k])
    
    
    # GETTING THE RELEVANT DATA.
    relevant_cases = []
    for m in range(len(Feature_filenames_L)):
        
        # Extracting case number.
        string = Feature_filenames_L[m]
        string1 = string.split(".")
        relevant_cases.append(int(string1[0]))
    
    
    # Getting the location index of relevant cases
    ind = []
    for i in range(len(relevant_cases)):
        for j in range(len(cases)):
            if(relevant_cases[i] == cases[j]):
                ind.append(j)  
                
    # Extracting gender info of relevant cases
    y_out = np.zeros(len(ind))
    for i in range(len(ind)):
        if (info_data[:,3][ind[i]] == 'F') :
            y_out[i] = 1 
    
    
    return y_out
    
def pmat_extractor(sheet_num):
    
     # Sheet_num: represents whether to load in training or testing data.
         
     info_file_name = '/Users/abdulwahab/Y3 Stuff/Project/HCP_s1200_unrestricted (1).csv'
     df = pd.read_csv(info_file_name)
     
     info_data = df.to_numpy()
     
     file_name = '/Users/abdulwahab/Y3 Stuff/Project/DATA/DATASETS.xlsx'
     
     df = pd.read_excel(io = file_name, sheet_name = sheet_num)
     data  = df.to_numpy()
     
     Feature_filenames_L = data[:,0]
     
     # ALL the cases
     cases = info_data[:,0]
     # Converting cases from string to int
     for k in range(len(cases)):
           int(cases[k])
     
     
     # GETTING THE RELEVANT DATA.
     relevant_cases = []
     for m in range(len(Feature_filenames_L)):
         
         # Extracting case number.
         string = Feature_filenames_L[m]
         string1 = string.split(".")
         relevant_cases.append(int(string1[0]))
     
     
     # Getting the location index of relevant cases
     ind = []
     for i in range(len(relevant_cases)):
         for j in range(len(cases)):
             if(relevant_cases[i] == cases[j]):
                 ind.append(j)  
                 
     # Extracting data of relevant cases, 
     # here PMAT24_A_CR is column 120
     # PMAT24_A_SI is column 121
     # PMAT24_A_RTCR is column 122
     y_out = np.zeros(len(ind))
     for i in range(len(ind)):
         # if (info_data[:,3][ind[i]] == 'F') :
         y_out[i] = info_data[:,120][ind[i]]
    
    
     return y_out 

# FUNCTION TO GENERATE THE FEATURE MATRICES OF DATA ACCORDING TO THE SPLITS.
def X_maker_splits(sheet_num):
    '''Generates matrices containing 
    rows = number of samples
    columns = features*labels
    --------------------------
    X_maker for splits.
    '''
    # Sheet_num: represents which split to load in. 
    # colL : column number of column containing features of L 
    # colR : column number of column containing features of R

    # OUTPUTS X_TRAIN, X_TEST, OR X_VALID DEPENDING ON SHEET NUM.

    # file_name =  # path to file + file name
    file_name = '/Users/abdulwahab/Y3 Stuff/Project/DATA/Crossval Splits.xlsx'
    df = pd.read_excel(io=file_name, sheet_name = sheet_num  )
    # Sheet_num: represents which split is loaded in
    data = df.to_numpy()
    
    Feature_filenames_L = data[:,0]
    Feature_filenames_R = data[:,1]
    
    Labels_filenames_L = data[:,2]
    Labels_filenames_R = data[:,3]
    
    path_features = '/Users/abdulwahab/Y3 Stuff/Project/DATA/ALL featuresets/'
    path_labels = '/Users/abdulwahab/Y3 Stuff/Project/DATA/ALL labels/'
        
    # samples in rows and 360x112 in columns
    # L_l1_f1 / R_l1_f1 /L_l1_f2 / R_l1_f2
    
    X_out = np.zeros([180*112*2])
        
    for m in range(len(Feature_filenames_L)):
        
        # LOADING UP.
        featuresL = nibabel.load(path_features+Feature_filenames_L[m])
        featuresR = nibabel.load(path_features+Feature_filenames_R[m])
        
        labelL = nibabel.load(path_labels+Labels_filenames_L[m])
        labelR = nibabel.load(path_labels+Labels_filenames_R[m])
        
        label_names_L = labelL.labeltable.get_labels_as_dict()
        label_names_R = labelR.labeltable.get_labels_as_dict()
        np.array([0])

        
        if (len(label_names_L) == 181):    
            row = [] 
            for i in np.arange(1,len(label_names_L)):
                for j in range(featuresL.numDA):
                # for j in range(100): # For removing the last 12 features.
                        
                    # Variable to give location of each label.
                    label_pos_L = np.where(labelL.darrays[0].data == i )[0]
                    
                    # fj_li_L = featuresL.darrays[j].data[label_pos_L]                 
                    # val_L = fj_li_L[0]
                    
                    temp_mean_L = np.mean(featuresL.darrays[j].data[label_pos_L])
                    # temp_mean_L = np.mean(abs(featuresL.darrays[j].data[label_pos_L]))
                    # temp_std_L = np.std(featuresL.darrays[j].data[label_pos_L])
        
                    label_pos_R = np.where(labelR.darrays[0].data == i )[0]
                    
                    temp_mean_R = np.mean(featuresR.darrays[j].data[label_pos_R])
                    # temp_mean_R = np.mean(abs(featuresR.darrays[j].data[label_pos_R]))
                    # temp_std_R = np.std(featuresR.darrays[j].data[label_pos_R])
        
                    # fj_li_R = featuresR.darrays[j].data[label_pos_R] 
                    # val_R = fj_li_R[0]
        
                    # FOR MEAN
                    row.append(temp_mean_L)
                    row.append(temp_mean_R)
                    
                    # FOR STD 
                    # row.append(temp_std_L)
                    # row.append(temp_std_R)
                
        if (len(label_names_L) == 361):   
            row = [] 
            for i in np.arange(1,len(label_names_L)/2):
                for j in range(featuresL.numDA):
                # for j in range(100): # For removing the last 12 features.
                        
                    # Variable to give location of each label.
                    label_pos_L = np.where(labelL.darrays[0].data == (180+i) )[0]
                    
                    # fj_li_L = featuresL.darrays[j].data[label_pos_L]                 
                    # val_L = fj_li_L[0]
                    
                    temp_mean_L = np.mean(featuresL.darrays[j].data[label_pos_L])
                    # temp_mean_L = np.mean(abs(featuresL.darrays[j].data[label_pos_L]))
                    # temp_std_L = np.std(featuresL.darrays[j].data[label_pos_L])
        
                    label_pos_R = np.where(labelR.darrays[0].data == i )[0]
                    
                    temp_mean_R = np.mean(featuresR.darrays[j].data[label_pos_R])
                    # temp_mean_R = np.mean(abs(featuresR.darrays[j].data[label_pos_R]))
                    # temp_std_R = np.std(featuresR.darrays[j].data[label_pos_R])
        
                    # fj_li_R = featuresR.darrays[j].data[label_pos_R] 
                    # val_R = fj_li_R[0]
        
                    # FOR MEAN
                    row.append(temp_mean_L)
                    row.append(temp_mean_R)
                    
                    # FOR STD
                    # row.append(temp_std_L)
                    # row.append(temp_std_R)


        X_out = np.vstack([X_out,row])
                          
        print('Percent Completed:' ,(m*100)/len(Feature_filenames_L))
    
    X_out = np.delete(X_out, (0), axis=0)
    
    return X_out
     
# FUNCTION TO EXTRACT THE PMAT SCORES OF SUBJECTS FROM THE HCP DATA FILES.
def pmat_extractor_splits(sheet_num):
     # Sheet_num: represents which split we are working on.
         
     info_file_name = '/Users/abdulwahab/Y3 Stuff/Project/HCP_s1200_unrestricted (1).csv'
     df = pd.read_csv(info_file_name)
     
     info_data = df.to_numpy()
     
     file_name = '/Users/abdulwahab/Y3 Stuff/Project/DATA/Crossval Splits.xlsx'
     
     df = pd.read_excel(io = file_name, sheet_name = sheet_num)
     data  = df.to_numpy()
     
     Feature_filenames_L = data[:,0]
     
     # ALL the cases
     cases = info_data[:,0]
     # Converting cases from string to int
     for k in range(len(cases)):
           int(cases[k])
     
     
     # GETTING THE RELEVANT DATA.
     relevant_cases = []
     for m in range(len(Feature_filenames_L)):
         
         # Extracting case number.
         string = Feature_filenames_L[m]
         string1 = string.split(".")
         relevant_cases.append(int(string1[0]))
     
     
     # Getting the location index of relevant cases
     ind = []
     for i in range(len(relevant_cases)):
         for j in range(len(cases)):
             if(relevant_cases[i] == cases[j]):
                 ind.append(j)  
                 
     # Extracting data of relevant cases, 
     # here PMAT24_A_CR is column 120
     # PMAT24_A_SI is column 121
     # PMAT24_A_RTCR is column 122
     y_out = np.zeros(len(ind))
     for i in range(len(ind)):
         # if (info_data[:,3][ind[i]] == 'F') :
         y_out[i] = info_data[:,120][ind[i]]
    
    
     return y_out     
        
#%% FUNCTION FOR HYPERPARAMETER OPTIMIZATION

# Needed for functions to work.
# -----------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
# -----------------------------------------------------


def hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid):
    # Getting the name of the model.
    name = type(model).__name__
    
    graph_x = []
    graph_y = []
    graph_z = []
    
    # Lasso Functionality
    if name == 'Lasso' or name == 'Ridge' :
        for alpha_value in np.arange(-5.0,2.5,0.5):
            alpha_value = pow(10,alpha_value)
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []

            hyperparams = alpha_value
            rmse = Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid)
            graph_x_row.append(alpha_value)
            graph_z_row.append(rmse)
    
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)
            print('')
            
            
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        opti_alpha = graph_x[pos_min_z[0], pos_min_z[1]]
        # opti_gamma = graph_y[pos_min_z[0], pos_min_z[1]]
        print('Minimum RMSE: %.4f' %(min_z))
        # print('Optimum alpha: %f' %(graph_x[pos_min_z[0],pos_min_z[1]]))
        # print('Optimum gamma: %f' %(graph_y[pos_min_z[0],pos_min_z[1]]))
        print('Optimum alpha: %f' %(opti_alpha))
        # print('Optimum gamma: %f' %(opti_gamma))
        return opti_alpha 
    
    # KernelRidge Functionality
    if name == 'KernelRidge':
        for alpha_value in np.arange(-5.0,2.5,0.5):
            alpha_value = pow(10,alpha_value)
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []
            
            for gamma_value in np.arange(0.0,22,2):
                hyperparams = (alpha_value,gamma_value)
                rmse = Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid)
                graph_x_row.append(alpha_value)
                graph_y_row.append(gamma_value)
                graph_z_row.append(rmse)
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)
            print('')
            
            
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        opti_alpha = graph_x[pos_min_z[0], pos_min_z[1]]
        opti_gamma = graph_y[pos_min_z[0], pos_min_z[1]]
        print('Minimum RMSE: %.4f' %(min_z))
        print('Optimum alpha: %f' %(opti_alpha))
        print('Optimum gamma: %f' %(opti_gamma))
        return opti_alpha , opti_gamma
        
    # RandomForestRegressor Functionality
    if name == 'RandomForestRegressor':
        # for max_feats in [100,250,500,750,1000,1250,1500]:
        for max_feats in [100,150,200,250,300]:
            # n_estim = pow(100,n_estim)
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []
            
            for min_samp_leaf in np.arange(5,25,5):
                hyperparams = (max_feats,min_samp_leaf)
                rmse = Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid)
                graph_x_row.append(max_feats)
                graph_y_row.append(min_samp_leaf)
                graph_z_row.append(rmse)
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)
            print('')
            
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        opti_feats = graph_x[pos_min_z[0], pos_min_z[1]]
        opti_samps = graph_y[pos_min_z[0], pos_min_z[1]]
        print('Minimum RMSE: %.4f' %(min_z))
        print('Optimum max features: %f' %(opti_feats))
        print('Optimum min_samples_leaf: %f' %(opti_samps))
        return opti_feats , opti_samps
    
    # AdaBoostRegressor Functionality
    if name == 'AdaBoostRegressor':
        for n_estims in [50,75,100,125,150]:
            # n_estim = pow(100,n_estim)
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []
            
            for loss in ['linear','square','exponential']:
                hyperparams = (n_estims,loss)
                rmse = Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid)
                graph_x_row.append(n_estims)
                graph_y_row.append(loss)
                graph_z_row.append(rmse)
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)
            print('')
            
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        opti_estims = graph_x[pos_min_z[0], pos_min_z[1]]
        opti_loss = graph_y[pos_min_z[0], pos_min_z[1]]
        print('Minimum RMSE: %.4f' %(min_z))
        print('Optimum num estimators: %f' %(opti_estims))
        print('Optimum loss: %s' %(opti_loss))
        return opti_estims , opti_loss
    
    if name == 'GradientBoostingRegressor':
        # for max_feats in [100,250,500,750,1000,1250,1500]:
        # Below loop for when k is 32,64,128 
        for max_feats in [100,150,200,250,300]:
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []
            
            for min_samp_leaf in np.arange(5,25,5):
                hyperparams = (max_feats,min_samp_leaf)
                rmse = Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid)
                graph_x_row.append(max_feats)
                graph_y_row.append(min_samp_leaf)
                graph_z_row.append(rmse)
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)
            print('')
            
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        opti_feats = graph_x[pos_min_z[0], pos_min_z[1]]
        opti_samps = graph_y[pos_min_z[0], pos_min_z[1]]
        print('Minimum RMSE: %.4f' %(min_z))
        print('Optimum max features: %f' %(opti_feats))
        print('Optimum min samp leaf: %f' %(opti_samps))
        return opti_feats , opti_samps
    
    # XBGBoostRegressor Functionality
    if name == 'XGBoostRegressor':
        for min_weight in [100,250,500,750,1000,1250,1500]:
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []
            
            for max_depth in np.arange(5,25,5):
                hyperparams = (min_weight,max_depth)
                rmse = Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid)
                graph_x_row.append(max_feats)
                graph_y_row.append(min_samp_leaf)
                graph_z_row.append(rmse)
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)
            print('')
            
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        opti_weight = graph_x[pos_min_z[0], pos_min_z[1]]
        opti_depth = graph_y[pos_min_z[0], pos_min_z[1]]
        print('Minimum RMSE: %.4f' %(min_z))
        print('Optimum min weight: %f' %(opti_weight))
        print('Optimum max depth %f' %(opti_depth))
        return opti_weight , opti_depth
        

def Man_Gridsearch(model,hyperparams,X_train,y_train,X_valid,y_valid):
    # Getting the name of the model.
    name = type(model).__name__
    print(name)

    # Feed X_VALID and y_valid to actually check each optimizing combination.
    
    # Maybe try feature selection here with Kbest or feature importance from Random forest.
    if name == 'XGBoostRegressor':
        # Assign hyper-parameters
        min_weight,max_depth = hyperparams
        model.set_params(min_child_weight = min_weight, max_depth = max_depth , random_state = 42) 
    
    if name == 'GradientBoostingRegressor':
        # Assign hyper-parameters
        max_feats,min_samp_leaf = hyperparams
        model.set_params(n_estimators = 100,  max_features = max_feats ,min_samples_leaf = min_samp_leaf, random_state = 42)
        
    if name == 'AdaBoostRegressor':
        # Assign hyper-parameters
        n_estims,loss = hyperparams
        model.set_params(n_estimators = n_estims, loss = loss ,random_state = 42)

    if name == 'RandomForestRegressor':
        # Assign hyper-parameters
        max_feats,min_samp_leaf = hyperparams
        model.set_params(n_estimators = 100, max_features = max_feats , min_samples_leaf = min_samp_leaf, random_state = 42) 
    
    if name == 'KernelRidge':
        # Assign hyper-parameters
        alpha_value,gamma_value = hyperparams
        model.set_params(kernel = 'rbf' , alpha = alpha_value, gamma = gamma_value)
        
    if name == 'Ridge' or name == 'Lasso':
        # Assign hyperparameters
        alpha_value = hyperparams
        model.set_params(alpha = alpha_value)
            
    # model.fit(X_train_scaled,y_train)
    model.fit(X_train,y_train)

    # y_pred  = model.predict(X_valid_scaled)
    y_pred  = model.predict(X_valid)

    
    # Calculate error metric of test and predicted values: rmse
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    r_pearson,_=pearsonr(y_valid,y_pred)
    
    if name == 'KernelRidge':
        print('Kernel Ridge. alpha: %7.6f, gamma: %7.4f, RMSE: %7.4f, r: %7.4f' %(alpha_value,gamma_value,rmse,r_pearson))
        
    if name == 'RandomForestRegressor':
        print('Random Forest Regressor. Max_feats: %7.6f, Min_samps: %7.4f, RMSE: %7.4f, r: %7.4f' %(max_feats, min_samp_leaf ,rmse,r_pearson))
        
    if name == 'AdaBoostRegressor':
        print('AdaBoost Regressor. n_estims: %7.6f, loss: %s, RMSE: %7.4f, r: %7.4f' %(n_estims, loss ,rmse,r_pearson))
    
    if name == 'GradientBoostingRegressor':
        print('Gradient Boosting Regressor. Max_feats: %7.6f, Min_samps: %7.4f, RMSE: %7.4f, r: %7.4f' %(max_feats, min_samp_leaf ,rmse,r_pearson))
        
    if name == 'Ridge' or name ==  'Lasso':
        print('alpha: %7.6f, RMSE: %7.4f, r: %7.4f' %(alpha_value,rmse,r_pearson))
        
    return rmse

#%%
 # FUNCTION FOR TTV SPLIT
def TTV_split(split_num):
    """A function that sets the train, test, and validation matrices according to split number."""
   
    X_split1= np.load('X_split1.npy')
    X_split2= np.load('X_split2.npy')
    X_split3= np.load('X_split3.npy')
    X_split4= np.load('X_split4.npy')
    X_split5= np.load('X_split5.npy')
    
    y_split1 = np.load('PMAT_y_split1.npy')
    y_split2 = np.load('PMAT_y_split2.npy')
    y_split3 = np.load('PMAT_y_split3.npy')
    y_split4 = np.load('PMAT_y_split4.npy')
    y_split5 = np.load('PMAT_y_split5.npy')
    
    # Dealing with NaN and infinite values.
    X_split1 = np.nan_to_num(X_split1)
    X_split2 = np.nan_to_num(X_split2)
    X_split3 = np.nan_to_num(X_split3)
    X_split4 = np.nan_to_num(X_split4)
    X_split5 = np.nan_to_num(X_split5)
    
    if split_num == 1:
        X_train = np.concatenate((X_split3,X_split4,X_split5)) 
        y_train = np.concatenate((y_split3,y_split4,y_split5)) 
        
        X_test = X_split1
        y_test = y_split1
        
        X_valid = X_split2
        y_valid = y_split2
    
    
    if split_num == 2:
        X_train = np.concatenate((X_split1,X_split4,X_split5)) 
        y_train = np.concatenate((y_split1,y_split4,y_split5))
        
        X_test = X_split2
        y_test = y_split2
        
        X_valid = X_split3
        y_valid = y_split3
        
    if split_num == 3:
        X_train = np.concatenate((X_split1,X_split2,X_split5)) 
        y_train = np.concatenate((y_split1,y_split2,y_split5)) 
        
        X_test = X_split3
        y_test = y_split3
                
        X_valid = X_split4
        y_valid = y_split4
    
    if split_num == 4:
        X_train = np.concatenate((X_split1,X_split2,X_split3)) 
        y_train = np.concatenate((y_split1,y_split2,y_split3)) 
        
        X_test = X_split4
        y_test = y_split4
        
        X_valid = X_split5
        y_valid = y_split5
        
    if split_num == 5:
        X_train = np.concatenate((X_split2,X_split3,X_split4)) 
        y_train = np.concatenate((y_split2,y_split3,y_split4))
        
        X_test = X_split5
        y_test = y_split5
        
        X_valid = X_split1
        y_valid = y_split1
        
    return X_train , y_train , X_test , y_test , X_valid , y_valid

     
     
         