#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 08:34:09 2022

@author: abdulwahab

Trying to Visualize the feature importances by models.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import nibabel
import time


filename = '1K_rbf_RF_MAE_optimized_models.sav'
# filename = str(1) + 'K_SS_RF_MAE_optimized_models.sav'
path = '/Users/abdulwahab/Y3 Stuff/Project/DATA/Output opti models err r2/'

optimized_models = pickle.load(open(path+filename, 'rb'))


# model scores.
weights = [0.041044387,0.013893533,0.08688614]
# weights = [0.109751299,0.041044387,0.013893533,0.08688614]


count = 0

lasso = optimized_models[0+count] 
print('Lasso model: ', lasso)
rf = optimized_models[10+count] 
print('RF model: ', rf)
ada = optimized_models[15+count]
print('ADA model: ', ada)
gb = optimized_models[20+count]
print('GB model: ', gb)


fimps_lasso = lasso.coef_ 

fimps_rf = rf.feature_importances_

fimps_ada= ada.feature_importances_

fimps_gb = gb.feature_importances_

fimps = np.vstack((fimps_rf,fimps_ada,fimps_gb))
# fimps = np.vstack((fimps_lasso,fimps_rf,fimps_ada,fimps_gb))


fimps = np.transpose(fimps)

# geting the weighted average of the features
averaged_imps = np.average(fimps , weights = weights , axis =  1)

print(np.argmax(fimps_lasso))
print(np.argmax(fimps_rf))
print(np.argmax(fimps_ada))
print(np.argmax(fimps_gb))

# Setting up book to keep track of data.
book = [] 
for j in np.arange(1,181):
    for k in np.arange(1,113):
        book.append('L' + '-l' + str(j) + '-f' +str(k))
        book.append('R' + '-l' + str(j) + '-f' +str(k))

importance = averaged_imps

indices = np.argsort(importance)[::-1]
# Print the feature ranking
print("Feature ranking:")

for f in range(20):
    # print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))
    print("%d. feature %s (%f)" % (f + 1, book[indices[f]], importance[indices[f]]))
    
plt.figure(figsize=(10,8))
plt.title("Feature importances")
plt.bar(range(20), importance[indices][0:20], color="r", align="center")
plt.xticks(range(20), indices[0:20])
plt.xlim([-1, 20])
plt.show()

# Separating the important labels and important features into 2 lists
imp_lbls_L = []
imp_feats_L= []
imp_L = [] 

imp_lbls_R = []
imp_feats_R = []
imp_R = []

for m in range(100):
    
# m = 0 

# status = 0 

# while status == 0:
    
#     if len(imp_lbls_L) == 50:
#         status = -1
    
    if 'L' in book[indices[m]]:
        
        lbl = book[indices[m]].split('-')[1]  
        lnum = lbl.split('l')[1]
        imp_lbls_L.append( int(lnum)) 
    # print(lnum)
    
        feat = book[indices[m]].split('-')[2] 
        fnum = feat.split('f')[1]
        imp_feats_L.append(  int(fnum)    )
        
        val = importance[indices[m]]
        imp_L.append(val)
        
        
        
    if 'R' in book[indices[m]]:
    
        lbl = book[indices[m]].split('-')[1]  
        lnum = lbl.split('l')[1]
        imp_lbls_R.append( int(lnum)) 
    # print(lnum)
        feat = book[indices[m]].split('-')[2] 
        fnum = feat.split('f')[1]
        imp_feats_R.append(  int(fnum)    )
        
        val = importance[indices[m]]
        imp_R.append(val)
    
    # m = m + 1 

# Testing this mode finder
def pr_N_mostFrequentNumber(arr, n, k):
  
    um = {}
    for i in range(n):
        if arr[i] in um:
            um[arr[i]] += 1
        else:
            um[arr[i]] = 1
    a = [0] * (len(um))
    j = 0
    for i in um:
        a[j] = [i, um[i]]
        j += 1
    a = sorted(a, key=lambda x: x[0],
               reverse=True)
    a = sorted(a, key=lambda x: x[1],
               reverse=True)
  
    # display the top k numbers
    print(k, "numbers with most occurrences are:")
    for i in range(k):
        print(a[i][0], end=" ")

print('in L: ')
pr_N_mostFrequentNumber(imp_lbls_L , n = len(imp_feats_L) , k = 5)

print('\n in R: ')
pr_N_mostFrequentNumber(imp_lbls_R , n = len(imp_feats_R) , k = 5)

#%%

path_features = '/Users/abdulwahab/Y3 Stuff/Project/100307/'
path_labels = '/Users/abdulwahab/Y3 Stuff/Project/100307/'

featuresL = nibabel.load(path_features +'100307.L.Meaned.Features_fs_LR.func.gii')
featuresR = nibabel.load(path_features +'100307.R.Meaned.Features_fs_LR.func.gii')

labelL= nibabel.load(path_labels +'100307.L.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')
labelR = nibabel.load(path_labels +'100307.R.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')

# Setting up output files of featuresets.

outputL = nibabel.load(path_features +'100307.L.Meaned.Features_fs_LR.func.gii')
outputR = nibabel.load(path_features +'100307.R.Meaned.Features_fs_LR.func.gii')

label_names_L = labelL.labeltable.get_labels_as_dict()
label_names_R = labelR.labeltable.get_labels_as_dict()


#%%
# FOR LEFT HEMI.    ZEROING
for i in np.arange(1,len(label_names_L)):
        # Variable to give location of each label.
    label_pos = np.where(labelL.darrays[0].data == i )[0] 
        
    outputL.darrays[0].data[label_pos ] = 0.0 

# FOR RIGHT HEMI.    ZEROING
for i in np.arange(1,len(label_names_R)):
        # Variable to give location of each label.
    label_pos = np.where(labelR.darrays[0].data == i )[0] 
        
    outputR.darrays[0].data[label_pos ] = 0.0

    
# FOR LEFT HEMI. 
for f in range(0,len(imp_lbls_L)):
    
    # Variable to give location of each label.
    label_pos = np.where(labelL.darrays[0].data == imp_lbls_L[f]-1 )[0] 
    
    outputL.darrays[0].data[label_pos ] = featuresL.darrays[imp_feats_L[f]-1 ].data[label_pos ][0] * imp_L[f]

nibabel.save(outputL,path_labels+'100307.L.Scaled_50_Important.Features_fs_LR.func.gii')

    
# FOR RIGHT HEMI. 
for f in range(0,len(imp_lbls_R)):
    # print(f)
    # Variable to give location of each label.
    label_pos = np.where(labelR.darrays[0].data == imp_lbls_R[f]-1 )[0] 
    
    outputR.darrays[0].data[label_pos ] = featuresR.darrays[imp_feats_R[f]-1 ].data[label_pos ][0] * imp_R[f]


nibabel.save(outputR,path_labels+'100307.R.Scaled_50_Important.Features_fs_LR.func.gii')
