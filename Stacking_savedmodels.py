#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 01:02:43 2022

@author: abdulwahab

Stacking previously saved models.
"""

# Stacking from saved optimized models file.
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import nibabel
import time

import functions as funcs 

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
    
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

r2_scores_stacked = []
corr_vec_stacked = []

start = time.time()

for kk in [1]:
    
    filename = str(kk) + 'K_rbf_RF_MAE_optimized_models.sav'
    path = '/Users/abdulwahab/Y3 Stuff/Project/DATA/Output opti models err r2/'
    
    optimized_models = pickle.load(open(path+filename, 'rb'))
    

    for jj in range(1,6):
        print('SPLIT: ', jj)
        
        # Setting the matrices according to the split number.
        X_train , y_train , X_test , y_test , X_valid , y_valid = funcs.TTV_split(jj)
        
        # # Scale X_train and X_valid
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_valid = scaler.transform(X_valid)
        
        # # SCALING EACH ONE SEPARATELY.
        # train_scaler = StandardScaler().fit(X_train)
        # X_train = train_scaler.transform(X_train)
        
        # test_scaler = StandardScaler().fit(X_test)
        # X_test = test_scaler.transform(X_test)
        
        # valid_scaler = StandardScaler().fit(X_valid)
        # X_valid = valid_scaler.transform(X_valid)
        
        # create ranking model
        # ranker = Lasso()
        ranker = RandomForestRegressor(random_state=42)
        
        selector = SelectFromModel(estimator = ranker , max_features = int(40320/kk),  threshold = -np.inf )
        
        selector.fit(X_train,y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        X_valid = selector.transform(X_valid)
    
        level0 = list()
    
        count = 0 
        
        for ii in np.arange(0+count,25+count,5):
            # print(i)
            # name = type(loaded_models[i]).__name__
            name = type(optimized_models[ii]).__name__
            
            if name == 'KernelRidge' or name == 'Lasso' or name == 'AdaBoostRegressor' :
                continue
            
            # print(name)
            # print(loaded_models[i])
            # level0.append((name,loaded_models[i]))
            level0.append((name,optimized_models[ii]))
            count = count + 1
            
        # define meta learner model
        level1 = LinearRegression()
        
        stacker = StackingRegressor(estimators = level0, final_estimator = level1 )
                
        stacker.fit(X_train,y_train)
        
        r2 = stacker.score(X_test,y_test)
        r2_scores_stacked.append((kk,jj,r2))
        
        corr = np.sqrt(r2)
        corr_vec_stacked.append((kk,jj,corr))
        
print('TIME TAKEN: ', time.time() - start )

