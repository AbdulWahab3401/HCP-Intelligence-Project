#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:57:04 2022

@author: abdulwahab

File containing the main code used in the project. Performed and tested feature selection, scaling, error metrics
and model optimization to obtain individual and stacked model scores on different splits.
"""

#%% IMPORTING BASIC MODULES
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import nibabel
import time

import functions as funcs 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#%% LOADING IN THE DATASETS
X_split1= np.load('X_split1.npy')
X_split2= np.load('X_split2.npy')
X_split3= np.load('X_split3.npy')
X_split4= np.load('X_split4.npy')
X_split5= np.load('X_split5.npy')


# LOADING IN SPLITS WITH 100 FEATURES.
# X_split1 = np.load('/Users/abdulwahab/Y3 Stuff/Project/DATA/Splits with 100 features/X_split1.npy')
# X_split2= np.load('/Users/abdulwahab/Y3 Stuff/Project/DATA/Splits with 100 features/X_split2.npy')
# X_split3= np.load('/Users/abdulwahab/Y3 Stuff/Project/DATA/Splits with 100 features/X_split3.npy')
# X_split4= np.load('/Users/abdulwahab/Y3 Stuff/Project/DATA/Splits with 100 features/X_split4.npy')
# X_split5= np.load('/Users/abdulwahab/Y3 Stuff/Project/DATA/Splits with 100 features/X_split5.npy')


y_split1 = np.load('PMAT_y_split1.npy')
y_split2 = np.load('PMAT_y_split2.npy')
y_split3 = np.load('PMAT_y_split3.npy')
y_split4 = np.load('PMAT_y_split4.npy')
y_split5 = np.load('PMAT_y_split5.npy')

# y_split1 = funcs.pmat_extractor_splits(0)
# y_split2 = funcs.pmat_extractor_splits(1)
# y_split3 = funcs.pmat_extractor_splits(2)
# y_split4 = funcs.pmat_extractor_splits(3)
# y_split5 = funcs.pmat_extractor_splits(4)

# Dealing with NaN and infinite values.
X_split1 = np.nan_to_num(X_split1)
X_split2 = np.nan_to_num(X_split2)
X_split3 = np.nan_to_num(X_split3)
X_split4 = np.nan_to_num(X_split4)
X_split5 = np.nan_to_num(X_split5)


#%% FUNCTION FOR TTV SPLIT
def TTV_split(split_num):
    """A function that sets the train, test, and validation matrices according to split number."""
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

#%% FUNCTION FOR HYPERPARAMETER OPTIMIZATION USING MANUAL COMBINATIONS

# Needed for functions to work.
# -----------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
from scipy.stats import pearsonr
# -----------------------------------------------------


def hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid):
    '''
    The main function performing manaual gridsearch for optimal hyperparameters.
    inputs = (model,X_train,y_train,X_valid,y_valid)
    outputs = (opti_param1, opti_param2)
    '''
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
        
        array_feats = [100,250,500,750,1000,1250,1500]
        if X_train.shape[1] <= 1260 :
            array_feats = [100,150,200,250,300]
        
        # Below for PCA 
        # array_feats = [1]
        for max_feats in array_feats:
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
        
        array_feats = [100,250,500,750,1000,1250,1500]
        if X_train.shape[1] <= 1260 :
            array_feats = [100,150,200,250,300]
        
        # Below for PCA 
        # array_feats = [1]
        
        for max_feats in array_feats:
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
    '''
    Sub function used in hyperparams_gridsearch for calculating error metric for a certain combination of parameters.
    '''
    # Getting the name of the model.
    name = type(model).__name__
    print(name)

    # Feed X_VALID and y_valid to actually check each  combination's result.

    # Maybe try feature selection here with Kbest or feature importance from Random forest.
    if name == 'XGBoostRegressor':
        # Assign hyper-parameters
        min_weight,max_depth = hyperparams
        model.set_params(min_child_weight = min_weight, max_depth = max_depth , random_state = 42) 
    
    if name == 'GradientBoostingRegressor':
        # Assign hyper-parameters
        max_feats,min_samp_leaf = hyperparams
        model.set_params(n_estimators = 100,  max_features = max_feats ,min_samples_leaf = min_samp_leaf, random_state = 42)
        # below for PCA --> removing max_features parameter
        # model.set_params(n_estimators = 100,  min_samples_leaf = min_samp_leaf, random_state = 42) 
        
    if name == 'AdaBoostRegressor':
        # Assign hyper-parameters
        n_estims,loss = hyperparams
        model.set_params(n_estimators = n_estims, loss = loss ,random_state = 42)

    if name == 'RandomForestRegressor':
        # Assign hyper-parameters
        max_feats,min_samp_leaf = hyperparams
        model.set_params(n_estimators = 100, max_features = max_feats , min_samples_leaf = min_samp_leaf, random_state = 42) 
        # below for PCA --> removing max_features parameter
        # model.set_params(n_estimators = 100,  min_samples_leaf = min_samp_leaf, random_state = 42) 

    if name == 'KernelRidge':
        # Assign hyper-parameters
        alpha_value,gamma_value = hyperparams
        # model.set_params(kernel = 'linear' , alpha = alpha_value, gamma = gamma_value)
        model.set_params(kernel = 'rbf' , alpha = alpha_value, gamma = gamma_value)

        
    if name == 'Ridge' or name == 'Lasso':
        # Assign hyperparameters
        alpha_value = hyperparams
        model.set_params(alpha = alpha_value)
            
    model.fit(X_train,y_train)

    y_pred  = model.predict(X_valid)
    
    
    # Calculate error metric of test and predicted values: rmse
    # err = np.sqrt(mean_squared_error(y_valid, y_pred))
    
    # Error metric = r2
    # err = r2_score(y_valid, y_pred)
    
    # Error metric = MAE
    err = mean_absolute_error(y_valid, y_pred)
    
    
    r_pearson,_=pearsonr(y_valid,y_pred)
    
    if name == 'KernelRidge':
        print('Kernel Ridge. alpha: %7.6f, gamma: %7.4f, RMSE: %7.4f, r: %7.4f' %(alpha_value,gamma_value,err,r_pearson))
        
    if name == 'RandomForestRegressor':
        print('Random Forest Regressor. Max_feats: %7.6f, Min_samps: %7.4f, RMSE: %7.4f, r: %7.4f' %(max_feats, min_samp_leaf ,err,r_pearson))
        
    if name == 'AdaBoostRegressor':
        print('AdaBoost Regressor. n_estims: %7.6f, loss: %s, RMSE: %7.4f, r: %7.4f' %(n_estims, loss ,err,r_pearson))
    
    if name == 'GradientBoostingRegressor':
        print('Gradient Boosting Regressor. Max_feats: %7.6f, Min_samps: %7.4f, RMSE: %7.4f, r: %7.4f' %(max_feats, min_samp_leaf ,err,r_pearson))
        
    if name == 'Ridge' or name ==  'Lasso':
        print('alpha: %7.6f, RMSE: %7.4f, r: %7.4f' %(alpha_value,err,r_pearson))
        
    return err

#%% FUNCTION FOR DEFINING MODELS.
def initial_models():
    ''' Function returning a set of models.
    inputs = None
    outputs = Array of models'''
    models = list()
    # models.append(Ridge())
    models.append(Lasso())
    models.append(KernelRidge())
    models.append(RandomForestRegressor())
    models.append(AdaBoostRegressor())
    models.append(GradientBoostingRegressor())
    # models.append(xgb.XGBRegressor())
    return model

#%% FEATURE SELECTION -> PARAMETER OPTIMIZATION -> STACKING

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
# import xgboost as xgb

from sklearn.preprocessing import StandardScaler


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor


from sklearn.decomposition import PCA


r2_scores = [] 
corr_vec = [] 
k = []

r2_scores_stacked = []
corr_vec_stacked = []

# These store the optimal params for each model on each split.
opti_alphas = []
opti_gammas = []

models = initial_models() 

start = time.time() 
# Manually change kk to change number of features selected. (features selected = 40320/kk)
for kk in [1,2,4,8,16,32,64,128]:
    
    # This will be the level 0 list for stacking.
    optimized_models = [] 

    for model in models:
        name = type(model).__name__
        print(name)
        
    
        for i in range(1,6):
            print('SPLIT: ', i)
            # Setting the matrices according to the split number.
            X_train , y_train , X_test , y_test , X_valid , y_valid = TTV_split(i)
            
            # Scale X_train, X_test and X_valid accordin to X_train
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
            
            # FEATURE SELECTION BY SELECTING A CERTAIN NUMBER OF BEST FEATURES.
            # Create ranking model manually choose Lasso or Random forest regressor ranker
            # ranker = Lasso()
            ranker = RandomForestRegressor(random_state = 42)
            
            selector = SelectFromModel(estimator = ranker , max_features = int(40320/kk),  threshold = -np.inf )
            # selector = SelectFromModel(estimator = ranker , max_features = int(36000/kk),  threshold = -np.inf )

            selector.fit(X_train,y_train)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
            X_valid = selector.transform(X_valid)
            
            # COULD MAKE THE CODE RUN FASTER BY HAVING THE FEATURE SELECTED SPLIT LOOP OVER THE MODELS INSTEAD.
            # WILL CHANGE IT LATER IF NECESSEARY.
            
            
            # PERFORMING PCA CODE
            # # pca = PCA(n_components= kk ,random_state=42) 
            # pca = PCA(random_state=42) 
            
            # pca.fit(X_train)
            # X_train = pca.transform(X_train)
            # X_test = pca.transform(X_test)
            # X_valid = pca.transform(X_valid)
            

            if name == 'Lasso':
                # Call hyperparams_gridsearchfunction on X_train and y_train
                opti_alpha = hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid)
                # Making optimized model
                opti_model = model.set_params(alpha = opti_alpha)
                # Storing optimized model
                optimized_models.append(opti_model)
                
                # Storing optimal hyperparameters for each split.
                opti_alphas.append((kk,name,i, opti_alpha))
                
            if name == 'KernelRidge':
                # Call hyperparams_gridsearchfunction on X_train and y_train
                opti_alpha, opti_gamma = hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid)
                # Making optimized model
                # opti_model = model.set_params(kernel = 'linear' , alpha = opti_alpha , gamma = opti_gamma)
                opti_model = model.set_params(kernel = 'rbf' , alpha = opti_alpha , gamma = opti_gamma)

                # Storing optimized model
                optimized_models.append(opti_model)
                
                # Storing optimal hyperparameters for each split.
                opti_alphas.append((kk,name,i, opti_alpha))
                opti_gammas.append((kk,name,i, opti_gamma))      
                
            if name == 'RandomForestRegressor':
                # Call hyperparams_gridsearchfunction on X_train and y_train
                opti_max_feats, opti_min_samps = hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid)
                # Making optimized model
                opti_model = model.set_params(n_estimators = 100 , max_features = opti_max_feats, min_samples_leaf = opti_min_samps, random_state = 42)
                # Storing optimized model
                optimized_models.append(opti_model)
                
                # Storing optimal hyperparameters for each split.
                opti_alphas.append((kk,name,i, opti_max_feats))
                opti_gammas.append((kk,name,i, opti_min_samps))
                
            if name == 'AdaBoostRegressor':
                # Call hyperparams_gridsearchfunction on X_train and y_train
                opti_estims, opti_loss = hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid)
                # Making optimized model
                opti_model = model.set_params(n_estimators = opti_estims, loss = str(opti_loss) ,random_state = 42)
                # Storing optimized model
                optimized_models.append(opti_model)
                
                # Storing optimal hyperparameters for each split.
                opti_alphas.append((kk,name,i, opti_estims))
                opti_gammas.append((kk,name,i, opti_loss))
            
            if name == 'GradientBoostingRegressor':
                # Call hyperparams_gridsearchfunction on X_train and y_train
                opti_max_feats, opti_min_samps = hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid)
                # Making optimized model
                opti_model = model.set_params(n_estimators = 100 , max_features = opti_max_feats, min_samples_leaf = opti_min_samps, random_state = 42)
                # Storing optimized model
                optimized_models.append(opti_model)
                
                # Storing optimal hyperparameters for each split.
                opti_alphas.append((kk,name,i, opti_max_feats))
                opti_gammas.append((kk,name,i, opti_min_samps))
                
            if name == 'XGBBoostRegressor':
                # Call hyperparams_gridsearchfunction on X_train and y_train
                opti_weight, opti_depth = hyperparams_gridsearch(model,X_train,y_train,X_valid,y_valid)
                # Making optimized model
                opti_model = model.set_params(min_child_weight = opti_weight ,max_depth = opti_depth ,random_state = 42)
                # Storing optimized model
                optimized_models.append(opti_model)
                
                # Storing optimal hyperparameters for each split.
                opti_alphas.append((kk,name,i, opti_estims))
                opti_gammas.append((kk,name,i, opti_loss))
                
            # Fitting the optimized model on the X_train.
            opti_model.fit(X_train,y_train)
            
            # Predict using optimized model on X_test
            r2 = opti_model.score(X_test,y_test)
            # Storing score on this split.
            r2_scores.append((kk,name,i, r2))
            
            corr = np.sqrt(r2)
            corr_vec.append((kk,name,i, corr))
            
            k.append(kk)
            
    # SAVING LIST OF OPTIMIZED MODELS
    filename = str(kk) +'K_rbf_RF_RMSE_optimized_models.sav'
    # path = '/Users/abdulwahab/Y3 Stuff/Project/DATA/Output data dumps/'
    path = '/Users/abdulwahab/Y3 Stuff/Project/DATA/Output opti models err r2/'
    # path = '/Users/abdulwahab/Y3 Stuff/Project/DATA/Splits with 100 features/Output opti models/'

    save_name = path +filename
    pickle.dump(optimized_models, open(save_name, 'wb'))
        
    # STACKING TIME. 
    for jj in range(1,6):
        print('SPLIT: ', jj)
        
        # Setting the matrices according to the split number.
        X_train , y_train , X_test , y_test , X_valid , y_valid = TTV_split(jj)
        
        # Scale X_train and X_valid
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
        
        # # create ranking model
        # ranker = Lasso()
        ranker = RandomForestRegressor(random_state = 42)
        
        selector = SelectFromModel(estimator = ranker , max_features = int(40320/kk),  threshold = -np.inf )
        # selector = SelectFromModel(estimator = ranker , max_features = int(36000/kk),  threshold = -np.inf )

        selector.fit(X_train,y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        X_valid = selector.transform(X_valid)
        
        # PERFORMING PCA
        # pca = PCA(n_components= kk ,random_state=42) 
        # pca = PCA(random_state=42) 
        
        # pca.fit(X_train)
        # X_train = pca.transform(X_train)
        # X_test = pca.transform(X_test)
        # X_valid = pca.transform(X_valid)

        level0 = list()

        count = 0 
        
        for ii in np.arange(0+count,25+count,5):
            # print(i)
            # name = type(loaded_models[i]).__name__
            name = type(optimized_models[ii]).__name__
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

# ranker RF
# TIME TAKEN:  42016.37368893623

# lasso TIME TAKEN:  1342.2931308746338

# RF TIME TAKEN:  21096.487411022186
# TIME TAKEN:  16239.893362045288

# RF MAE k 1-128 --> TIME TAKEN:  88076.77021408081

# 36000 SS RF TIME TAKEN:  65060.26137781143

# NOS RF MAE TIME TAKEN:  83023.88084673882

# TIME TAKEN:  48859.344517707825

# TIME TAKEN:  84630.95842790604 --> rbf RF MAE

# TIME TAKEN:  86572.63571023941 --> rbf RF RMSE


# CONVERTING TO DATAFRAMES TO SAVE RESULTS LATER
df_p1 = pd.DataFrame(opti_alphas)
df_p2 = pd.DataFrame(opti_gammas)

df_r2 = pd.DataFrame(r2_scores)
df_corr = pd.DataFrame(corr_vec)

df_str = pd.DataFrame(r2_scores_stacked)
df_stcorr = pd.DataFrame(corr_vec_stacked)

#%% Stacking from saved optimized models file.

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

for kk in [1,2,4,8,16,32,64,128]:
    
    filename = str(kk) + 'K_rbf_RF_MAE_optimized_models.sav'
    path = '/Users/abdulwahab/Y3 Stuff/Project/DATA/Output opti models err r2/'
    
    optimized_models = pickle.load(open(path+filename, 'rb'))
    

    for jj in range(1,6):
        print('SPLIT: ', jj)
        
        # Setting the matrices according to the split number.
        X_train , y_train , X_test , y_test , X_valid , y_valid = TTV_split(jj)
        
        # # Scale X_train and X_valid
        # scaler = StandardScaler().fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)
        # X_valid = scaler.transform(X_valid)
        
        # SCALING EACH ONE SEPARATELY.
        train_scaler = StandardScaler().fit(X_train)
        X_train = train_scaler.transform(X_train)
        
        test_scaler = StandardScaler().fit(X_test)
        X_test = test_scaler.transform(X_test)
        
        valid_scaler = StandardScaler().fit(X_valid)
        X_valid = valid_scaler.transform(X_valid)
        
        # create ranking model
        # ranker = Lasso()
        ranker = RandomForestRegressor(random_state = 42)
        # Maybe try RF as ranker.
        # Maybe try PCA followed by further feature selection.
        # Plot eigenvalue spread on plot, and observe the variance threshold.
        # Also try 32,64,128 for k.
        # PCA should give a better idea of how many features are meaningful.
        # save PCA eigenvectors --> take the mean and "step in the direction of the eigenvector"
        # prior to PCA perform Kbest to try removing noise. maybe split in half initially.
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
            
            if name == 'KernelRidge' :
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
# TIME TAKEN:  3274.9606132507324
# TIME TAKEN:  18620.40162706375

# TIME TAKEN:  9951.666650056839

#%%

# df_r2 = pd.DataFrame(r2_scores)
# df_corr = pd.DataFrame(corr_vec)

df_str = pd.DataFrame(r2_scores_stacked)
df_stcorr = pd.DataFrame(corr_vec_stacked)
