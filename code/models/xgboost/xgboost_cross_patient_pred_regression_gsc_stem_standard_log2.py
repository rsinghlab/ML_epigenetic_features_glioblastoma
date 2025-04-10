'''
This is a script to implement a 
XGBoost algorithm model. 

Goal: Predict gene expression value (regression) 
from epigenetic signal inputs.

It is configured for cross-patient prediction with
logging, results output and visualizations for 
downstream analysis.

The script input is two patient datafiles 
and an index file (all in numpy format). 

The model is trained and validated (if applicable) on 
the first position datafile. The model is then 
tested on the second position file.

This script is designed to accept 'raw' input values. It 
applies the log2 scaling to the target variable before 
the dataset split process and applies standardization 
to the train, validation and test datasets seperately 
after the split.  
'''

import sys
import numpy as np
import os
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn import preprocessing
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from itertools import product
import random
import datetime
import math
import pandas as pd

def get_data_patient_1(file_path, indices, gene_dict, num_genes, 
                       preprocess, validation):
    '''
    return: X_train, Y_train, X_val, Y_val; where X refers 
    to inputs and Y refers to labels.
    return: gene_dict (gene name dictionary)
    return: num_genes (the number of genes in the test set)
    '''
    
    if preprocess:
       
        print("Loading patient 1 dataset...")
        # Col 1 = gene names, 2 = bin number, 3-6 = features, 7 = labels
        combined_diff = np.load(file_path, allow_pickle=True)
        
        # Get all the unique gene names
        gene_names = np.unique(combined_diff[:, 0])
        num_genes = len(gene_names)
        
        # Create a dictionary to map each gene name to a 
        # unique index (number like 0, 1, 2,...,#genes-1)
        gene_dict = dict(zip(gene_names, range(num_genes)))

        # Get the number of features (last column is labels - RNAseq)
        num_features = 4
        num_bins = 50
        
        # Inputs data shape
        # X = np.zeros((num_genes, num_features, num_bins))
        X = np.zeros((num_genes, num_bins, num_features))

        # Labels data shape
        Y = np.zeros((num_genes, 1))

        for name in tqdm(gene_names):
            # Each subset is of shape 100 x 6 (number of 
            # 100bp bins x number of columns)
            subset = combined_diff[np.where(combined_diff[:, 0] == name)]

            # Create matrix of data
            gene_ind = gene_dict[name]
            data = subset[:, 2:]
            
            # data_inputs = np.transpose(data[:, :-1])
            data_inputs = data[:, :-1]
                                    
            # Add to array at the unique id position
            X[gene_ind] = data_inputs

            # Set corresponding value to be first bin's RNAseq 
            # value (since all 50 bins have the same value when 
            # using the featureCounts utility and process).
            Y[gene_ind] = data[0, -1]

            #NOTE: Evaluating different methods of determining the RNAseq value.
            # 1. Calculate the mean of the values over all 50 bins.
            #Y[gene_ind] = np.mean(data[:, -1])

            # 2. Calculate the median of the values over all 50 bins.
            #Y[gene_ind] = np.median(data[:, -1])

            # 3. Calculate the sum of the values over all 50 bins.
            #Y[gene_ind] = np.sum(data[:, -1])



        # Log2 scale the Y response variable.
        Y = np.log2(Y + 1)
        print('max training Y value:', np.max(Y))
        print('min training Y value:', np.min(Y))
        print('range training Y values:', np.ptp(Y))

        
        # Shuffle the data
        # ind = np.arange(0, num_genes)
        # np.random.shuffle(ind)
        ind = np.load(indices, allow_pickle = True)
        print('First X dataset shape')
        print(X.shape)
        # Collect the indices that need to be deleted from the array
        # because the number of genes is lower than the 20,015 due to 
        # experiments keeping only the expressed genes in combined_diff
        # or different numbers of genes in various test datasets. 
        print('Patient 1 dataset shape : ')
        print(combined_diff.shape)
        indexes = np.where(ind > X.shape[0] - 1)
        patient1_ind = np.delete(ind, indexes)
        print('Patient 1 indeces shape : ')
        print(patient1_ind.shape)


        
        if validation == True:
            # HYPERPARAMETER TUNING SPLITS
            # Create train (70%), validation (30%).
            #train_ind = ind[0: int(0.7*num_genes)]
            #val_ind = ind[int(0.7*num_genes):]
            
            train_ind = patient1_ind[0: int(0.7*num_genes)]
            val_ind = patient1_ind[int(0.7*num_genes):]
            
            X_train = X[train_ind]
            X_val = X[val_ind]
        
            Y_train = Y[train_ind]
            Y_val = Y[val_ind]

            # List of all datasets after split operation.
            # datasets = [X_train, X_val, Y_train, Y_val]
            # Standardization ONLY on input variables.
            datasets = [X_train, X_val]

        
        else:
            # TESTING SPLITS
            # The training set will have 100% of the 
            # patient 1 data to train the model.
            train_ind = patient1_ind
            X_train = X[train_ind]
            Y_train = Y[train_ind]
            datasets = [X_train]

        
        


        # Perform calculation on each column of the seperate train, validation and test sets. 
        for dataset in datasets:
            for i in range(dataset.shape[2]): # Standardize the column values.
                #dataset[:, :, i] = (dataset[:, :, i] - np.mean(dataset[:, :, i])) / np.std(dataset[:, :, i], ddof = 1)
                
            # The lines below are for PERTURBATION ANALYSIS ONLY. Uncomment the line that applies to the feature
            # with all zero values. The for loop line above needs to be commented when using one of the 
            # lines below.
            
            #for i in [1,2,3]: ### NO standardization on column 0 - H3K27ac for perturbation analysis ONLY.
            #for i in [0,2,3]: ### NO standardization on column 1 - CTCF for perturbation analysis ONLY.
            #for i in [0,1,3]: ### NO standardization on column 2 - ATAC for perturbation analysis ONLY.
            #for i in [0,1,2]: ### NO standardization on column 3 - RNA Pol II for perturbation analysis ONLY.

            #for i in [0,1]: ### NO standardization on column 3 - ATAC and RNA Pol II for neural crest cell testing ONLY.
            #for i in [0,1,2]: ### NO standardization on column 4 - RNA Pol II for neural progenitor cell testing ONLY.
            #for i in [0,1]: ### NO standardization on column 3 and 4 - ATAC RNA Pol II for redued feature training 11-19-23 of neural crest cells ONLY.
            #for i in [0,1,2]: ### NO standardization on column 4 - ATAC RNA Pol II for redued feature training 11-20-23 of neural progenitor cells ONLY.
                # Standardize the column values.
                dataset[:, :, i] = (dataset[:, :, i] - np.mean(dataset[:, :, i])) / np.std(dataset[:, :, i], ddof = 1)

                
        np.save("X_cross_patient_regression_patient_1_stem_standard_log2_train", 
                X_train, 
                allow_pickle = True)
        np.save("Y_cross_patient_regression_patient_1_stem_standard_log2_train", 
                Y_train, 
                allow_pickle = True)

        if validation == True:
            np.save("X_cross_patient_regression_patient_1_stem_standard_log2_val",
                    X_val,
                    allow_pickle = True)

            np.save("Y_cross_patient_regression_patient_1_stem_standard_log2_val", 
                    Y_val,
                    allow_pickle = True)
        
    
    else:
        X_train = np.load("X_cross_patient_regression_patient_1_stem_standard_log2_train.npy", 
                          allow_pickle = True)
        Y_train = np.load("Y_cross_patient_regression_patient_1_stem_standard_log2_train.npy", 
                          allow_pickle = True)
        
        if validation == True:
            X_val = np.load("X_cross_patient_regression_patient_1_stem_standard_log2_val.npy", 
                            allow_pickle = True)
            Y_val = np.load("Y_cross_patient_regression_patient_1_stem_standard_log2_val.npy",
                            allow_pickle = True)
        
    
        gene_dict = gene_dict
        num_genes = num_genes

    if validation == True:
        return X_train, X_val, Y_train, Y_val, gene_dict, num_genes
    else:
        return X_train, Y_train, gene_dict, num_genes

def get_data_patient_2(file_path, indices, gene_dict, 
                       num_genes, preprocess):
    '''
    return: X_test, Y_test; where X refers to inputs and Y refers to labels.
    return: gene_dict (gene name dictionary)
    return: num_genes (the number of genes in the test set)
    return: patient2_ind (the indices for the patient 2 dataset. This
    may be a subset of the indices shuffle index if the number of genes
    in the test set is lower than the 20,015 in the training set) 
    '''

    if preprocess:
        print("Loading patient 2 dataset...")
        # Col 1 = gene names, 2 = bin number, 3-6 = features, 7 = labels
        combined_diff = np.load(file_path, allow_pickle = True)
        
        # Get all the unique gene names
        gene_names = np.unique(combined_diff[:, 0])
        num_genes = len(gene_names)

        # Create a dictionary to map each gene name to a unique 
        # index (number like 0, 1, 2,...,#genes-1)
        gene_dict = dict(zip(gene_names, range(num_genes)))

        # Get the number of features (last column is labels - RNAseq)
        num_features = 4
        num_bins = 50

        # Inputs data shape
        # X = np.zeros((num_genes, num_features, num_bins))
        X = np.zeros((num_genes, num_bins, num_features))

        # Labels data shape
        Y = np.zeros((num_genes, 1))

        for name in tqdm(gene_names):

            # Each subset is of shape 100 x 6 (number of 
            # 100bp bins x number of columns)
            subset = combined_diff[np.where(combined_diff[:, 0] == name)]

            # Create matrix of data. 
            gene_ind = gene_dict[name]
            data = subset[:, 2:]

            data_inputs = data[:, :-1]

            # Add to array at the unique id position
            X[gene_ind] = data_inputs
   
            # Set corresponding value to be first bin's 
            # RNAseq value (since all 50 bins
            # have the same value when using the 
            # featureCounts utility and process).
            Y[gene_ind] = data[0, -1]

            #NOTE: Evaluating different methods of determining the RNAseq value.
            # 1. Calculate the mean of the values over all 50 bins.
            #Y[gene_ind] = np.mean(data[:, -1])

            # 2. Calculate the median of the values over all 50 bins.
            #Y[gene_ind] = np.median(data[:, -1])

            # 3. Calculate the sum of the values over all 50 bins.
            #Y[gene_ind] = np.sum(data[:, -1])


        # Log2 scale the Y response variable
        Y = np.log2(Y + 1)
        print('max test Y value:', np.max(Y))
        print('min test Y value:', np.min(Y))
        print('range test Y values:', np.ptp(Y))
        
        # Shuffle the data
        #ind = np.arange(0, num_genes)
        # np.random.shuffle(ind)
        # Original shuffle index file loaded
        ind = np.load(indices, allow_pickle = True)
        print('Second X dataset shape')
        print(X.shape)
        # Collect the indices that need to be deleted from the array
        # because the number of genes is lower than the 20,015 due to 
        # experiments keeping only the expressed genes in combined_diff
        # or different numbers of genes in various test datasets. 
        print('Patient 2 dataset shape : ') 
        print(combined_diff.shape)
        indexes = np.where(ind > X.shape[0] - 1)
        patient2_ind = np.delete(ind, indexes)
        print('Patient 2 indeces shape : ')
        print(patient2_ind.shape)

        
        # 10/28/24 Custom shuffle index for Omnibus datasets loaded
        #ind = np.load('../../../data/ind_shuffle_for_Omnibus_datasets.npy', allow_pickle=True)
        # 11/1/24 Custom shuffle index for Omnibus v3 datasets loaded
        #ind = np.load('../../../data/ind_shuffle_for_Omnibus_v3_datasets.npy', allow_pickle=True)        

        # Splits for this patient data can be adjusted here.
        #train_ind = ind[0: int(0.7*num_genes)]
        #val_ind = ind[int(0.7*num_genes):int(0.85*num_genes)]
        #test_ind = ind[int(0.85*num_genes):]


        # NOTE: For now use entire dataset for test set.
        #test_ind = ind
        test_ind = patient2_ind
        
        # Use all of the dataset for test.
        X_test = X[test_ind]

        # Use all of the dataset for test.
        Y_test = Y[test_ind]

        # List of all datasets after split operation.
        # Standardization ONLY on input variables.
        datasets = [X_test]

        # Perform calculation on each column of the seperate train, validation and test sets.
        for dataset in datasets:
            for i in range(dataset.shape[2]): ### Standardization on all columns.
                
            # The lines below are for PERTURBATION ANALYSIS ONLY. Uncomment the line that applies to the feature
            # with all zero values. The for loop line above needs to be commented when using one of the 
            # lines below.
            
            #for i in [1,2,3]: ### NO standardization on column 0 - H3K27ac for perturbation analysis ONLY.
            #for i in [0,2,3]: ### NO standardization on column 1 - CTCF for perturbation analysis ONLY.
            #for i in [0,1,3]: ### NO standardization on column 2 - ATAC for perturbation analysis ONLY.
            #for i in [0,1,2]: ### NO standardization on column 3 - RNA Pol II for perturbation analysis ONLY.

            #for i in [0,1]: ### NO standardization on columns 2 and 3 - ATAC and RNA Pol II for neural crest cell testing ONLY.
            #for i in [0,1,2]: ### NO standardization on column 3 - RNA Pol II for neural progenitor cell testing ONLY.
            #for i in [0,1]: ### NO standardization on column 3 and 4 - ATAC RNA Pol II for redued feature testing of GSC1 and GSC2 ONLY.

            #for i in [0,1,2]: ### NO standardization on column 3 - RNA Pol II for neural progenitor cell testing ONLY.

            #10/28/24
            ####for i in [0]: ### NO standardization on columns 1,2,and 3 
            # - CTCF, ATAC and RNA Pol II for Omnibus cell testing ONLY.
                # Standardize the column values.
                dataset[:, :, i] = (dataset[:, :, i] - np.mean(dataset[:, :, i])) / np.std(dataset[:, :, i], ddof = 1)

        #dataset[:, :, 2] = 0.000000 # Experiment 11/07/23 set values for ATAC (DNase-seq) in progenitor cell dataset to 0.000000 to match crest cell dataset feature values 
        #dataset[:, :, 2:5] = 0.000000 # Experiment 11/14/23 set values for ATAC and RNApol2 to 0.000000 to match crest and progenitor cell datasets feature values
        # Experiment 11/19/23 set values for ATAC and RNApol2 to 0.000000 to match progenitor cell datasets feature values

        np.save("X_cross_patient_regression_patient_2_stem_standard_log2_test", 
                X_test, 
                allow_pickle = True)
        np.save("Y_cross_patient_regression_patient_2_stem_standard_log2_test", 
                Y_test, 
                allow_pickle=True)

    else:
        X_test = np.load("X_cross_patient_regression_patient_2_stem_standard_log2_test.npy", 
                         allow_pickle = True)
        Y_test = np.load("Y_cross_patient_regression_patient_2_stem_standard_log2_test.npy", 
                         allow_pickle = True)

        gene_dict = gene_dict
        num_genes = num_genes

    return X_test, Y_test, gene_dict, num_genes, patient2_ind


def reset_random_seeds(seed):
    '''
    Takes a given number and assigns it
    as a random seed to various generators and the
    os environment.
    '''

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)

    return None

def train_model(X_train, Y_train, validation, 
                learning_rates, n_estimators, max_depths, 
                min_child_weight, colsample_bytree, 
                subsample, gamma, count, random_state, 
                X_val, Y_val):
    """
    Implements and trains a XGBoost model.
    param X_train: the training inputs
    param Y_train: the training labels
    param X_val: the validation inputs
    param Y_val: the validation labels
    return: a trained model and training metrics
    """

    #random_state = count
    random_state = random_state
    
    # Set random seed
    reset_random_seeds(random_state)
    
    #4/4/25 Use only H3K27Ac for training and testing
    #X_train = X_train[:,:,0]
    #print('training and testing with 1 feature')
    #print('Shape of training X (features) data: ', X_train.shape)
    
        
    # Reshape data into 2 dimensions.
    reshaped_X_train = X_train.reshape((X_train.shape[0], -1), 
                                       order = 'F')
    reshaped_Y_train = np.squeeze(Y_train)
    
    if validation == True:
        #11/5/25 Use only H3K27ac for training and testing
        #X_val = X_val[:,:,0]
        #print('training, validating and testing with 1 feature')
        #print('Shape of validation X (features) data: ', X_val.shape)
        reshaped_X_val = X_val.reshape((X_val.shape[0], -1), 
                                   order = 'F')
        reshaped_Y_val = np.squeeze(Y_val)
    
    # Define model.
    xgb_reg = xgb.XGBRegressor(objective = 'reg:squarederror', 
                               tree_method = 'gpu_hist', 
                               learning_rate = learning_rates, 
                               n_estimators = n_estimators, 
                               max_depth = max_depths, 
                               min_child_weight = min_child_weight, 
                               colsample_bytree = colsample_bytree, 
                               subsample = subsample, 
                               gamma = gamma, seed = random_state)
    # Fit model to data. 
    xgb_reg.fit(reshaped_X_train, reshaped_Y_train)
    
    # Evaluate model with validation data if 
    # stipulated and return validation metrics.
    if validation == True:
        Y_pred = xgb_reg.predict(reshaped_X_val)
        PCC = pearsonr(reshaped_Y_val, Y_pred)[0]
        SCC = spearmanr(reshaped_Y_val, Y_pred)[0]
        R2 = r2_score(reshaped_Y_val, Y_pred)
    
        return xgb_reg, PCC, SCC, R2
    
    else:       
        return xgb_reg
    


def test_model(model, X_test, Y_test, learning_rates, 
               n_estimators, max_depths, 
               min_child_weight, colsample_bytree, 
               subsample, gamma, count):
    '''
    Evaluates the trained XGBoost model.
    param X_test: the testing inputs
    param Y_test: the testing labels
    return: testing metric results
    '''
    
    #4/4/25 Use only H3K27ac for training and testing
    #X_test = X_test[:,:,0]
    #print('testing with 1 feature')
    #print('Shape of testing X (features) data: ', X_test.shape)
        
    # Reshape data into 2 dimensions.
    reshaped_X_test = X_test.reshape((X_test.shape[0], -1), 
                                     order = 'F')
    reshaped_Y_test = np.squeeze(Y_test)
    
    # Evaluate the model using the test dataset.
    Y_pred = model.predict(reshaped_X_test)
    PCC = pearsonr(reshaped_Y_test, Y_pred)[0]
    SCC = spearmanr(reshaped_Y_test, Y_pred)[0]
    R2 = r2_score(reshaped_Y_test, Y_pred)
    
    return PCC, SCC, R2

def get_feature_importances(model):
    importances = model.feature_importances_
    h3k27ac_importances = importances[:50]
    ctcf_importances = importances[50:100]
    atac_importances = importances[100:150]
    rnapii_importances = importances[150:200]
    
    print('Length of H3K27Ac importance array : ')
    print(len(h3k27ac_importances))
    h3k27ac_mean_importances = np.mean(h3k27ac_importances)
    ctcf_mean_importances = np.mean(ctcf_importances)
    atac_mean_importances = np.mean(atac_importances)
    rnapii_mean_importances = np.mean(rnapii_importances)

    h3k27ac_sum_importances = np.sum(h3k27ac_importances)
    ctcf_sum_importances = np.sum(ctcf_importances)
    atac_sum_importances = np.sum(atac_importances)
    rnapii_sum_importances = np.sum(rnapii_importances)

    # Create csv file to hold each epigenetic feature's mean importance.
    with open(save_directory + 
              '/xgboost_cross_patient_regression_gsc_stem_standandard_mean_feature_importances.csv', 'w') \
              as log:
            log.write(f'H3K27ac mean importance,{h3k27ac_mean_importances}'),
            log.write('\n' f'CTCF mean importance,{ctcf_mean_importances}'),
            log.write('\n' f'ATAC mean importance,{atac_mean_importances}'),           
            log.write('\n' f'RNAPII mean importance,{rnapii_mean_importances}')

    # Create csv file to hold each epigenetic feature's sum importance.
    with open(save_directory + 
              '/xgboost_cross_patient_regression_gsc_stem_standandard_sum_feature_importances.csv', 'w') \
            as log:
            log.write(f'H3K27ac sum importance,{h3k27ac_sum_importances}'),
            log.write('\n' f'CTCF sum importance,{ctcf_sum_importances}'),
            log.write('\n' f'ATAC sum importance,{atac_sum_importances}'),            
            log.write('\n' f'RNAPII sum importance,{rnapii_sum_importances}')
            
    return h3k27ac_mean_importances, atac_mean_importances, ctcf_mean_importances, \
           rnapii_mean_importances, h3k27ac_sum_importances, atac_sum_importances, \
           ctcf_sum_importances, rnapii_sum_importances


def visualize_feature_importances_sums(h3k27ac_mean_importances, atac_mean_importances,
                                  ctcf_mean_importances, rnapii_mean_importances, 
                                  h3k27ac_sum_importances, atac_sum_importances,
                                  ctcf_sum_importances, rnapii_sum_importances):
    
    importance_sums = pd.DataFrame({'metric' : ['h3k27ac_sum_importances', 
                                                'ctcf_sum_importances', 
                                                'atac_sum_importances', 
                                                'rnapii_sum_importances'],
                                   'value' : [h3k27ac_sum_importances, 
                                              ctcf_sum_importances,
                                              atac_sum_importances, 
                                              rnapii_sum_importances]})

    
    sn.set(style ='white', font_scale = 1.5)
    fig, ax = plt.subplots(figsize=(9, 2.5))
    
    feature_importance_plot = sn.barplot(data = importance_sums, 
                          palette = ['red', 'red', 'red', 'red'], 
                          x = 'value', 
                          y = 'metric')

    feature_importance_plot.set_yticklabels(feature_importance_plot.get_yticklabels())
    plt.tick_params(axis = 'both', 
                which = 'major', 
                labelbottom = True, 
                bottom = True, 
                top = False, 
                labeltop = False, 
                left = True)
    sn.despine()
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(['0.0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1.0'])

    plt.xlabel('Sums of importance values over bins')
    ax.set_yticklabels(['H3K27Ac importance sum', 
                        'CTCF importance sum', 
                        'ATAC importance sum', 
                        'RNAPII importance sum'])
    plt.ylabel('')
    plt.xlim(0.0, 1.0)
    for i in [0, 1, 2, 3]:
        p = feature_importance_plot.patches[i]
        feature_importance_plot.annotate("%.6f" % p.get_width(), 
                          xy = (p.get_width(), 
                                p.get_y() + 
                                p.get_height() / 2), 
                          color = 'black', 
                          xytext = (30, 0), 
                          textcoords = 'offset points', 
                          ha = 'left', va = 'center')
        
    plt.savefig(save_directory + 
                '/xgboost_sums_of_feature_importances.png', 
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')
    plt.savefig(save_directory + 
                '/xgboost_sums_of_feature_importances.eps', 
                format = 'eps',
                dpi = 600,
                bbox_inches = 'tight')
    plt.savefig(save_directory + 
                '/xgboost_sums_of_feature_importances.tiff', 
                format = 'tiff',
                dpi = 600,
                bbox_inches = 'tight')
    plt.savefig(save_directory + 
                '/xgboost_sums_of_feature_importances.svg', 
                format = 'svg',
                dpi = 600,
                bbox_inches = 'tight')
    
    return None

def visualize_feature_importances_means(h3k27ac_mean_importances, atac_mean_importances,
                                  ctcf_mean_importances, rnapii_mean_importances, 
                                  h3k27ac_sum_importances, atac_sum_importances,
                                  ctcf_sum_importances, rnapii_sum_importances):

    importance_means = pd.DataFrame({'metric' : ['h3k27ac_mean_importances',
                                                 'ctcf_mean_importances', 
                                                 'atac_mean_importances',
                                                 'rnapii_mean_importances'],
                                   'value' : [h3k27ac_mean_importances,
                                              ctcf_mean_importances,
                                              atac_mean_importances,
                                              rnapii_mean_importances]})
    
    sn.set(style = 'white', font_scale = 1.5)
    fig, ax = plt.subplots(figsize = (9, 2.5))
    importance_plot = sn.barplot(data = importance_means, 
                       palette = ['springgreen', 'springgreen', 'springgreen', 'springgreen'], 
                       x = 'value', 
                       y = 'metric')
    
    importance_plot.set_yticklabels(importance_plot.get_yticklabels())
    plt.tick_params(axis = 'both', 
                which = 'major', 
                labelbottom = True, 
                bottom = True, 
                top = False, 
                labeltop = False, 
                left = True)
    plt.xlabel('mean importance values over bins')
    #ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xticklabels(['0.0', '', '0.005', '', '0.01', '', '0.015', '', '0.02'])
    ax.set_yticklabels(['H3K27Ac importance mean', 
                    'CTCF importance mean', 
                    'ATAC importance mean', 
                    'RNAPII importance mean'])
    sn.despine()
    plt.ylabel('')
    plt.xlim(0.000, 0.02)
    for i in [0, 1, 2, 3]:
        p = importance_plot.patches[i]
        importance_plot.annotate("%.6f" % p.get_width(), 
                      xy=(p.get_width(), 
                          p.get_y() + 
                          p.get_height() / 2), 
                      color='black', 
                      xytext = (30, 0), 
                      textcoords = 'offset points', 
                      ha= 'left', 
                      va = 'center')

    plt.savefig(save_directory + 
                '/xgboost_means_of_feature_importances.png',
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')
    
    plt.savefig(save_directory + 
                '/xgboost_means_of_feature_importances.eps',
                format = 'eps',
                dpi = 600,
                bbox_inches = 'tight')
        
    plt.savefig(save_directory + 
                '/xgboost_means_of_feature_importances.tiff',
                format = 'tiff',
                dpi = 600,
                bbox_inches = 'tight')

    plt.savefig(save_directory + 
                '/xgboost_means_of_feature_importances.svg',
                format = 'svg',
                dpi = 600,
                bbox_inches = 'tight')    
    
    
    return None


def get_gene_names(gene_dict, indices, test_data_shape, num_genes, shuffle_index):
    '''
    Using the input dictionary of gene names and their unique identifier
    this function extracts the genes and their index position in X_test data
    for future information for visualization. This position is after the
    shuffle operation and cross_validation split.

    returns: Returns a list of the gene names
    '''
    # Load indices file used for shuffle operation.
    #shuffle_index = np.load(indices, allow_pickle=True)
    # 10/28/24 Load custom shuffle index for Omnibus datasets
    #shuffle_index = np.load('../../../data/ind_shuffle_for_Omnibus_datasets.npy', allow_pickle=True)
    # 11/1/24 Custom shuffle index for Omnibus v3 datasets loaded
    #shuffle_index = np.load('../../../data/ind_shuffle_for_Omnibus_v3_datasets.npy', allow_pickle=True)    
    #shuffle_index = np.arange(0, 20015) 

    ### Discontinue using the loaded index file because it 
    ### is setup for 20,0015 genes and does not accomidate
    ### the smaller set of genes in avaliable in the later
    ### datasets used for cross-patient testing.

    # Invert order of keys and values for gene dictionary.
    inverted_gene_dict = {v:k for k, v in gene_dict.items()}

    # Using entire second dataset for test set
    test_ind = shuffle_index

    gene_names_in_test_set = [inverted_gene_dict[i] for i in test_ind]
    
    return gene_names_in_test_set

def make_prediction(model, input_data):
    '''
    param model: a trained model
    param input_data: model inputs
    return: the model's predictions for the provided input data
    '''
    #4/4/25 Use only H3K27Ac for training and testing
    #input_data = input_data[:,:,0]
    #print('training and testing with 1 feature')
    #print('prediction test dataset shape :', input_data.shape)

    reshaped_X_test = input_data.reshape((input_data.shape[0], -1), order = 'F')
    return np.asarray(model.predict(reshaped_X_test), dtype='float')

def calculate_se_for_predictions(y_true, y_pred):
    '''
    Calculates the Squared Error on the test set predictions 
    after the model is trained.
    param y_true: the true (observed) RNAseq values.
    param y_pred: the model's predicted values.

    returns: Array of SE values the same length as Y_test.
    '''

    prediction_se = np.round_(np.square(np.squeeze(y_true) - y_pred),
                              decimals = 9)

    return prediction_se

def prediction_csv(se_value, y_true, y_pred, gene_names):
    '''
    Writes each gene name with it's true and predicted RNAseq values along
    with the calculation of the Squared Error into a csv file saved to the 
    model's images directory.
    
    param se_value: Array of the calculated Squared Error values per gene.
    param y_true: the true (observed) RNAseq values.
    param y_pred: the model's predicted values.
    param gene_names: list of each gene name in the test set.
    
    return: Nothing
    '''
   # Create csv file to hold prediction information.
    with open(save_directory + 
              '/xgboost_cross_patient_regression_gsc_stem_standard_test_predictions.csv', 'w') \
    as log:
            log.write(f'gene name, true RNAseq value, predicted RNAseq value, prediction Squared Error (SE)')

    for i in tqdm(range(len(gene_names))):
        #index = gene_indices[i]
        gn = gene_names[i]
        se = se_value[i]
        tv = y_true[i]
        pv = y_pred[i]
        with open(save_directory + 
                  '/xgboost_cross_patient_regression_gsc_stem_standard_test_predictions.csv', 'a') \
        as log:
            #log.write('\n' f'{gn}, {tv[0]}, {pv[0]}, {se[0]}')
            log.write('\n' f'{gn}, {tv[0]}, {pv}, {se}')

    return None

def visualize_training_distributions(y_train):
    '''
    Creates multiple visualizations for the 
    training set. Visualizations are saved to the 
    script's output folder.
    param y_train: the true (observed) RNAseq values 
                   for the test set.

    return: Nothing
    '''

    # Build dataframe for visualization.
    #train_and_val_rnaseq = pd.DataFrame({'training set' : y_train.tolist(),
                                        #'validation set' : y_val.tolist()})
    
    plt.close()
    sn.set_theme(style = 'whitegrid')
    fig, ax = plt.subplots(figsize = (8, 5))
    ax = sn.histplot(y_train, 
                     legend = False, 
                     palette = ['red'], 
                     bins = 50)
    ax.set(title = f'Training set genes\' RNAseq value counts.')
    plt.ylabel('count')
    plt.xlabel('RNAseq value after log(2) transformation.')
    plt.savefig(save_directory + '/Training_set_genes_RNAseq_value_counts-_histogram_plot.png',
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')
    
    plt.savefig(save_directory + '/Training_set_genes_RNAseq_value_counts-_histogram_plot.eps',
                format = 'eps',
                dpi = 600,
                bbox_inches = 'tight')
    
    plt.savefig(save_directory + '/Training_set_genes_RNAseq_value_counts-_histogram_plot.tiff',
                format = 'tiff',
                dpi = 600,
                bbox_inches = 'tight')
    
    plt.savefig(save_directory + '/Training_set_genes_RNAseq_value_counts-_histogram_plot.svg',
                format = 'svg',
                dpi = 600,
                bbox_inches = 'tight')    


    training_RNAseq_dataframe = pd.Series(np.squeeze(y_train))
    
    # Define gene expression catagories for analysis.

    training_all_zero_true_expression = training_RNAseq_dataframe[training_RNAseq_dataframe  == 0]
    training_true_expression_between_0_and_5 = training_RNAseq_dataframe[(training_RNAseq_dataframe > 0) & (training_RNAseq_dataframe < 5)]
    training_true_expression_between_5_and_10 = training_RNAseq_dataframe[(training_RNAseq_dataframe >= 5) & (training_RNAseq_dataframe < 10)]
    training_true_expression_between_10_and_15 = training_RNAseq_dataframe[(training_RNAseq_dataframe >= 10) & (training_RNAseq_dataframe <= 15)]
    

    training_expression_counts_dataframe = pd.DataFrame({"expression catagory after log(2) transformation" : 
                                                         ["all zero", 
                                                          "between 0_and 5",
                                                          "between 5 and 10", 
                                                          "between 10 and 15" ],
                                                         "count": [len(training_all_zero_true_expression),
                                                                   len(training_true_expression_between_0_and_5), 
                                                                   len(training_true_expression_between_5_and_10),
                                                                   len(training_true_expression_between_10_and_15)]})
    

    dataframes = [training_expression_counts_dataframe]
    dataset_names = ['Training Set']
    # Visualize number of genes in each catagory.
    for l in range(len(dataframes)):
        plt.close()
        sn.set_theme(style = 'whitegrid')
        sn.set(font_scale = 1)
        fig, ax = plt.subplots(figsize = (8, 5))
        ax = sn.barplot(data = dataframes[l], 
                        x = "expression catagory after log(2) transformation", 
                        y = "count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation = "45")
        ax.set(title = f'{dataset_names[l]} - XGBoost Cross Patient Regression Expression Catagory Counts')
        for i in ax.containers:
            ax.bar_label(i,)
        plt.savefig(save_directory + 
                    '/xgboost_cross_patient_regression_' + dataset_names[l] + 
                    '_Expression_Catagory_Counts.png',
                    format = 'png',
                    dpi = 600,
                    bbox_inches = 'tight')

    return None

def visualize_validation_distributions(y_val):
    '''
    Creates multiple visualizations for the validation set
    if available. Visualizations are saved to the script's
    output image folder.
    param y_val: the true (observed) RNAseq values for the validation set.

    return: Nothing
    '''
    
    plt.close()
    sn.set_theme(style = 'whitegrid')
    sn.set(font_scale = 1)
    fig, ax = plt.subplots(figsize = (8, 5))
    ax = sn.histplot(y_val, legend = False, palette = ['yellow'], bins = 50)
    ax.set(title = f'Validation set genes\' RNAseq value counts.')
    plt.ylabel('count')
    plt.xlabel('RNAseq value after log(2) transformation.') 
    plt.savefig(save_directory + '/Validation_set_genes_RNAseq_value_counts-_histogram_plot.png',
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')

    plt.savefig(save_directory + '/Validation_set_genes_RNAseq_value_counts-_histogram_plot.eps',
                format = 'eps',
                dpi = 600,
                bbox_inches = 'tight')
    
    plt.savefig(save_directory + '/Validation_set_genes_RNAseq_value_counts-_histogram_plot.tiff',
                format = 'tiff',
                dpi = 600,
                bbox_inches = 'tight')
    
    plt.savefig(save_directory + '/Validation_set_genes_RNAseq_value_counts-_histogram_plot.svg',
                format = 'svg',
                dpi = 600,
                bbox_inches = 'tight')
    
    validation_RNAseq_dataframe = pd.Series(np.squeeze(y_val))
    
    # Define gene expression catagories for analysis.
    validation_all_zero_true_expression = validation_RNAseq_dataframe[validation_RNAseq_dataframe  == 0]
    validation_true_expression_between_0_and_5 = validation_RNAseq_dataframe[(validation_RNAseq_dataframe > 0) & (validation_RNAseq_dataframe < 5)]
    validation_true_expression_between_5_and_10 = validation_RNAseq_dataframe[(validation_RNAseq_dataframe >= 5) & (validation_RNAseq_dataframe < 10)]
    validation_true_expression_between_10_and_15 = validation_RNAseq_dataframe[(validation_RNAseq_dataframe >= 10) & (validation_RNAseq_dataframe <= 15)]

    
    validation_expression_counts_dataframe = pd.DataFrame({"expression catagory after log(2) transformation" : 
                                                           ["all zero", 
                                                            "between 0 and 5",
                                                            "between 5 and 10", 
                                                            "between 10 and 15" ],
                                                           "count": [len(validation_all_zero_true_expression),
                                                                     len(validation_true_expression_between_0_and_5), 
                                                                     len(validation_true_expression_between_5_and_10),
                                                                     len(validation_true_expression_between_10_and_15)]})

    dataframes = [validation_expression_counts_dataframe]
    dataset_names = ['Validation Set']
    # Visualize number of genes in each catagory.
    for l in range(len(dataframes)):
        plt.close()
        sn.set_theme(style = 'whitegrid')
        fig, ax = plt.subplots(figsize = (8, 5))
        ax = sn.barplot(data = dataframes[l], 
                        x = "expression catagory after log(2) transformation", 
                        y = "count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation = "45")
        ax.set(title = f'{dataset_names[l]} - XGBoost Cross Patient Regression Expression Catagory Counts')
        sn.set(font_scale = 1)
        for i in ax.containers:
            ax.bar_label(i,)
        plt.savefig(save_directory + 
                    '/xgboost_cross_patient_regression_' + dataset_names[l] + 
                    '_Expression_Catagory_Counts.png',
                    format = 'png',
                    dpi = 600,
                    bbox_inches = 'tight')
        
        plt.savefig(save_directory + 
                    '/xgboost_cross_patient_regression_' + dataset_names[l] + 
                    '_Expression_Catagory_Counts.eps',
                    format = 'eps',
                    dpi = 600,
                    bbox_inches = 'tight')
        
        plt.savefig(save_directory + 
                    '/xgboost_cross_patient_regression_' + dataset_names[l] + 
                    '_Expression_Catagory_Counts.tiff',
                    format = 'tiff',
                    dpi = 600,
                    bbox_inches = 'tight')
        
        plt.savefig(save_directory + 
                    '/xgboost_cross_patient_regression_' + dataset_names[l] + 
                    '_Expression_Catagory_Counts.svg',
                    format = 'svg',
                    dpi = 600,
                    bbox_inches = 'tight')        

    return None

def visualize_model_test_results(result_1, result_2, result_3):
    '''
    Creates a bar graph visualization of the test metric 
    results for the current model run. The image is saved to the model's 
    image directory.
    
    param results_list: The output of model.evaluate .
    
    returns: Nothing
    '''
    
    results_dataframe = pd.DataFrame({"metric" : ["PCC", "SCC", "R2"],
                                      "test set results": [result_1, result_2, result_3]})
    plt.close()
    sn.set_theme(style = 'whitegrid')
    fig, ax = plt.subplots(figsize = (8, 5))
    ax = sn.barplot(data = results_dataframe, x = "metric", y = "test set results")
    ax.set_xticklabels(ax.get_xticklabels())
    ax.set(title = 'XGBoost Cross Patient Regression Test Results')
    sn.set(font_scale=1)
    for i in ax.containers:
        ax.bar_label(i,)
    plt.savefig(save_directory + 
                '/xgboost_cross_patient_regression_gsc_stem_standard_test_metric_results.png', 
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')
    return None


def visualize_se_heatmap(se_values, gene_names_in_test_set):
    '''
    Creates a heatmap visualization of portions of the number of genes 
    in the test set and their Squared Error values.
    
    param se_values: Array of calculated SE per gene.
    param gene_names_in_test_set: List of each gene name in the test set.
    
    returns: Nothing
    '''
    
    plt.close()
    fig, ax = plt.subplots(figsize = (5, 20))
    
    # The number and range of genes presented can be adjusted by slicing.
    genes_for_vis = se_values[:50]
    
    ax = sn.heatmap(genes_for_vis.reshape(genes_for_vis.shape[0], 1), 
                    cmap = "YlGnBu", annot = True)
    
    # The number and range of genes presented can be adjusted by slicing as above.
    ax.set_yticklabels(gene_names_in_test_set[:50], rotation = 0)
    
    ax.tick_params(left=True, bottom=False)
    ax.set_xticklabels([])
    #ax.set_yticks(np.arange(len(gene_names_in_test_set[:10])), labels = gene_names_in_test_set[:10], rotation = 0)
    plt.savefig(save_directory + 
                '/xgboost_cross_patient_regression_gsc_stem_standard_test_gene_squared_error_heatmap.png',
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')

    
    return None

def load_csv_and_create_dataframes():
    
    prediction_dataframe = pd.read_csv(save_directory + '/xgboost_cross_patient_regression_gsc_stem_standard_test_predictions.csv', 
                                       float_precision='round_trip')
    
    # Define gene expression catagories for analysis.

    all_zero_true_expression = prediction_dataframe[prediction_dataframe[' true RNAseq value'] == 0]
    #true_expression_between_0_and_5 = prediction_dataframe[(prediction_dataframe[' true RNAseq value'] > 0) & (prediction_dataframe[' true RNAseq value'] < 5)]
    
    ## Modified 1/27/25 to produce new visualizations
    ## Defining "low" expression range to be 0 to 5
    true_expression_between_0_and_5 = prediction_dataframe[(prediction_dataframe[' true RNAseq value'] > 0) & (prediction_dataframe[' true RNAseq value'] <= 5)]
    true_expression_between_5_and_10 = prediction_dataframe[(prediction_dataframe[' true RNAseq value'] >= 5) & (prediction_dataframe[' true RNAseq value'] < 10)]
    #true_expression_between_10_and_15 = prediction_dataframe[(prediction_dataframe[' true RNAseq value'] >= 10) & (prediction_dataframe[' true RNAseq value'] <= 15)]
    
    ## Modified 1/27/25 to produce new visualizations
    ## Defining "high" expression range to be 10 and above
    true_expression_between_10_and_15 = prediction_dataframe[(prediction_dataframe[' true RNAseq value'] >= 10)]
                                                             
    #print(len(all_zero_expression) + len(between_0_and_5) + len(between_5_and_10) + len(between_10_and_15))
    
    return prediction_dataframe, \
           all_zero_true_expression, \
           true_expression_between_0_and_5, \
           true_expression_between_5_and_10, \
           true_expression_between_10_and_15


def visualize_testing_distributions(all_zero_true_expression, 
                                    true_expression_between_0_and_5, 
                                    true_expression_between_5_and_10, 
                                    true_expression_between_10_and_15):
    
    # Create dataframe for catagory counts. The dataframe will be used for visualization.
    expression_counts_dataframe = pd.DataFrame({"expression catagory after log(2) transform" : 
                                                ["all zero", 
                                                 "between 0 and 5",
                                                 "between 5 and 10", 
                                                 "between 10 and 15" ],
                                                "count": [len(all_zero_true_expression), 
                                                          len(true_expression_between_0_and_5),
                                                          len(true_expression_between_5_and_10),
                                                          len(true_expression_between_10_and_15)]})

    # Visualize number of genes in each catagory.
    plt.close()
    sn.set_theme(style = 'whitegrid')
    fig, ax = plt.subplots(figsize = (8, 5))

    ax = sn.barplot(data = expression_counts_dataframe, 
                    x = "expression catagory after log(2) transform", 
                    y = "count")
    ax.set_xticklabels(ax.get_xticklabels(), 
                       rotation = "45")
    ax.set(title = 'Testing Set - XGBoost Cross Patient Regression Expression Catagory Counts')
    sn.set(font_scale=1)
    for i in ax.containers:
        ax.bar_label(i,)

    plt.savefig(save_directory + 
                '/xgboost_cross_patient_regression_gsc_stem_standard_test_expression_catagory_counts.png',
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')
    
    return None

def visualize_testing_set_mse_by_catagory(test_set_MSE, 
                                          all_zero_true_expression, 
                                          true_expression_between_0_and_5, 
                                          true_expression_between_5_and_10, 
                                          true_expression_between_10_and_15):
    
    # Create dataframe for the calculation of the mean SE. 
    # The dataframe will be used for visualization.
    expression_mean_dataframe = pd.DataFrame({"expression catagory after log(2) transform" : 
                                              ["all zero", 
                                               "between 0 and 5",
                                               "between 5 and 10", 
                                               "between 10 and 15" ],
                                              "mean squared error (MSE)" : 
                                              [all_zero_true_expression[' prediction Squared Error (SE)'].mean(),
                                               true_expression_between_0_and_5[' prediction Squared Error (SE)'].mean(),
                                               true_expression_between_5_and_10[' prediction Squared Error (SE)'].mean(),
                                               true_expression_between_10_and_15[' prediction Squared Error (SE)'].mean()]})
    
    plt.close()
    sn.set_theme(style = 'whitegrid')
    fig, ax = plt.subplots(figsize = (8, 5))
    ax = sn.barplot(data = expression_mean_dataframe, 
                    x = "expression catagory after log(2) transform", 
                    y = "mean squared error (MSE)")
    ax.set(title = 'Testing Set - XGBoost Cross Patient Regression MSE Per Expression Catagory')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = "45")
    ax.axhline(test_set_MSE, label = f'MSE of entire test set : {test_set_MSE}')
    plt.legend(loc = 'upper left')
    for i in ax.containers:
        ax.bar_label(i,)   
    plt.savefig(save_directory + 
                '/xgboost_cross_patient_regression_gsc_stem_standard_test_expression_catagory_MSE.png',
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')
    
    return None

def visualize_prediction_mse(prediction_ses, y_true, y_pred):
    '''
    Creates multiple visualizations for the for the 
    test set.
    param prediction_ses: Array of calculated Squared Error values per gene.
    param y_true: the true (observed) RNAseq values.
    param y_pred: the model's predicted values.

    returns: Nothing.
    '''
    plt.close()
    sn.set_theme(style = 'whitegrid')
    plt.title('Mean Squared Error Values for Test Set Genes')
    plt.ylabel('SE')
    plt.xlabel('Gene index in test set')
    plt.scatter(np.arange(prediction_ses.shape[0]), prediction_ses)
    plt.savefig(save_directory + 
                '/xgboost_cross_patient_regression_test_set_mse_values.png',
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')


    plt.close()
    sn.set_theme(style = 'whitegrid')
    plt.title('Squared Error value counts for the Test Set Genes')
    plt.ylabel('count')
    plt.xlabel('Squared Error')
    sn.histplot(prediction_ses, legend = False, palette = ['orange'], bins = 50)
    plt.savefig(save_directory + 
                '/xgboost_cross_patient_regression_test_set_mse_values_-_histogram_plot.png',
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')


    plt.close()
    sn.set_theme(style = 'whitegrid')
    plt.title('True RNAseq Values for the Test Set Genes')
    plt.ylabel('count')
    plt.xlabel('True values')
    sn.histplot(y_true, legend = False, palette = ['red'], bins = 50)
    plt.savefig(save_directory + 
                '/xgboost_cross_patient_regression_test_set_true_RNAseq_values_-_histogram_plot.png',
                format = 'png',
                dpi = 600, 
                bbox_inches = 'tight')


    plt.close()
    sn.set_theme(style = 'whitegrid')
    plt.title('Predicted RNAseq Values for the Test Set Genes')
    plt.ylabel('count')
    plt.xlabel('Predicted values')
    sn.histplot(y_pred, legend = False, palette = ['green'], bins = 50)
    plt.savefig(save_directory + 
                '/xgboost_cross_patient_regression_test_set_predicted_RNAseq_values_-_histogram_plot.png',
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')


    return None

def visualize_test_obs_pred(y_true, y_pred):
    '''
    Creates multiple visualizations showing the observed (true) RNAseq values
    vs the predicted RNAseq values.
    
    param y_true: the true (observed) RNAseq values.
    param y_pred: the model's predicted values.

    return: Nothing
    
    '''
    plt.close()
    fig, ax = plt.subplots()
    plt.title("RNAseq Observed Values vs Predicted Values")
    plt.ylabel("Predicted Values")
    plt.xlabel("True Values")
    ax.scatter(y_true, y_pred)
    ax.axline([0, 0], [1, 1], color = 'black')
    plt.xlim(0, 15)
    plt.xlim(0, 15)
    plt.axis('square')
    plt.savefig(save_directory + 
                '/xgboost_cross_patient_regression_test_set_observed_vs_predicted.png',
                format = 'png',
                dpi = 600, 
                bbox_inches='tight')


    # Create dataframe of true and predicted values for visualization.
    data = {'True Values': np.squeeze(y_true), 'Predicted Values': np.squeeze(y_pred)}
    df = pd.DataFrame(data = data)

    # Regular joint plot
    plt.close()
    #plt.title("RNAseq Observed Values vs Predicted Values Joint Plot")
    sn.jointplot(x = 'True Values', 
                 y = 'Predicted Values', 
                 data = df)
    plt.savefig(save_directory + 
                '/xgboost_cross_patient_regression_test_set_observed_vs_predicted_joint_plot.png',
                format = 'png',
                dpi = 600, 
                bbox_inches='tight')


    # Kernal Density Estimation joint plot
    plt.close()
    #plt.title("RNAseq Observed Values vs Predicted Values KDE Joint Plot")
    sn.jointplot(x = 'True Values', 
                 y = 'Predicted Values', 
                 data = df, 
                 kind = 'kde')
    plt.savefig(save_directory + 
                '/xgboost_cross_patient_regression_test_set_observed_vs_predicted_KDE_joint_plot.png',
                format = 'png',
                dpi = 600, 
                bbox_inches='tight')


    return None

def visualize_aggregated_input_profiles(test_dataset, 
                                        all_zero_true_expression, 
                                        true_expression_between_0_and_5, 
                                        true_expression_between_5_and_10, 
                                        true_expression_between_10_and_15, 
                                        prediction_dataframe):
    '''
    Creates aggregated heatmap visualizations for genes within the predefined
    true expression value groups. 
    
    NOTE: Although a provision was coded to normalize the features for visualization,
    it is currently not in use because the input features are already standardized.
    
    returns: Nothing
    '''

    
    heatmap_indexes = [list(all_zero_true_expression.index), list(true_expression_between_0_and_5.index), list(true_expression_between_5_and_10.index), list(true_expression_between_10_and_15.index), list(prediction_dataframe.index)]
    heatmap_names = ['Aggregate model input - Zero Expression Genes','Aggregate model input - Genes with true expression values between 0 and 5.', 'Aggregate model input - Genes with true expression values between 5 and 10.','Aggregate model input - Genes with true expression values from 10 to 15.', 'Aggregate model input - All genes in test set.']
    #min_max_scaler = preprocessing.MinMaxScaler()
    for h in tqdm(range(len(heatmap_indexes))):
        mean_gene_vals = np.mean(test_dataset[heatmap_indexes[h]], axis = 0)
        # Script log file.
        with open(save_directory + 
                  '/H3K27Ac_signal_values.csv', 'a') as log:
            log.write('\n' f'H3K27Ac values - {heatmap_names[h]}')
            log.write('\n' f'{mean_gene_vals[:,0]}')                                                 
        # Normalize each feature seperately.
        # NOTE: Normalization has been disabled here because the input has already been standardized.
        #for i in range(mean_gene_vals.shape[1]):
            #mean_gene_vals[:, i] = mean_gene_vals[:, i] / mean_gene_vals[:, i].max() # Normalize by dividing by max
            #mean_gene_vals[:, i] = mean_gene_vals[:, i] / np.sum(mean_gene_vals[:, i]) # Normalize by dividing by sum
            ####mean_gene_vals[:, i] = np.squeeze(min_max_scaler.fit_transform(mean_gene_vals[:, i].reshape(-1,1)))

        #plt.close()
        #plt.imshow(mean_gene_vals)
        #fig, ax = plt.subplots()
        #c = ax.pcolormesh(mean_gene_vals.T, cmap = 'Greys')
        #fig.colorbar(c)
        #fig.suptitle(f'{heatmap_names[h]}')
        #ax.set(title = "Values standardized for model input.")
        #ax.set_xlabel('bins')
        #ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        #ax.set_yticklabels(['RNA Pol II','ATAC','CTCF','H3K27ac'])
        #ax.set_ylabel('epigenetic features')
        #plt.savefig('pngs_regression_gsc_stem_standard_log2/' + heatmap_names[h] + '.png')
        #plt.show()

        # Use seaborn to plot
        plt.close()
        #grid_kws = {'height_ratios': (0.9, 0.05), 'hspace': 0.3}
        #fig, (ax, cbar_ax) = plt.subplots(2, gridspec_kw = grid_kws, figsize = (20, 5))
        fig, ax = plt.subplots(figsize=(15, 5))
        #fig, ax = plt.subplots(figsize = (20, 5))  
        #ax = sn.heatmap(mean_gene_vals.T, annot = True, cbar_ax = cbar_ax, cbar_kws = {'orientation': 'horizontal'})
        fig.suptitle(f'{heatmap_names[h]}')
        
        ax = sn.heatmap(mean_gene_vals.T, cbar = False, annot = True, fmt='.7f', cmap = 'OrRd', annot_kws={'rotation':90})
        ax.set_xlabel('bins')
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(['H3K27Ac','CTCF','ATAC','RNPPII'])
        ax.set_ylabel('epigenetic features')
        ax.text(x = 0.5, y = 1.04, 
                s = f'Feature values after standardization.', 
                fontsize = 10, 
                ha = 'center', 
                va = 'bottom', 
                transform = ax.transAxes)
        plt.savefig(save_directory + '/' + heatmap_names[h] + '_seaborn.png',
                    format = 'png',
                    dpi = 600,
                    bbox_inches='tight')
        plt.savefig(save_directory + '/' + heatmap_names[h] + '_seaborn.eps',
                    format = 'eps',
                    dpi = 600,
                    bbox_inches='tight')        
        plt.savefig(save_directory + '/' + heatmap_names[h] + '_seaborn.tiff',
                    format = 'tiff',
                    dpi = 600,
                    bbox_inches='tight')
        plt.savefig(save_directory + '/' + heatmap_names[h] + '_seaborn.svg',
                    format = 'svg',
                    dpi = 600,
                    bbox_inches='tight')        
        
        # Use seaborn to plot
        #plt.close
        #ax = sn.heatmap(mean_gene_vals)
        #plt.show()

        # Create dataframe for visualization 
        ####df = pd.DataFrame(mean_gene_vals, columns = ['H3K27ac','CTCF','ATAC', 'RNA Pol II'])
        ####plt.close()
        ####fig, ax = plt.subplots()
        #fig, ax = plt.subplots(figsize = (20, 5))
        ####fig.suptitle(f'Cross patient regression pairplot for {heatmap_names[h]}')
        #plt.title('Regression Pairplot for' + heatmap_names[h]) 
        ####ax = sn.pairplot(df, palette ='coolwarm')
        ####plt.savefig('pngs_cnn_cross_patient_pred_regression_gsc_stem_standard_log2/' + heatmap_names[h] + '_pairplot.png')
        #plt.show()  

    return None


def main(loss_dict, pcc_dict, r2_score_dict, scc_dict, 
         val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, 
         gene_dict, num_genes, count, 
         learning_rates, n_estimators, max_depths, 
         min_child_weight, colsample_bytree, subsample, gamma):
    
    # Save directory - path where result files and figures are saved
    global save_directory
    
    now = datetime.datetime.now()

    if sys.argv[5:]:
        # Save path given by the user in the 5th argument to the global variable
        save_directory = str(sys.argv[5])
        # Create the given directory
        print('*'*25)
        print(f'Using {save_directory} as the save directory.')
        print('*'*25)
        os.makedirs(save_directory, exist_ok = True)

    else:
        save_directory = './cross_patient_regression_using_xgboost_-_results_and_figures_-_' + \
        str(now.month) + '-' + str(now.day) + '-' + str(now.year) + '_' + \
        'at_' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)
        print('*'*25)
        print('Using the default save directory:')
        print(f'{save_directory}')
        print('since a directory was not provided.')
        print('*'*25)
        os.makedirs(save_directory, exist_ok = True)
    
    # Indicate True or False for the creation of a validation set. The script will fit the model accordingly.
    validation = False
    
    # Get file path from command line
    # NOTE: file_path_1 is the datafile for training and validating the model.    
    # NOTE: file_path_2 is the datafile for testing the model.
    # NOTE: indices is the 'shuffle_index' used to shuffle the datasets
    # NOTE: random_state is the random seed for the run
    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]
    indices = sys.argv[3]
    random_state = int(sys.argv[4])
    print('*'*25)
    print('The random seed is set to: ')
    print(random_state)
    print('*'*25)
    
    # Call get_data() to process the data, preprocess = True will read in processed .npy files,
    # if false then will re-preprocess data
    print("Processing data")
    if count == 0:
        preprocess_bool = True
    else:
        preprocess_bool = False
        
    if validation == True:
        validation_bool = True
    else:
        validation_bool = False


    # Processing data for patient 1 file to produce train and validation sets OR training
    # set only if validation == False
    if validation_bool == True:
        X_train, X_val, Y_train, Y_val, gene_dict, num_genes = get_data_patient_1(file_path_1,
                                                                                  indices,
                                                                                  gene_dict,
                                                                                  num_genes,
                                                                                  preprocess = preprocess_bool,
                                                                                  validation = validation_bool)
        
    else:
        X_train, Y_train, gene_dict, num_genes = get_data_patient_1(file_path_1,
                                                                    indices,
                                                                    gene_dict,
                                                                    num_genes,
                                                                    preprocess = preprocess_bool,
                                                                    validation = validation_bool)
        

    # Processing data for patient 2 file to produce test set.
    X_test, Y_test, gene_dict, num_genes, test_set_indices = get_data_patient_2(file_path_2, 
                                                                                indices, 
                                                                                gene_dict, 
                                                                                num_genes, 
                                                                                preprocess = preprocess_bool)

    # Call train_model() to train the model
    print("Training model...")
    
    if validation_bool == True:
        model, PCC, SCC, R2  = train_model(X_train, Y_train,  
                                           validation = validation_bool, 
                                           learning_rates = learning_rates, 
                                           n_estimators = n_estimators, 
                                           max_depths = max_depths, 
                                           min_child_weight = min_child_weight, 
                                           colsample_bytree = colsample_bytree, 
                                           subsample = subsample, 
                                           gamma = gamma, 
                                           count = count,
                                           random_state = random_state, 
                                           X_val = X_val, Y_val = Y_val)

        max_val_pcc = PCC
        max_val_r2_score = R2
        max_val_scc = SCC
        

    else:
        model = train_model(X_train, Y_train, 
                            validation = validation_bool, 
                            learning_rates = learning_rates, 
                            n_estimators = n_estimators, 
                            max_depths = max_depths, 
                            min_child_weight = min_child_weight, 
                            colsample_bytree = colsample_bytree, 
                            subsample = subsample, 
                            gamma = gamma, count = count,
                            random_state = random_state,
                            X_val = None, Y_val = None)
        
        max_val_pcc = 'TRAINING SET ONLY'
        max_val_r2_score = 'TRAINING SET ONLY'
        max_val_scc = 'TRAINING SET ONLY'


    val_pcc_dict[count] = max_val_pcc
    val_r2_score_dict[count] = max_val_r2_score
    val_scc_dict[count] = max_val_scc


    print("CURRENT COUNT:", count,
          "\n learning rate: ", learning_rates,
          "\n n estimator: ", n_estimators,
          "\n max depth: ", max_depths,
          "\n Val PCC: ", max_val_pcc,
          "\n Val SCC: ", max_val_scc,
          "\n Val R2: ", max_val_r2_score)        

    print('*'*25)
    print("Evaluating model...")
    print('*'*25)
    
    test_PCC, test_SCC, test_R2  = test_model(model, X_test, Y_test, 
                                              learning_rates = learning_rates, 
                                              n_estimators = n_estimators, 
                                              max_depths = max_depths, 
                                              min_child_weight = min_child_weight, 
                                              colsample_bytree = colsample_bytree, 
                                              subsample = subsample, 
                                              gamma = gamma, 
                                              count = count)
    print("Test results:")
    print(f"PCC,{test_PCC}")
    print(f"SCC,{test_SCC}")
    print(f"R2,{test_R2}")
    print('*'*25)




    # Script log file.
    with open(save_directory + 
              '/xgboost_cross_patient_regression_gsc_stem_standard_log2_info.csv', 'a') as log:
        log.write('\n' f'{now.strftime("%H:%M on %A %B %d")},')
        log.write(f'CURRENT COUNT: {count},learning rate: {learning_rates},n estimator: {n_estimators}, max depth: {max_depths},')
        log.write(f'min child weight: {min_child_weight},column sample by tree: {colsample_bytree},subsample: {subsample}, gamma: {gamma},')
        log.write(f'Val PCC: {max_val_pcc},Val SCC: {max_val_scc}, Val R2: {max_val_r2_score},')
        log.write(f'Test PCC: {test_PCC},Test SCC: {test_SCC}, Test R2: {test_R2}')

        

    # Calculate mean and sum feature importances from trained model    
    h3k27ac_mean_importances, atac_mean_importances, ctcf_mean_importances, \
           rnapii_mean_importances, h3k27ac_sum_importances, atac_sum_importances, \
           ctcf_sum_importances, rnapii_sum_importances = get_feature_importances(model)
    
    
    print('*'*25)
    print("Feature importances...")
    print('*'*25)
    print('Mean')
    print(f"H3K27Ac Mean Importance,{h3k27ac_mean_importances}")
    print(f"CTCF Mean Importance,{ctcf_mean_importances}")
    print(f"ATAC Mean Importance,{atac_mean_importances}")
    print(f"RNAPII Mean Importance,{rnapii_mean_importances}")
    print('*'*25)
    print('Sum')
    print(f"H3K27Ac Sum Importance,{h3k27ac_sum_importances}")    
    print(f"CTCF Sum Importance,{ctcf_sum_importances}")
    print(f"ATAC Sum Importance,{atac_sum_importances}")
    print(f"RNAPII Sum Importance,{rnapii_sum_importances}")
    print('*'*25)
    
    # Feature importance visualizations
    visualize_feature_importances_sums(h3k27ac_mean_importances, atac_mean_importances, ctcf_mean_importances, \
           rnapii_mean_importances, h3k27ac_sum_importances, atac_sum_importances, \
           ctcf_sum_importances, rnapii_sum_importances)

    visualize_feature_importances_means(h3k27ac_mean_importances, atac_mean_importances, ctcf_mean_importances, \
           rnapii_mean_importances, h3k27ac_sum_importances, atac_sum_importances, \
           ctcf_sum_importances, rnapii_sum_importances)

    
    gene_names_in_test_set = get_gene_names(gene_dict, indices, X_test.shape[0], num_genes, test_set_indices)
    
    
    Y_pred = make_prediction(model, X_test)    
    prediction_se = calculate_se_for_predictions(Y_test, Y_pred)    
    prediction_csv(prediction_se, Y_test, Y_pred, gene_names_in_test_set)    
    
    visualize_training_distributions(Y_train)
    
    if validation_bool == True:
        visualize_validation_distributions(Y_val)
    
    visualize_model_test_results(test_PCC, test_SCC, test_R2)
    visualize_se_heatmap(prediction_se, gene_names_in_test_set)
    prediction_dataframe, all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15 = load_csv_and_create_dataframes()
    visualize_testing_distributions(all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15)
    
    # Mean Squared Error for entire test set.
    test_set_MSE = prediction_dataframe[' prediction Squared Error (SE)'].mean()

    visualize_testing_set_mse_by_catagory(test_set_MSE, all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15)
    visualize_prediction_mse(prediction_se, Y_test, Y_pred)    
    visualize_test_obs_pred(Y_test, Y_pred)
    visualize_aggregated_input_profiles(X_test, all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15,prediction_dataframe)    
    
    #return loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, X_train, X_val, X_test, Y_train, Y_val, Y_test, indices, model, test_PCC, test_SCC, test_R2

    return loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, indices, model, test_PCC, test_SCC, test_R2

if __name__ == '__main__':

    loss_dict = {}
    pcc_dict = {}
    r2_score_dict = {}
    scc_dict = {}
    val_loss_dict = {}
    val_pcc_dict = {}
    val_r2_score_dict = {}
    val_scc_dict = {}
    gene_dict = {}
    num_genes = 0

    
    # New Hyperparameter tuning combination specifically for GSC2 -> GSC1
    #parameters = dict(min_child_weight = [1], colsample_bytree = [0.4], subsample = [1.0], gamma = [0.7], n_estimators = [150], max_depths = [7], learning_rates = [0.04])
    
    
    # Hyperparameter combination that produces the best validation metrics
    parameters = dict(min_child_weight = [1], colsample_bytree = [0.2], subsample = [1.0], gamma = [0.3], n_estimators = [150], max_depths = [6], learning_rates = [0.03])
    
    
    
    # Hyperparameter grid search
    #parameters = dict(min_child_weight = [1, 3, 5, 7, 9], colsample_bytree = [0.2, 0.4, 0.6, 0.8, 1.0], subsample = [0.2, 0.4, 0.6, 0.8, 1.0], gamma = [0.0, 0.1, 0.3, 0.5, 0.7], n_estimators = [25, 50, 75, 100, 110, 125, 130, 150], max_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25], learning_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.17, 0.20, 0.25, 0.30, 0.40, 0.50])

    
       
    param_values = [v for v in parameters.values()]

    count = 0

    for mc, cs, ss, ga, ne, md, lr in product(*param_values): 
        loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, indices, model, test_PCC, test_SCC, test_R2 = main(loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, count, learning_rates = lr, n_estimators = ne, max_depths = md, min_child_weight = mc, colsample_bytree = cs, subsample = ss, gamma = ga)

#    for mc, cs, ss, ga, ne, md, lr in product(*param_values): 
#        loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, #num_genes, X_train, X_val, X_test, Y_train, Y_val, Y_test, indices, model, test_PCC, test_SCC, test_R2 = main(loss_dict, #pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, count, #learning_rates = lr, n_estimators = ne, max_depths = md, min_child_weight = mc, colsample_bytree = cs, subsample = ss, gamma = #ga)
        count += 1

    #min_loss_count = min(loss_dict, key=loss_dict.get)
    #max_pcc_count = max(pcc_dict, key=pcc_dict.get)
    #max_r2_count = max(r2_score_dict, key=r2_score_dict.get)
    #max_scc_count = max(scc_dict, key=scc_dict.get)
    #min_val_loss_count = min(val_loss_dict, key=val_loss_dict.get)
    max_val_pcc_count = max(val_pcc_dict, key=val_pcc_dict.get)
    max_val_r2_count = max(val_r2_score_dict, key=val_r2_score_dict.get)
    max_val_scc_count = max(val_scc_dict, key=val_scc_dict.get)
    

    #print("\n Min training loss and count: ", min(loss_dict.values()), min_loss_count, "\n Max training pcc and count: ", max(pcc_dict.values()), max_pcc_count, "\n Max training R2 and count: ", max(r2_score_dict.values()), max_r2_count, "\n Max training scc and count: ", max(scc_dict.values()), max_scc_count, "\n Min val loss and count: ", min(val_loss_dict.values()), min_val_loss_count, "\n Max val pcc and count: ", max(val_pcc_dict.values()), max_val_pcc_count, "\n Max val R2 and count: ", max(val_r2_score_dict.values()), max_val_r2_count, "\n Max val scc and count: ", max(val_scc_dict.values()), max_val_scc_count)
    
