'''
This is a Support Vector Regression version of a Support Vector Machine 
algorithm model. It is intended to be one of the baseline models 
for the cross patient regression analysis.
'''

import sys
import numpy as np
import os
from sklearn import svm
from sklearn.svm import SVR
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
import shap
import pandas as pd

def get_data_patient_1(file_path, indices, gene_dict, num_genes, preprocess, validation):
    '''
    Returns X_train, X_val, Y_train, Y_val; where X refers to inputs and Y refers to labels.
    '''
    
    if preprocess:
       
        print("Loading patient 1 dataset...")
        # Col 1 = gene names, 2 = bin number, 3-6 = features, 7 = labels
        combined_diff = np.load(file_path, allow_pickle=True)
        
        # Get all the unique gene names
        gene_names = np.unique(combined_diff[:, 0])
        num_genes = len(gene_names)
        
        # Create a dictionary to map each gene name to a unique index (number like 0, 1, 2,...,#genes-1)
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
            # Each subset is of shape 100 x 6 (number of 100bp bins x number of columns)
            subset = combined_diff[np.where(combined_diff[:, 0] == name)]

            # Create matrix of data
            gene_ind = gene_dict[name]
            data = subset[:, 2:]
            
            # data_inputs = np.transpose(data[:, :-1])
            data_inputs = data[:, :-1]
                                    
            # Add to array at the unique id position
            X[gene_ind] = data_inputs

            # Set corresponding value to be first bin's RNAseq value (since all 50 bins
            # have the same value when using the featureCounts utility and process).
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
        
        # Shuffle the data
        #ind = np.arange(0, num_genes)
        # np.random.shuffle(ind)
        ind = np.load(indices, allow_pickle=True)

        
        if validation == True:
            # HYPERPARAMETER TUNING SPLITS
            # Create train (70%), validation (30%).
            train_ind = ind[0: int(0.7*num_genes)]
            val_ind = ind[int(0.7*num_genes):]
        
        else:
            # TESTING SPLITS
            # The training set will have 99% of the patient 1 data to train the model.
            # The validation set is reduced to 1% but ket to not break the function.
            train_ind = ind
            #train_ind = ind[0: int(0.99*num_genes)]
            val_ind = ind[int(0.99*num_genes):]

        
        X_train = X[train_ind]
        X_val = X[val_ind]
        

        Y_train = Y[train_ind]
        Y_val = Y[val_ind]
        

        # List of all datasets after split operation.
        # datasets = [X_train, X_val, X_test, Y_train, Y_val, Y_test]
        # Standardization ONLY on input variables.
        #datasets = [X_train, X_val, X_test]
        datasets = [X_train, X_val]

        # Perform calculation on each column of the seperate train, validation and test sets. 
        for dataset in datasets:
            for i in range(dataset.shape[2]):
                # Standardize the column values.
                dataset[:, :, i] = (dataset[:, :, i] - np.mean(dataset[:, :, i])) / np.std(dataset[:, :, i], ddof = 1) # The degrees of freedom is set t
                
        np.save("X_cross_patient_regression_patient_1_stem_standard_log2_train", X_train, allow_pickle=True)
        np.save("X_cross_patient_regression_patient_1_stem_standard_log2_val", X_val, allow_pickle=True)
        
        np.save("Y_cross_patient_regression_patient_1_stem_standard_log2_train", Y_train, allow_pickle=True)
        np.save("Y_cross_patient_regression_patient_1_stem_standard_log2_val", Y_val, allow_pickle=True)
        
    
    else:
        X_train = np.load("X_cross_patient_regression_patient_1_stem_standard_log2_train.npy", allow_pickle=True)
        X_val = np.load("X_cross_patient_regression_patient_1_stem_standard_log2_val.npy", allow_pickle=True)
        
        Y_train = np.load("Y_cross_patient_regression_patient_1_stem_standard_log2_train.npy", allow_pickle=True)
        Y_val = np.load("Y_cross_patient_regression_patient_1_stem_standard_log2_val.npy", allow_pickle=True)
        
    
        gene_dict = gene_dict
        num_genes = num_genes


    return X_train, X_val, Y_train, Y_val, gene_dict, num_genes

def get_data_patient_2(file_path, indices, gene_dict, num_genes, preprocess):
    '''
    Returns X_test, Y_test; where X refers to inputs and Y refers to labels.
    '''

    if preprocess:
        print("Loading patient 2 dataset...")
        # Col 1 = gene names, 2 = bin number, 3-6 = features, 7 = labels
        combined_diff = np.load(file_path, allow_pickle=True)
        
        # Get all the unique gene names
        gene_names = np.unique(combined_diff[:, 0])
        num_genes = len(gene_names)

        # Create a dictionary to map each gene name to a unique index (number like 0, 1, 2,...,#genes-1)
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

            # Each subset is of shape 100 x 6 (number of 100bp bins x number of columns)
            subset = combined_diff[np.where(combined_diff[:, 0] == name)]

            # Create matrix of data. 
            gene_ind = gene_dict[name]
            data = subset[:, 2:]

            data_inputs = data[:, :-1]

            # Add to array at the unique id position
            X[gene_ind] = data_inputs
   
            # Set corresponding value to be first bin's RNAseq value (since all 50 bins
            # have the same value when using the featureCounts utility and process).
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

        # Shuffle the data
        #ind = np.arange(0, num_genes)
        # np.random.shuffle(ind)
        ind = np.load(indices, allow_pickle=True)

        # Splits for this patient data can be adjusted here.
        #train_ind = ind[0: int(0.7*num_genes)]
        #val_ind = ind[int(0.7*num_genes):int(0.85*num_genes)]
        #test_ind = ind[int(0.85*num_genes):]


        # NOTE: For now use entire dataset for test set.
        test_ind = ind

        #X_train = X[train_ind]
        #X_val = X[val_ind]
        # Use all of the dataset for test.
        X_test = X[test_ind]

        #Y_train = Y[train_ind]
        #Y_val = Y[val_ind]
        # Use all of the dataset for test.
        Y_test = Y[test_ind]

        # List of all datasets after split operation.
        # datasets = [X_train, X_val, X_test, Y_train, Y_val, Y_test]
        # Standardization ONLY on input variables.
        #datasets = [X_train, X_val, X_test]
        datasets = [X_test]

        # Perform calculation on each column of the seperate train, validation and test sets.
        for dataset in datasets:
            for i in range(dataset.shape[2]): ### Standardization on all columns.
            #for i in range(1,dataset.shape[2]): ### NO standardization on column 0 - H3K27ac for perturbation analysis ONLY.
                # Standardize the column values.
                dataset[:, :, i] = (dataset[:, :, i] - np.mean(dataset[:, :, i])) / np.std(dataset[:, :, i], ddof = 1)
                

        np.save("X_cross_patient_regression_patient_2_stem_standard_log2_test", X_test, allow_pickle=True)
        np.save("Y_cross_patient_regression_patient_2_stem_standard_log2_test", Y_test, allow_pickle=True)

    else:
        X_test = np.load("X_cross_patient_regression_patient_2_stem_standard_log2_test.npy", allow_pickle=True)
        Y_test = np.load("Y_cross_patient_regression_patient_2_stem_standard_log2_test.npy", allow_pickle=True)

        gene_dict = gene_dict
        num_genes = num_genes

    return X_test, Y_test, gene_dict, num_genes


def reset_random_seeds(seed):
    '''
    Takes a given number and assigns it
    as a random seed to various generators and the
    os environment.
    '''

    os.environ['PYTHONHASHSEED'] = str(seed)
    #tf.random.set_seed(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)

    return None

def train_model(X_train, X_val, Y_train, Y_val, validation, C_value, epsilon_value):
    """
    Train function for a SVM model.
    param X_train: the training epigenetic inputs.
    param Y_train: the training RNAseq values.
    param X_val: the validation epigenetic inputs if appropriate.
    param Y_val: the validation RNAseq values if appropriate.
    return: Trained model and validation metrics if appropriate.
    """

    # Set random seed
    reset_random_seeds(9)
    
    #X_train = StandardScaler.fit_transform(X_train)
    #X_val = StandardScaler.fit_transform(X_val)
    #Y_train = StandardScaler.fit_transform(Y_train)
    #Y_val = StandardScaler.fit_transform(Y_val)
    
    #Code to reshape data into 2 dimensions.
    reshaped_X_train = X_train.reshape((X_train.shape[0], -1), order = 'F')
    reshaped_X_val = X_val.reshape((X_val.shape[0], -1), order = 'F')
    reshaped_Y_train = np.squeeze(Y_train)
    reshaped_Y_val = np.squeeze(Y_val)
    
    #mean_features_X_train = np.mean(X_train, axis = 1)
    #mean_features_X_val = np.mean(X_val, axis = 1)
    #reshaped_Y_train = np.squeeze(Y_train)
    #reshaped_Y_val = np.squeeze(Y_val)
    
    #regr = make_pipeline(StandardScaler(), SVR(kernel='linear', C=C_value, epsilon=epsilon_value))
    regr = SVR(kernel='linear', C=C_value, epsilon=epsilon_value)
    regr.fit(reshaped_X_train, reshaped_Y_train)
    if validation == True:
        Y_pred = regr.predict(reshaped_X_val)
        PCC = pearsonr(reshaped_Y_val, Y_pred)[0]
        SCC = spearmanr(reshaped_Y_val, Y_pred)[0]
        R2 = r2_score(reshaped_Y_val, Y_pred)
    
        return regr, PCC, SCC, R2
    
    else:
        
        return regr
    


def test_model(model, X_test, Y_test, C_value, epsilon_value):
    """
    Test/prediction function for a SVM model.
    param X_test: the testing epigenetic inputs.
    param Y_test: the testing RNAseq values
    return: Metric results on test set.
    """
    
    #Code to reshape data into 2 dimensions.
    reshaped_X_test = X_test.reshape((X_test.shape[0], -1), order = 'F')
    reshaped_Y_test = np.squeeze(Y_test)
    
    #mean_features_X_test = np.mean(X_test, axis = 1)
    #reshaped_Y_test = np.squeeze(Y_test)
    
    #regr = SVR(kernel='linear', C=C_value, epsilon=epsilon_value)
    #regr.fit(reshaped_X_test, reshaped_Y_test) 
    Y_pred = model.predict(reshaped_X_test)
    PCC = pearsonr(reshaped_Y_test, Y_pred)[0]
    SCC = spearmanr(reshaped_Y_test, Y_pred)[0]
    R2 = r2_score(reshaped_Y_test, Y_pred)
    
    return PCC, SCC, R2


def main(loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, count, C_value, epsilon_value):

    # Indicate True or False for the creation of a validation set. The script will fit the model accordingly.
    validation = False
    
    # Get file path from command line
    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]
    indices = sys.argv[3]
    
    # NOTE: file_path_1 is the data file for training and validating the model.
    #file_path_1 = "/gpfs/data/rsingh47/Tapinos_Data/Realigned_data_files_GSC1_Stem_with_featurecounts_RNAseq_entire_gene/raw/gsc1_stem_with_featurecounts_RNAseq_entire_gene.npy"
    #file_path_1 = "/gpfs/data/rsingh47/Tapinos_Data/Realigned_data_files_GSC2_Stem_with_featurecounts_RNAseq_entire_gene/raw/gsc2_stem_old_ATAC_process_with_featurecounts_RNAseq_entire_gene.npy"
    
    # NOTE file_path_2 is the datafile for testing the model.
    #file_path_2 = "/gpfs/data/rsingh47/Tapinos_Data/Realigned_data_files_GSC2_Stem_with_featurecounts_RNAseq_entire_gene/raw/gsc2_stem_old_ATAC_process_with_featurecounts_RNAseq_entire_gene.npy"
    #file_path_2 = "/gpfs/data/rsingh47/Tapinos_Data/Realigned_data_files_GSC1_Stem_with_featurecounts_RNAseq_entire_gene/raw/gsc1_stem_with_featurecounts_RNAseq_entire_gene.npy"
    
    #indices = "/gpfs/data/rsingh47/Tapinos_Data/Realigned_data_files/ind_shuffle.npy"
    
    # Call get_data() to process the data, preprocess = True will read in processed .npy files,
    # if false then will re-preprocess data
    print("Processing data")
    if count==0:
        preprocess_bool = True
    else:
        preprocess_bool = False
        
    if validation == True:
        validation_bool = True
    else:
        validation_bool = False


    # Processing data for patient 1 file to produce train and validation sets.
    X_train, X_val, Y_train, Y_val, gene_dict, num_genes = get_data_patient_1(file_path_1, indices, gene_dict, num_genes, preprocess = preprocess_bool, validation = validation_bool)

    # Processing data for patient 2 file to produce test set.
    X_test, Y_test, gene_dict, num_genes = get_data_patient_2(file_path_2, indices, gene_dict, num_genes, preprocess = preprocess_bool)

    # Call train_model() to train the model
    print("Training model...")
    
    if validation_bool == True:
        model, PCC, SCC, R2  = train_model(X_train, X_val, Y_train, Y_val, validation=validation_bool, C_value=C_value, epsilon_value=epsilon_value)

        max_val_pcc = PCC
        max_val_r2_score = R2
        max_val_scc = SCC
        

    else:
        model  = train_model(X_train, X_val, Y_train, Y_val, validation=validation_bool, C_value=C_value, epsilon_value=epsilon_value)
        max_val_pcc = 'TRAINING SET ONLY'
        max_val_r2_score = 'TRAINING SET ONLY'
        max_val_scc = 'TRAINING SET ONLY'


    val_pcc_dict[count] = max_val_pcc
    val_r2_score_dict[count] = max_val_r2_score
    val_scc_dict[count] = max_val_scc


    print("CURRENT COUNT:", count, "\n C value: ", C_value, "\n epsilon value: ", epsilon_value, "\n Val PCC: ", max_val_pcc , "\n Val SCC: ", max_val_scc, "\n Val R2: ", max_val_r2_score)    

    print('*'*25)
    print("Evaluating model...")
    print('*'*25)
    test_PCC, test_SCC, test_R2  = test_model(model, X_test, Y_test, C_value=C_value, epsilon_value=epsilon_value)
    print("Test results:")
    print(f"PCC,{test_PCC}")
    print(f"SCC,{test_SCC}")
    print(f"R2,{test_R2}")
    print('*'*25)



    now = datetime.datetime.now()
    # Script log file.
    with open('pngs_svm_cross_patient_pred_regression_gsc_stem_standard_log2/svm_cross_patient_regression_gsc_stem_standard_log2_info.csv', 'a') as log:
        log.write('\n' f'{now.strftime("%H:%M on %A %B %d")},')
     
        log.write(f'CURRENT COUNT: {count},C value: {C_value},epsilon value: {epsilon_value},')
        log.write(f'Val PCC: {max_val_pcc},Val SCC: {max_val_scc}, Val R2: {max_val_r2_score},')
        log.write(f'Test PCC: {test_PCC},Test SCC: {test_SCC}, Test R2: {test_R2}')


    
    return loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, X_train, X_val, X_test, Y_train, Y_val, Y_test, indices, model

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

    
    
    #epsilon_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01 , 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02 , 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03 , 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04 , 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05 , 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06 , 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07 , 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08 , 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09 , 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099]
    
    # Hyperparameter combination that produces the best validation metrics
    parameters = dict(C_values = [0.01], epsilon_values = [0.01])
    
    # GOOD
    #parameters = dict(C_values = [0.08], epsilon_values = [0.099])
    
    # GOOD
    #parameters = dict(C_values = [0.01], epsilon_values = [0.001])

    # Hyperparameter grid search
    ####parameters = dict(C_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], epsilon_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01 , 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02 , 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03 , 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04 , 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05 , 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06 , 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07 , 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08 , 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09 , 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099])

    
       
    param_values = [v for v in parameters.values()]

    count=0

    for Cv, ev in product(*param_values): 
        loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, X_train, X_val, X_test, Y_train, Y_val, Y_test, indices, model = main(loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, count, C_value = Cv, epsilon_value = ev)
        count+=1

    #min_loss_count = min(loss_dict, key=loss_dict.get)
    #max_pcc_count = max(pcc_dict, key=pcc_dict.get)
    #max_r2_count = max(r2_score_dict, key=r2_score_dict.get)
    #max_scc_count = max(scc_dict, key=scc_dict.get)
    #min_val_loss_count = min(val_loss_dict, key=val_loss_dict.get)
    max_val_pcc_count = max(val_pcc_dict, key=val_pcc_dict.get)
    max_val_r2_count = max(val_r2_score_dict, key=val_r2_score_dict.get)
    max_val_scc_count = max(val_scc_dict, key=val_scc_dict.get)
    

    #print("\n Min training loss and count: ", min(loss_dict.values()), min_loss_count, "\n Max training pcc and count: ", max(pcc_dict.values()), max_pcc_count, "\n Max training R2 and count: ", max(r2_score_dict.values()), max_r2_count, "\n Max training scc and count: ", max(scc_dict.values()), max_scc_count, "\n Min val loss and count: ", min(val_loss_dict.values()), min_val_loss_count, "\n Max val pcc and count: ", max(val_pcc_dict.values()), max_val_pcc_count, "\n Max val R2 and count: ", max(val_r2_score_dict.values()), max_val_r2_count, "\n Max val scc and count: ", max(val_scc_dict.values()), max_val_scc_count)
    