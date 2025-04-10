'''
This is a script to implement a 
multi-input Multi-layered Perceptron algorithm model.
It is designed to evaluate the model response to combining
the the same epigenetic features used in the other experimental
setups with portions of each gene's HG38 sequence.

Goal: Predict gene expression value (regression) from epigenetic 
signal inputs and gene sequences across two patients. 

The model takes two GSC (patient) data files as input. 
The first is used to create train and validation (if applicaple)
sets. The second, to create the test set. 
A third input is the reference genome (HG38) data.
 
This model is designed to accept 'raw' input values for the epigenetic features. 
It applies a log(2) transformation on the target variable before the 
cross-validation dataset split step.

The model applies standardization to the train, validation
(if applicaple) and test datasets seperately after the dataset splits.  

The Reference Human Genome (HG38) data file is derived by first converting 
the reference genome to a numerical representation where each nucleotide 
is given a representative number. Within this script's training and testing 
processed, the numerical representations are one-hot encoded for model input.
'''

import sys
import random
import numpy as np
import os
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
import tensorflow as tf
import tensorflow_addons as tfa
# Method used to troubleshoot issues between SHAP and Tensorflow.
#tf.compat.v1.disable_v2_behavior()
from matplotlib import pyplot as plt
import seaborn as sn
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, LeakyReLU, Concatenate
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn import preprocessing
from tqdm import tqdm
from tensorflow.keras import backend as K
#from scipy.stats import pearsonr
from scipy.stats import spearmanr
from itertools import product
import datetime
import math
import shap
import pandas as pd

def get_data_patient_1(file_path_1, file_path_3, indices, gene_dict, num_genes, preprocess, validation):
    '''
    Returns X_train, X_val, Y_train, Y_val; where X refers to inputs and Y refers to labels.
    '''

    if preprocess:
        
        print("Loading patient 1 epigenetic and sequence data...")
        ### EPIGENETIC DATA
        # Col 1 = gene names, 2 = bin number, 3-6 = features, 7 = labels
        combined_diff = np.load(file_path_1, allow_pickle = True)
        #combined_diff = combined_diff[:51200] # TEMPORARILY limit data for faster testing.

        ### SEQUENCE DATA
        # Col 1 = gene names, 2 = bin number, 3-102 = features
        loaded_sequence = np.load(file_path_3, allow_pickle = True)
        #loaded_sequence = loaded_sequence[:51200] # TEMPORARILY limit data for faster testing.
        
        # Get all the unique gene names
        gene_names = np.unique(combined_diff[:, 0])
        num_genes = len(gene_names)
        
        # Create a dictionary to map each gene name to a unique index (number like 0, 1, 2,...,#genes-1)
        gene_dict = dict(zip(gene_names, range(num_genes)))

        # Get the number of features (last column is labels - RNAseq)
        num_features = 4
        num_epigenetic_bins = 50
        num_sequence_bins = 21
        num_sequence_features = 100
        num_sequence_encoding_positions = 5

        # Epigenetic inputs data shape
        # X = np.zeros((num_genes, num_features, num_bins))
        X = np.zeros((num_genes, num_epigenetic_bins, num_features))
        # Sequence inputs data shape
        X_S = np.zeros((num_genes, num_sequence_bins, num_sequence_features))
        #X_S = np.zeros((num_genes, num_bins, num_sequence_features, num_sequence_encoding_positions))
        # Labels data shape
        Y = np.zeros((num_genes, 1))

        for name in tqdm(gene_names):
            # Each epigenetic subset is of shape 50 x 6 (number of 100bp bins x number of columns)
            subset = combined_diff[np.where(combined_diff[:, 0] == name)]
            # Each sequence subset is of shape 50 x 100 (number of 100bp bins x number of columns)
            subset_S = loaded_sequence[np.where(loaded_sequence[:, 0] == name)]

            # Create matrix of data
            gene_ind = gene_dict[name]
            data = subset[:, 2:]
            data_S = subset_S[:, 2:]

            # data_inputs = np.transpose(data[:, :-1])
            data_inputs = data[:, :-1]
            ### Here the number of bins around the TSS site could be controlled
            data_inputs_S = data_S[15:36,:]

            # Add to arrays at the unique id position
            X[gene_ind] = data_inputs
            X_S[gene_ind] = data_inputs_S

            # Set corresponding value to be first bin's RNAseq value (since all 100 bins)
            # have the same value
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
        #ind = np.arange(0, num_genes)
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
            
            X_Seq_train = X_S[train_ind]
            X_Seq_val = X_S[val_ind]
            
            # List of all datasets after split operation.
            # Standardization ONLY on input variables.
            datasets = [X_train, X_val]
        
        else:
            # TESTING SPLITS
            # The training set will have 100% of 
            # the patient 1 data to train the model.
            # train_ind = ind
            train_ind = patient1_ind
            X_train = X[train_ind]
            Y_train = Y[train_ind]
            X_Seq_train = X_S[train_ind]
            datasets = [X_train] 


        # Perform calculation on each column of the seperate train, validation and test sets. 
        for dataset in datasets:
            for i in range(dataset.shape[2]):
                # Standardize the column values.
                dataset[:, :, i] = (dataset[:, :, i] - np.mean(dataset[:, :, i])) / np.std(dataset[:, :, i], ddof = 1) # The degrees of freedom is set t
        
        # Save epigenetic data.
        np.save("X_cross_patient_regression_patient_1_stem_standard_log2_train", 
                X_train, 
                allow_pickle = True)
        np.save("Y_cross_patient_regression_patient_1_stem_standard_log2_train", 
                Y_train, 
                allow_pickle = True)
        
        # Save sequence data.
        np.save("X_Sequence_cross_patient_regression_patient_1_stem_standard_log2_train", 
                X_Seq_train, 
                allow_pickle = True)
        
    if validation == True:
            np.save("X_cross_patient_regression_patient_1_stem_standard_log2_val", 
                X_val, 
                allow_pickle = True)
            np.save("Y_cross_patient_regression_patient_1_stem_standard_log2_val", 
                Y_val, 
                allow_pickle = True)
            
            # Save sequence data.
            np.save("X_Sequence_cross_patient_regression_patient_1_stem_standard_log2_val", 
                X_Seq_val, 
                allow_pickle = True)


    
    else:
        # Load epigenetic data.
        X_train = np.load("X_cross_patient_regression_patient_1_stem_standard_log2_train.npy", 
                          allow_pickle = True)
        Y_train = np.load("Y_cross_patient_regression_patient_1_stem_standard_log2_train.npy", 
                          allow_pickle = True)
        
        # Load sequence data.
        X_Seq_train = np.load("X_Sequence_cross_patient_regression_patient_1_stem_standard_log2_train.npy", 
                              allow_pickle = True)
        
        if validation == True:
            X_val = np.load("X_cross_patient_regression_patient_1_stem_standard_log2_val.npy", 
                            allow_pickle = True)
            Y_val = np.load("Y_cross_patient_regression_patient_1_stem_standard_log2_val.npy", 
                            allow_pickle = True)
            X_Seq_val = np.load("X_Sequence_cross_patient_regression_patient_1_stem_standard_log2_val.npy", allow_pickle = True)

    
        gene_dict = gene_dict
        num_genes = num_genes


    if validation == True:
        return X_train, X_val, Y_train, Y_val, gene_dict, num_genes, X_Seq_train, X_Seq_val
    else:
        return X_train, Y_train, gene_dict, num_genes, X_Seq_train

def get_data_patient_2(file_path_2, file_path_3, 
                       indices, gene_dict, num_genes, preprocess):
    '''
    Returns X_test, Y_test; where X refers to inputs and Y refers to labels.
    '''

    if preprocess:
        
        print("Loading patient 2 epigenetic and sequence data...")
        ### EPIGENETIC DATA
        # Col 1 = gene names, 2 = bin number, 3-6 = features, 7 = labels
        combined_diff = np.load(file_path_2, allow_pickle = True)
        #combined_diff = combined_diff[:51200] # TEMPORARILY limit data for faster testing.

        ### SEQUENCE DATA
        # Col 1 = gene names, 2 = bin number, 3-102 = features
        loaded_sequence = np.load(file_path_3, allow_pickle = True)
        #loaded_sequence = loaded_sequence[:51200] # TEMPORARILY limit data for faster testing.
        
        # Get all the unique gene names
        gene_names = np.unique(combined_diff[:, 0])
        num_genes = len(gene_names)
        
        # Create a dictionary to map each gene name to a unique index (number like 0, 1, 2,...,#genes-1)
        gene_dict = dict(zip(gene_names, range(num_genes)))

        # Get the number of features (last column is labels - RNAseq)
        num_features = 4
        num_epigenetic_bins = 50
        num_sequence_bins = 21
        num_sequence_features = 100
        num_sequence_encoding_positions = 5

        # Epigenetic inputs data shape
        # X = np.zeros((num_genes, num_features, num_bins))
        X = np.zeros((num_genes, num_epigenetic_bins, num_features))
        # Sequence inputs data shape
        X_S = np.zeros((num_genes, num_sequence_bins, num_sequence_features))
        #X_S = np.zeros((num_genes, num_bins, num_sequence_features, num_sequence_encoding_positions))
        # Labels data shape
        Y = np.zeros((num_genes, 1))

        for name in tqdm(gene_names):
            # Each epigenetic subset is of shape 50 x 6 (number of 100bp bins x number of columns)
            subset = combined_diff[np.where(combined_diff[:, 0] == name)]
            # Each sequence subset is of shape 50 x 100 (number of 100bp bins x number of columns)
            subset_S = loaded_sequence[np.where(loaded_sequence[:, 0] == name)]

            # Create matrix of data
            gene_ind = gene_dict[name]
            data = subset[:, 2:]
            data_S = subset_S[:, 2:]

            # data_inputs = np.transpose(data[:, :-1])
            data_inputs = data[:, :-1]
            ### Here the number of bins around the TSS site could be controlled
            data_inputs_S = data_S[15:36,:]

            # Add to arrays at the unique id position
            X[gene_ind] = data_inputs
            X_S[gene_ind] = data_inputs_S

            # Set corresponding value to be first bin's RNAseq value (since all 100 bins)
            # have the same value
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
        ind = np.load(indices, allow_pickle = True)
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

        # Splits for this patient data can be adjusted here.
        #train_ind = ind[0: int(0.7*num_genes)]
        #val_ind = ind[int(0.7*num_genes):int(0.85*num_genes)]
        #test_ind = ind[int(0.85*num_genes):]


        # NOTE: For now use entire dataset for test set.
        test_ind = patient2_ind

        # Use all of the dataset for test.
        X_test = X[test_ind]

        # Use all of the dataset for test.
        Y_test = Y[test_ind]

        X_Seq_test = X_S[test_ind]
        
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
                # Standardize the column values.
                dataset[:, :, i] = (dataset[:, :, i] - np.mean(dataset[:, :, i])) / np.std(dataset[:, :, i], ddof = 1)

        # Save epigenetic data.
        np.save("X_cross_patient_regression_patient_2_stem_standard_log2_test", 
                X_test, 
                allow_pickle = True)
        np.save("Y_cross_patient_regression_patient_2_stem_standard_log2_test", 
                Y_test,
                allow_pickle = True)

        # Save sequence data.
        np.save("X_Sequence_cross_patient_regression_patient_1_stem_standard_log2_test", 
                X_Seq_test,
                allow_pickle = True)
  

    else:
        # Load epigenetic data.
        X_test = np.load("X_cross_patient_regression_patient_2_stem_standard_log2_test.npy", 
                         allow_pickle = True)
        Y_test = np.load("Y_cross_patient_regression_patient_2_stem_standard_log2_test.npy", 
                         allow_pickle = True)
        
        # Load sequence data.
        X_Seq_test = np.load("X_Sequence_cross_patient_regression_patient_1_stem_standard_log2_test.npy",
                             allow_pickle = True)
        
        gene_dict = gene_dict
        num_genes = num_genes

    return X_test, Y_test, gene_dict, num_genes, X_Seq_test, patient2_ind

def reset_random_seeds(seed):
    '''
    Takes a given number and assigns it
    as a random seed to various generators and the
    os environment.
    '''

    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return None

def train_model(X_train, Y_train, X_Seq_train, epochs, validation, batch_size, learning_rate, epigenetic_branch_hidden_size_1, epigenetic_branch_hidden_size_2, epigenetic_branch_hidden_size_3, epigenetic_branch_dropout_rate, sequence_branch_hidden_size_1, sequence_branch_hidden_size_2, sequence_branch_hidden_size_3, sequence_branch_dropout_rate, combined_hidden_size_1, combined_hidden_size_2, combined_hidden_size_3, combined_dropout_rate, random_state, X_val, Y_val, X_Seq_val):
    """
    Implements and trains the model using a multiple MLP branches.
    return: a trained model and training history
    """


    metrics = [pearson_r, 
               spearman_r,
               tfa.metrics.r_square.RSquare()
    ]

    
    epigenetic_branch_dropout_rate = epigenetic_branch_dropout_rate
    sequence_branch_dropout_rate = sequence_branch_dropout_rate
    combined_dropout_rate = combined_dropout_rate
    

    X_Seq_train_one_hot = tf.one_hot(X_Seq_train, depth = 5, axis = -1)
    if validation == True:
        X_Seq_val_one_hot = tf.one_hot(X_Seq_val, depth = 5, axis = -1)


   
    # Model composed with tensorflow's functional API
    
    # Branch for epigenetic features.
    epigenetic_input = Input(shape=X_train.shape[1:], name = 'epigenetic_features_input')
    epigenetic_flatten = Flatten(name = 'flatten_epigenetic_features')(epigenetic_input)
    epigenetic = Dense(epigenetic_branch_hidden_size_1, activation='relu', name = 'epigenetic_dense_layer_1')(epigenetic_flatten)
    epigenetic = Dropout(epigenetic_branch_dropout_rate, name = 'epigenetic_dropout_layer_1')(epigenetic)  
    epigenetic = Dense(epigenetic_branch_hidden_size_2, activation='relu', name = 'epigenetic_dense_layer_2')(epigenetic)
    epigenetic = Dropout(epigenetic_branch_dropout_rate, name = 'epigenetic_dropout_layer_2')(epigenetic)
    epigenetic = Dense(epigenetic_branch_hidden_size_3, activation='relu', name = 'epigenetic_dense_layer_3')(epigenetic)

    ####epigenetic_output = Dense(1,)(epigenetic)
    
    # Branch for sequence features.
    sequence_input = Input(shape=X_Seq_train_one_hot.shape[1:], name = 'reference_genome_sequence_features_input')
    sequence_flatten = Flatten(name = 'flatten_sequence_features')(sequence_input)
    sequence = Dense(sequence_branch_hidden_size_1, activation='relu', name = 'sequence_dense_layer_1')(sequence_flatten)
    sequence = Dropout(sequence_branch_dropout_rate, name = 'sequence_dropout_layer_1')(sequence)  
    sequence = Dense(sequence_branch_hidden_size_2, activation='relu', name = 'sequence_dense_layer_2')(sequence)
    sequence = Dropout(sequence_branch_dropout_rate, name = 'sequence_dropout_layer_2')(sequence)
    sequence = Dense(sequence_branch_hidden_size_3, activation='relu', name = 'sequence_dense_layer_3')(sequence)
        
    # Combined branches to output.
    combined = Concatenate(axis = 1)([epigenetic, sequence])
    combined = Dropout(combined_dropout_rate, name = 'combined_branch_output_dropout_layer_1')(combined)
    combined = Dense(combined_hidden_size_1, activation = 'relu', name = 'combined_branch_output_dense_layer_1')(combined)
    combined = Dropout(combined_dropout_rate, name = 'combined_branch_output_dropout_layer_2')(combined)
    combined = Dense(combined_hidden_size_2, activation = 'relu', name = 'combined_branch_output_dense_layer_2')(combined)
    combined = Dropout(combined_dropout_rate, name = 'combined_branch_output_dropout_layer_3')(combined)
    combined = Dense(combined_hidden_size_3, activation = 'relu', name = 'combined_branch_output_dense_layer_3')(combined)
    #### Single output with no activation for regression.
    output = Dense(1, name = 'model_prediction_output')(combined)    


    
    #model = Model(inputs = [epigenetic_input], outputs = epigenetic_output)
    model = Model(inputs = [epigenetic_input, sequence_input], outputs = output)

    
    # Plot model graph
    plot_model(model, to_file = save_directory + '/mlp_cross_patient_pred_branched_regression_model.png', 
               show_shapes = True, 
               dpi = 600)
    
    # Set random seed
    reset_random_seeds(random_state)
    
    # Compile model
    model.compile(loss = 'mean_squared_error', optimizer = Adam(learning_rate), metrics = metrics)
    
    
    if validation == True:
        history = model.fit({'epigenetic_features_input': X_train, 'reference_genome_sequence_features_input': X_Seq_train_one_hot}, Y_train, validation_data = ({'epigenetic_features_input': X_val, 'reference_genome_sequence_features_input': X_Seq_val_one_hot}, Y_val), epochs = epochs, batch_size = batch_size, shuffle = False)
        #history = model.fit({'epigenetic_features_input': X_train}, Y_train, validation_data=({'epigenetic_features_input' : X_val}, Y_val), epochs=epochs, batch_size=batch_size, shuffle=False)
    else:
        history = model.fit({'epigenetic_features_input': X_train, 'reference_genome_sequence_features_input': X_Seq_train_one_hot}, Y_train, epochs = epochs, batch_size = batch_size, shuffle = False)
        #history = model.fit({'epigenetic_features_input': X_train}, Y_train, epochs=epochs, batch_size=batch_size, shuffle=False)

    #history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle=False)

    return model, history

def test_model(model, X_test, Y_test, X_Seq_test_one_hot, batch_size):
    
    results  = model.evaluate({'epigenetic_features_input': X_test, 'reference_genome_sequence_features_input': X_Seq_test_one_hot}, Y_test, batch_size = batch_size)
    
    return results



def pearson_r(y_true, y_pred):
    '''
    Calculate Pearson Correlation Coefficient (PCC) as a metric for the model.
    '''
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    
    return K.mean(r)

def spearman_r(y_true, y_pred):
    '''
    Calculate Spearman Correlation Rank 
    Coefficient (SCRC) as a metric for the model.
    '''
    spearman_value = tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout = tf.float32)
    return spearman_value

def calculate_se_for_predictions(y_true, y_pred):
    '''
    Calculates the Squared Error on the test set predictions 
    after the model is trained.
    param y_true: the true (observed) RNAseq values.
    param y_pred: the model's predicted values.

    returns: Array of SE values the same length as Y_test.
    '''

    prediction_se = np.round_(np.square(y_true - y_pred), decimals = 6)

    return prediction_se

def calculate_prediction_high_low(mse, y_true, y_pred):
    '''
    Determines the index positions for genes with the "best" and "worst"
    predictions. The 25% with the lowest MSE and the 25% with the highest.
    The mse values should be originally in the order of their index positions
    in the test set. They are sorted into ascending order here. A second sort is then performed to 
    seperate the genes into groups where the true RNAseq values indicate
    they are "highly" or "lowly" expressed.

    returns: Two arrays of index positions for genes in test set. 
    '''

    # collect together the mse value and its index position in the test set.
    #mse_with_idx = [[v[0].astype("float32"), i] for i, v in enumerate(y_pred)]
    #mse_with_idx = [[v, i] for i, v in enumerate(mse)]
    idx = np.arange(mse.shape[0], dtype = int).tolist()
    # Stack the y_true and y_pred with the MSE and index.
    #stacked_arrays = np.hstack((mse, idx , np.squeeze(y_true), np.squeeze(y_pred)))
    mse = np.squeeze(mse).tolist()
    y_true = np.squeeze(y_true).tolist()
    y_pred = np.squeeze(y_pred).tolist()
    
    stacked_lists = [i for i in zip(mse, idx, y_true, y_pred)]
    # Sort the list of lists by the mse in ascending order
    sorted_by_mse = sorted(stacked_lists, key = lambda x: x[0])

    # Create list of low mse indexes
    #low_mse_group = [sorted_by_mse[i][1] for i in range(int(0.25*len(sorted_mse_with_idx)))]
    low_mse_group = [sorted_by_mse[i] for i in range(int(0.25*len(sorted_by_mse)))]
    # Create list of high mse indexes
    #high_mse_group = [sorted_by_mse[i][1] for i in range(int(0.75*len(sorted_mse_with_idx)), len(sorted_mse_with_idx))] 
    high_mse_group = [sorted_by_mse[i] for i in range(int(0.75*len(sorted_by_mse)), len(sorted_by_mse))]

    # Sort the smaller groups by their true expression value.
    sorted_low_mse_group = sorted(low_mse_group, key = lambda x: x[2])
    sorted_high_mse_group = sorted(high_mse_group, key = lambda x: x[2])

    # Create lists for 'expressed' and 'not expressed' within each group.
    # Calculate the 25% of the highest y_true values and count them as 'highly expressed'.
    # Calculate the 25% of the lowest y_true values and count them as 'lowly expressed'.
    highly_expressed_low_mse_group = [sorted_low_mse_group[i][1] for i in range(int(0.75*len(sorted_low_mse_group)), len(sorted_low_mse_group))]
    lowly_expessed_low_mse_group = [sorted_low_mse_group[i][1] for i in range(int(0.25*len(sorted_low_mse_group)))]
    highly_expressed_high_mse_group = [sorted_high_mse_group[i][1] for i in range(int(0.75*len(sorted_high_mse_group)), len(sorted_high_mse_group))]
    lowly_expressed_high_mse_group = [sorted_high_mse_group[i][1] for i in range(int(0.25*len(sorted_high_mse_group)))]

    #return low_mse_group, high_mse_group, sorted_mse_with_idx
    return highly_expressed_low_mse_group, lowly_expessed_low_mse_group, highly_expressed_high_mse_group, lowly_expressed_high_mse_group
    #return highly_expressed_low_mse_group
    

def get_gene_names(gene_dict, indices, test_data_shape, 
                   num_genes, shuffle_index):
    '''
    Using the input dictionary of gene names and their unique identifier
    this function extracts the genes and their index position in X_test data
    for future information for visualization. This position is after the
    shuffle operation and cross_validation split.

    returns: Returns a list of the gene names
    '''
    ### Discontinue using the loaded index file because it 
    ### is setup for 20,0015 genes and does not accomidate
    ### the smaller set of genes in avaliable in the later
    ### datasets used for cross-patient testing.
    # Load indices file used for shuffle operation.
    # shuffle_index = np.load(indices, allow_pickle=True)
    
    #shuffle_index = np.arange(0, 20015) 
    # Invert order of keys and values for gene dictionary.
    inverted_gene_dict = {v:k for k, v in gene_dict.items()}

    # Using entire second dataset for test set
    test_ind = shuffle_index

    gene_names_in_test_set = [inverted_gene_dict[i] for i in test_ind]
    
    return gene_names_in_test_set


def visualize_loss(history, validation, count):
    '''
    Visualization of model loss over all epochs.
    returns: Nothing. A visualization will be saved.
    '''
    plt.close()
    plt.plot(history.history['loss'], color = 'orchid')
    if validation:
        plt.plot(history.history['val_loss'], color = 'wheat')
    plt.title("Loss - Branched MLP epigenetics and sequence (standardized input/log2 target)")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if validation:
        plt.legend(['train loss', 'validation loss'], loc='upper left')
    else:
        plt.legend(['train loss'], loc='upper left')
    plt.savefig(save_directory + '/mlp_loss_' + str(count) + '.png')


    return None

def visualize_pcc(history, validation, count):
    '''
    Visualization of the Pearson Correlation Coefficient (PCC)
    values over all epochs.
    returns: Nothing. A visualization will appear and/or be saved.
    '''
    plt.close()
    plt.plot(history.history['pearson_r'], color = 'springgreen')
    if validation:
        plt.plot(history.history['val_pearson_r'], color = 'gray')
    #plt.title(name)
    plt.title("PCC - Branched MLP epigenetics and sequence (standardized input/log2 target)")
    plt.ylabel('PCC')
    plt.xlabel('epoch')
    if validation:
        plt.legend(['train pcc', 'validation pcc'], loc='upper left')
    else:
        plt.legend(['train pcc'], loc='upper left')
    plt.savefig(save_directory + '/mlp_pcc_' + str(count) + '.png')


    return None

def visualize_scc(history, validation, count):
    '''
    Visualization of the Spearman Correlation Rank Coefficient (SCC)
    values over all epochs.
    returns: Nothing. A visualization will be saved.
    '''
    plt.close()
    plt.plot(history.history['spearman_r'], color = 'palegoldenrod')
    if validation:
        plt.plot(history.history['val_spearman_r'], color = 'crimson')
    plt.title("SCC - Branched MLP epigenetics and sequence (standardized input/log2 target)")
    plt.ylabel('SCC')
    plt.xlabel('epoch')
    if validation:
        plt.legend(['train scc', 'validation scc'], loc='upper left')
    else:
        plt.legend(['train scc'], loc='upper left')
    plt.savefig(save_directory + '/mlp_scc_' + str(count) + '.png')


    return None

def visualize_rsquare(history, validation, count):
    '''
    Visualize the RSquare metric accross all epochs.

    returns: Nothing. The visualization will be saved.
    '''
    plt.close()
    plt.plot(history.history['r_square'], color = 'yellow')
    if validation:
        plt.plot(history.history['val_r_square'], color = 'red')
    plt.title("R2 - Branched MLP epigenetics and sequence (standardized input/log2 target)")
    plt.ylabel('R2')
    plt.xlabel('epoch')
    if validation:
        plt.legend(['train R2', 'validation R2'], loc = 'upper left')
    else:
        plt.legend(['train R2'], loc = 'upper left')
    plt.savefig(save_directory + '/mlp_rsquare_' + str(count) + '.png')


    return None

def visualize_metrics_together(history, validation, count):
    '''
    Combined visualization of the metrics
    values over all epochs.
    returns: Nothing. A visualization will be saved.
    '''
    plt.close()
    plt.plot(history.history['spearman_r'], color = 'tomato')
    plt.plot(history.history['pearson_r'], color = 'green')
    plt.plot(history.history['r_square'], color = 'cornflowerblue')
    if validation:
        plt.plot(history.history['val_spearman_r'], color = 'whitesmoke')
        plt.plot(history.history['val_pearson_r'], color = 'gainsboro')
        plt.plot(history.history['val_r_square'], color = 'darkgrey')
    plt.title("Branched MLP epigenetics and sequence (standardized input/log2 target)")
    plt.ylabel('Metric value')
    plt.xlabel('epoch')
    if validation:
        plt.legend(['train scc', 'train pcc', 'train R^2', 'validation scc', 'validation pcc', 'validation R^2'], loc='lower right')
    else:
        plt.legend(['train scc', 'train pcc', 'train R^2'], loc='upper left')
    plt.savefig(save_directory + '/mlp_metrics_' + str(count) + '.png')


    return None


def visualize_training_distributions(y_train):
    '''
    Creates multiple visualizations for the for the 
    training set. 
    param y_train: the true (observed) RNAseq values
                   for the test set.

    return: Nothing. Visualizations are saved to
    the script save folder.
    '''

    # Build dataframe for visualization.
    #train_and_val_rnaseq = pd.DataFrame({'training set' : y_train.tolist(),
                                        #'validation set' : y_val.tolist()})
    
    plt.close()
    sn.set_theme(style = 'whitegrid')
    plt.title('Training set genes\' RNAseq value counts.')
    plt.ylabel('count')
    plt.xlabel('RNAseq value after log(2) transformation.')
    sn.histplot(y_train, 
                legend = False, 
                palette= ['red'], 
                bins = 50)
    plt.savefig(save_directory + '/Training_set_genes_RNAseq_value_counts-_histogram_plot.png',
               format = 'png', 
                dpi = 600, 
                bbox_inches = 'tight')


    training_RNAseq_dataframe = pd.Series(np.squeeze(y_train))

    
    # Define gene expression catagories for analysis.

    training_all_zero_true_expression = training_RNAseq_dataframe[training_RNAseq_dataframe  == 0]
    training_true_expression_between_0_and_5 = training_RNAseq_dataframe[(training_RNAseq_dataframe > 0) & (training_RNAseq_dataframe < 5)]
    training_true_expression_between_5_and_10 = training_RNAseq_dataframe[(training_RNAseq_dataframe >= 5) & (training_RNAseq_dataframe < 10)]
    training_true_expression_between_10_and_15 = training_RNAseq_dataframe[(training_RNAseq_dataframe >= 10) & (training_RNAseq_dataframe <= 15)]
    
    training_expression_counts_dataframe = pd.DataFrame({'expression catagory after log(2) transformation' : ['all zero', 'between 0_and 5', 
                                                                    'between 5 and 10', 'between 10 and 15' ],
                                            'count': [len(training_all_zero_true_expression), len(training_true_expression_between_0_and_5), 
                                                     len(training_true_expression_between_5_and_10), len(training_true_expression_between_10_and_15)]})

    
    dataframes = [training_expression_counts_dataframe]
    dataset_names = ['Training Set']
    
        
    # Visualize number of genes in each catagory.
    for l in range(len(dataframes)):
        plt.close()
        sn.set_theme(style = 'whitegrid')
        fig, ax = plt.subplots(figsize = (8, 5))
        ax = sn.barplot(data = dataframes[l], x = 'expression catagory after log(2) transformation', y = 'count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation = "45")
        ax.set(title = f'{dataset_names[l]} - MLP Cross Patient Regression Expression Catagory Counts')
        sn.set(font_scale = 1)
        for i in ax.containers:
            ax.bar_label(i,)
        plt.savefig(save_directory + '/' + dataset_names[l] + 'Expression_Catagory_Counts.png', 
                    format = 'png', 
                    dpi = 600, 
                    bbox_inches='tight')

    return None

def visualize_validation_distributions(y_val):
    '''
    Creates multiple visualizations for the for the 
    training and validation sets. 
    param y_train: the true (observed) RNAseq values
                   for the test set.
    param y_val: the true (observed) RNAseq values 
                 for the validation set.

    return: Nothing. Visualizations are saved to
    the script save folder.
    '''
    
    plt.close()
    sn.set_theme(style = 'whitegrid')
    plt.title('Validation set genes\' RNAseq value counts.')
    plt.ylabel('count')
    plt.xlabel('RNAseq value after log(2) transformation.')
    sn.histplot(y_val, 
                legend = False, 
                palette = ['darkblue'], 
                bins = 50)
    plt.savefig(save_directory + '/Validation_set_genes_RNAseq_value_counts-_histogram_plot.png', 
                format = 'png', 
                dpi = 600, 
                bbox_inches = 'tight')

    validation_RNAseq_dataframe = pd.Series(np.squeeze(y_val))
    
    # Define gene expression catagories for analysis.
    validation_all_zero_true_expression = validation_RNAseq_dataframe[validation_RNAseq_dataframe  == 0]
    validation_true_expression_between_0_and_5 = validation_RNAseq_dataframe[(validation_RNAseq_dataframe > 0) & (validation_RNAseq_dataframe < 5)]
    validation_true_expression_between_5_and_10 = validation_RNAseq_dataframe[(validation_RNAseq_dataframe >= 5) & (validation_RNAseq_dataframe < 10)]
    validation_true_expression_between_10_and_15 = validation_RNAseq_dataframe[(validation_RNAseq_dataframe >= 10) & (validation_RNAseq_dataframe <= 15)]

    validation_expression_counts_dataframe = pd.DataFrame({'expression catagory after log(2) transformation' : ['all zero', 'between 0 and 5', 
                                                                    'between 5 and 10', 'between 10 and 15'],
                                                "count": [len(validation_all_zero_true_expression), len(validation_true_expression_between_0_and_5), 
                                                     len(validation_true_expression_between_5_and_10), len(validation_true_expression_between_10_and_15)]})

    dataframes = [validation_expression_counts_dataframe]
    dataset_names = ['Validation Set']
    
        
    # Visualize number of genes in each catagory.
    for l in range(len(dataframes)):
        plt.close()
        sn.set_theme(style = 'whitegrid')
        fig, ax = plt.subplots(figsize = (8, 5))
        ax = sn.barplot(data = dataframes[l], 
                        x = 'expression catagory after log(2) transformation', 
                        y = 'count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation = "45")
        ax.set(title = f'{dataset_names[l]} - MLP Cross Patient Regression Expression Catagory Counts')
        sn.set(font_scale=1)
        for i in ax.containers:
            ax.bar_label(i,)
        plt.savefig(save_directory + '/' + dataset_names[l] + 'Expression_Catagory_Counts.png', 
                    format = 'png', 
                    dpi = 600, 
                    bbox_inches='tight')

    return None


def visualize_prediction_mse(prediction_mses, y_true, y_pred):
    '''
    Creates multiple visualizations for the 
    mean squared error values for the test set.
    Visualizations are saved to the script output folder.

    return: Nothing.  
    '''
    plt.close()
    sn.set_theme(style = "whitegrid")
    plt.title('Mean Squared Error Values for Test Set Genes')
    plt.ylabel('MSE')
    plt.xlabel('Gene index in test set')
    plt.scatter(np.arange(prediction_mses.shape[0]), prediction_mses)
    plt.savefig(save_directory + '/MLP_regression_test_set_mse_values.png', bbox_inches = 'tight')


    plt.close()
    sn.set_theme(style = "whitegrid")
    plt.title('Mean Squared Error Values for the Test Set Genes')
    plt.ylabel('count')
    plt.xlabel('Mean Squared Error')
    sn.histplot(prediction_mses, legend = False, color = 'red', bins = 50)
    plt.savefig(save_directory + '/MLP_regression_test_set_mse_values_-_histogram_plot.png', bbox_inches = 'tight')



    plt.close()
    sn.set_theme(style = "whitegrid")
    plt.title('True RNAseq Values for the Test Set Genes')
    plt.ylabel('count')
    plt.xlabel('True values')
    sn.histplot(y_true, legend = False, color = 'blue', bins = 50)
    plt.savefig(save_directory + '/MLP_regression_test_set_true_RNAseq_values_-_histogram_plot.png', bbox_inches = 'tight')


    plt.close()
    sn.set_theme(style = "whitegrid")
    plt.title('Predicted RNAseq Values for the Test Set Genes')
    plt.ylabel('count')
    plt.xlabel('Predicted values')
    sn.histplot(y_pred, legend = False, color = 'red', bins = 50)
    plt.savefig(save_directory + '/MLP_regression_test_set_predicted_RNAseq_values_-_histogram_plot.png', bbox_inches = 'tight')

    return None


def visualize_prediction(highly_expressed_low_mse_group, lowly_expressed_low_mse_group, highly_expressed_high_mse_group, lowly_expressed_high_mse_group, mse_value, test_true_values, test_prediction, gene_names):
    '''
    This function provides insight into the model's prediction on test set data in
    both graphical and text forms. It includes two helper functions.

    
    param test_labels: The Y labels for the test set.
    param test_predictions: The returned predictions from make_prediction function.
    param gene_names: The dictionary produced in the get_data function that collects the
                      gene_name and its unique identifier.
    returns: Indices of correct and incorrect predictions.
    Two visualization should appear and be saved.
    '''

    # Create csv file to hold prediction information.
    #with open(save_directory + '/mlp_tuning_regression_baseline_gsc_stem_standard_test_predictions.csv', 'w') as log:
    #        log.write(f'gene_name, gene_index_in_test_set, MSE for prediction, true_RNAseq_value, predicted_RNAseq_value')


    #def prediction_csv(gene_indices, mse_value, true_RNAseq_value, predicted_RNAseq_value, gene_names):
    #    '''
    #    Helper function to output prediction information to csv file created above.
    #    '''
    #    for i in range(len(gene_indices)):
    #        index = gene_indices[i]
    #        gn = gene_names[index]
    #        mse = mse_value[i]
    #        tv = true_RNAseq_value[index]
    #        pv = predicted_RNAseq_value[index]
    #        with open(save_directory + '/mlp_tuning_regression_baseline_gsc_stem_standard_test_predictions.csv', 'a') as log:
    #            log.write('\n' f'{gn}, {index}, {mse}, {tv}, {pv}')


    def prediction_plot(gene_indices, chart_label, mse_value, true_RNAseq_value, predicted_RNAseq_value, gene_names):
        '''
        Plots a smaller portion of the full prediction information.
        '''
        number_columns = 3
        number_rows = math.ceil(len(gene_indices) / number_columns)
        plt.close()
        #fig = plt.figure(figsize = (10, 4))
        fig = plt.figure()
        for i in range(len(gene_indices)):
            index = gene_indices[i]
            ax = fig.add_subplot(number_rows , number_columns, i+1)
            ax.axis('off')
            gn = gene_names[index]
            mse = mse_value[index]
            tv = np.squeeze(true_RNAseq_value[index])
            pv = np.squeeze(predicted_RNAseq_value[index])
            #pl = first_label if test_prediction[index] == 0 else second_label
            #al = first_label if test_labels[index] == 0 else second_label
            #prob = sigmoid_array[index]
            ax.set(title = "{}\n{}\nTV: {}\nPV: {}\nGI: {}".format(gn, mse, tv, pv, index))
            #ax.set(title = "PL: {}, AL: {}, GI: {}, Prob: {} ".format(pl, al, index, prob))
            plt.setp(ax.get_xticklabels(), visible = False)
            plt.setp(ax.get_yticklabels(), visible = False)
            ax.tick_params(axis = 'both', which = 'both', length = 0)
        fig.suptitle("{} genes from test set\nTV = True RNAseq Value\nPV = Predicted RNAseq Value, GI = Gene index position in test set".format(chart_label))
        fig.tight_layout()
        plt.savefig(save_directory + '/MLP_regression_Prediction_' + chart_label + '.png')



    #gene_indices = [i[1] for i in sorted_mse_with_idx]
    #mse_value = [i[0] for i in sorted_mse_with_idx]

    #prediction_csv(gene_indices, mse_value, test_true_values, test_prediction)
    #prediction_csv(highly_expressed_low_mse_group, mse_value, test_true_values, test_prediction, gene_names)
    #prediction_csv(lowly_expressed_low_mse_group, mse_value, test_true_values, test_prediction, gene_names)
    #prediction_csv(highly_expressed_high_mse_group, mse_value, test_true_values, test_prediction, gene_names)
    #prediction_csv(lowly_expressed_high_mse_group, mse_value, test_true_values, test_prediction, gene_names)    

    #prediction_plot(gene_indices[:9], "Low MSE Prediction", mse_value[:9], test_true_values, test_prediction)
    #prediction_plot(gene_indices[-9:], "High MSE Prediction", mse_value[-9:], test_true_values, test_prediction)
    prediction_plot(highly_expressed_low_mse_group[:9], "Highly expressed with low MSE", mse_value, test_true_values, test_prediction, gene_names)
    prediction_plot(lowly_expressed_low_mse_group[:9], "Lowly expressed with low MSE", mse_value, test_true_values, test_prediction, gene_names)
    prediction_plot(highly_expressed_high_mse_group[-9:], "Highly expressed with high MSE", mse_value, test_true_values, test_prediction,gene_names)
    prediction_plot(lowly_expressed_high_mse_group[-9:], "Lowly expressed with high MSE", mse_value, test_true_values, test_prediction,gene_names)
    
    return None

def visualize_test_obs_pred(y_true, y_pred):
    '''
    Creates multiple visualizations showing the observed (true) RNAseq values
    vs the predicted RNAseq values
    '''
    plt.close()
    fig, ax = plt.subplots()
    plt.title("RNAseq Observed Values vs Predicted Values")
    plt.ylabel("Predicted Values")
    plt.xlabel("True Values")
    ax.scatter(y_true, y_pred)
    ax.axline([0, 0], [1, 1])
    plt.xlim(0, 15)
    plt.xlim(0, 15)
    plt.savefig(save_directory + '/MLP_regression_test_set_observed_vs_predicted.png', bbox_inches = 'tight')


    # Create dataframe of true and predicted values for visualization.
    data = {'True Values': np.squeeze(y_true), 'Predicted Values': np.squeeze(y_pred)}
    df = pd.DataFrame(data = data)

    # Regular joint plot
    plt.close()
    plt.title("RNAseq Observed Values vs Predicted Values Joint Plot")
    sn.jointplot(x = 'True Values', y = 'Predicted Values', data = df)
    plt.savefig(save_directory + '/MLP_regression_test_set_observed_vs_predicted_joint_plot.png', bbox_inches = 'tight')
    

    # Kernal Density Estimation joint plot
    plt.close()
    plt.title("RNAseq Observed Values vs Predicted Values KDE Joint Plot")
    sn.jointplot(x = 'True Values', y = 'Predicted Values', data = df, kind = 'kde')
    plt.savefig(save_directory + '/MLP_regression_test_set_observed_vs_predicted_KDE_joint_plot.png', bbox_inches = 'tight')


    return None

def visualize_aggregated_input_profiles(test_dataset, all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15, prediction_dataframe):
    '''
    Creates aggregated heatmap visualizations for genes within the predefined
    true expression value groups. 
    
    NOTE: Although a provision was coded to normalize the features for visualization,
    it is currently not in use because the input features are already standardized.
    
    return: Nothing
    '''

    
    heatmap_indexes = [list(all_zero_true_expression.index), list(true_expression_between_0_and_5.index), list(true_expression_between_5_and_10.index), list(true_expression_between_10_and_15.index), list(prediction_dataframe.index)]
    heatmap_names = ['Testing set - Mean of model input values - Zero Expression Genes','Testing set - Mean of model input values - Genes with true expression values between 0 and 5.', 'Testing set - Mean of model input values - Genes with true expression values between 5 and 10.','Testing set - Mean of model input values - Genes with true expression values from 10 to 15.', 'Testing set - Mean of model input values - All genes in test set.']
    min_max_scaler = preprocessing.MinMaxScaler()
    for h in tqdm(range(len(heatmap_indexes))):
        mean_gene_vals = np.mean(test_dataset[heatmap_indexes[h]], axis = 0)
        
        # Normalize each feature seperately.
        # NOTE: Normalization has been disabled here because the input has already been standardized.
        #for i in range(mean_gene_vals.shape[1]):
            #mean_gene_vals[:, i] = mean_gene_vals[:, i] / mean_gene_vals[:, i].max() # Normalize by dividing by max
            #mean_gene_vals[:, i] = mean_gene_vals[:, i] / np.sum(mean_gene_vals[:, i]) # Normalize by dividing by sum
            ####mean_gene_vals[:, i] = np.squeeze(min_max_scaler.fit_transform(mean_gene_vals[:, i].reshape(-1,1)))

        # Use seaborn to plot
        plt.close()
        #grid_kws = {'height_ratios': (0.9, 0.05), 'hspace': 0.3}
        #fig, (ax, cbar_ax) = plt.subplots(2, gridspec_kw = grid_kws, figsize = (20, 5))
        fig, ax = plt.subplots(figsize=(15, 5))
        #fig, ax = plt.subplots(figsize = (20, 5))
        
        # Visualization with color bar
        #ax = sn.heatmap(mean_gene_vals.T, annot = True, cbar_ax = cbar_ax, cbar_kws = {'orientation': 'horizontal'})
        
        fig.suptitle(f'{heatmap_names[h]}')      
        ax = sn.heatmap(mean_gene_vals.T, cbar = False, annot = True, fmt='.7f', cmap = 'OrRd', annot_kws={'rotation':90})
        ax.set_xlabel('bins')
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(['H3K27Ac','CTCF','ATAC','RNAPII'])
        ax.set_ylabel('epigenetic features')
        ax.text(x = 0.5, y = 1.04, s = f'Feature values after standardization.', fontsize = 10, ha = 'center', va = 'bottom', transform = ax.transAxes)
        plt.savefig(save_directory + '/' + heatmap_names[h] + '_seaborn.png', dpi = 150, bbox_inches = 'tight')


    return None


def visualize_correlation(test_data, highly_expressed_low_mse_group, lowly_expessed_low_mse_group, highly_expressed_high_mse_group, lowly_expressed_high_mse_group):
    '''
    Creates multiple visualizations showing the feature correlation
    for "accurate" and "inaccurate" predictions within groups where the 
    true RNAseq shows relatively low or high gene expression.  

    returns: Nothing. Visualization should appar and/or be saved.
    '''
    feature_labels = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII']
    #correlation_indexes = [low_mse_indexes, high_mse_indexes]
    correlation_indexes = [highly_expressed_low_mse_group, lowly_expessed_low_mse_group, highly_expressed_high_mse_group, lowly_expressed_high_mse_group]
    correlation_names = ['Regression - Correlation - genes with high exp, low MSE', 'Regression - Correlation - genes with low exp, low MSE', 'Regression - Correlation - genes with high exp, high MSE', 'Regression - Correlation - genes with low exp, high MSE']
    for h in range(len(correlation_indexes)):
        mean_gene_vals = np.mean(test_data[correlation_indexes[h]], axis = 0)

        df = pd.DataFrame(mean_gene_vals, columns = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'])
        corr_matrix = df.corr(method = 'pearson')
        
        #plt.matshow(df.corr(method='pearson'))
        plt.close()
        fig, ax = plt.subplots()
        #c = ax.matshow(corr_matrix, cmap = 'Oranges')
        #for i in range(corr_matrix.shape[0]):
        #    for l in range(corr_matrix.shape[1]):
        #        ax.text(x = l, y = i, s = corr_matrix[i,l], va = 'center', ha = 'center', size = 'xx-large')
        #ax.set(title = correlation_names[h])
        #ax.set_xlabel('Predicted Label')
        #ax.set_ylabel('Actual Label')
        #ax.set_yticklabels([''] + feature_labels)
        #ax.set_xticklabels([''] + feature_labels)
        #plt.colorbar(c)
        cmap = sn.diverging_palette(98, 230, as_cmap = True)
        ax = sn.heatmap(corr_matrix, annot = True, cmap = cmap, square = True, cbar_kws = {'shrink': 0.5})
        plt.title(correlation_names[h])
        plt.savefig(save_directory + '/MLP_' + correlation_names[h] +'.png')



        plt.close()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        fig, ax = plt.subplots()
        cmap = sn.diverging_palette(98, 230, as_cmap = True)
        ax = sn.heatmap(corr_matrix, annot = True, mask = mask, cmap = cmap, vmax = 0.3, square = True, linewidths = 0, cbar_kws = {'shrink': 0.5})
        plt.title(correlation_names[h])
        plt.savefig(save_directory + '/MLP_' + correlation_names[h] +'_diagonal_plot.png')


    return None

def calculate_shap_values(model, X_train, X_Seq_train, X_test, X_Seq_test_one_hot):
    '''
    Calculates Shapeley (SHAP) values and creates
    visualizations for model intrepreatation.
    param model: The trained model.
    param X_train: The training set feature inputs.
    param X_test: The test set feature inputs.
    returns: SHAP values and explainer.
    '''

    # Using deep explainer.
    #to address the "shap_LeakyRelu" error
    #shap.explainers._deep.deep_tf.op_handlers["LeakyRelu"] = shap.explainers._deep.deep_tf.passthrough
    X_Seq_train_one_hot = tf.one_hot(X_Seq_train, depth = 5, axis = -1)
    rand_idx = np.random.choice(X_train.shape[0], 500, replace = False)
    deep_explainer = shap.DeepExplainer(model, {'epigenetic_features_input': X_train[rand_idx], 'reference_genome_sequence_features_input': X_Seq_train_one_hot[rand_idx]})
    #deep_shap_values = deep_explainer.shap_values(X_test[:100, :, :])
    deep_shap_values = deep_explainer.shap_values({X_test, X_Seq_test_one_hot})
    
    # Using gradient explainer.
    #shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    #gradient_explainer = shap.GradientExplainer(model, X_train[:100, :, :])
    #gradient_shap_values = gradient_explainer(X_test[:100, :, :])

    return deep_shap_values, deep_explainer



def visualize_shap_analysis(deep_shap_values, gene_name_list, genes_of_interest_list, visualize_full_dataset_results = False):
    # Create a dictionary that can be used to map the gene name to its
    # index position for the following visualizations.
    genes_in_test_set_dict = {}
    for c, v in enumerate(gene_name_list):
        genes_in_test_set_dict[v] = c

    if visualize_full_dataset_results: 
        genes_of_interest = gene_list # Create plots for all of the genes in test set.
    else:
        genes_of_interest = genes_of_interest_list
    #### NOTE: The list functionality below has been moved to the function call where a 
    #### list of gene names can be passed into the function.
    
    # The SHAP 'local' plots will accept a specific bin out of the 50
    # for measurement.
    bin_of_interest = 25
    region_of_interest_start = 20
    region_of_interest_stop = 31

    print('Saving SHAP analysis images...')
    for gene in tqdm(genes_of_interest):
        # Lookup position of gene
        index = genes_in_test_set_dict[gene]

        # Summary plot using deepexplainer

        # Plot gene of interest in dot plot.
        
        #shap.image_plot(deep_shap_values, -X_test[0:5])   
        plt.close()
        sn.set_theme(style = 'whitegrid')
        #fig, ax = plt.subplots()
        #fig.suptitle(f"SHAP summary for gene test index {index} : {gene}")
        #ax.set(title = "SHAP summary for gene test index {} : {}".format(index, gene)) 
        #ax = shap.summary_plot(deep_shap_values[0][index], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], plot_type = "dot", show = False)
        shap.summary_plot(deep_shap_values[0][index], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], plot_type = "dot", show = False)
        plt.title(f"SHAP summary for gene test index {index} : {gene}")
        plt.savefig(save_directory + '/' + gene + '_-_regression_SHAP_summary_plot - deep_explainer_dot_plot.png', dpi = 100, bbox_inches = 'tight')


        # Plot region of interest for gene in dot plot.
        ####plt.close()
        ####fig, ax = plt.subplots()
        ####fig.suptitle(f"SHAP summary for gene test index {index} : {gene}; over region {region_of_interest_start}:{region_of_interest_stop - 1}")
        #ax.set(title = "SHAP summary for gene test index {} : {}".format(index, gene)) 
        ####ax = shap.summary_plot(deep_shap_values[0][index][region_of_interest_start:region_of_interest_stop], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], plot_type = "dot", show = False)
        ####plt.savefig(save_directory + '/' + gene + '_-_regional_analysis_-_regression_SHAP_summary_plot - deep_explainer_dot_plot.png')


        # Plot gene of interest in bar plot.
        plt.close()
        sn.set_theme(style = 'whitegrid')
        #fig, ax = plt.subplots()
        #fig.suptitle(f"SHAP summary for gene test index {index} : {gene}")
        #ax.set(title = "SHAP summary for gene test index {} : {}".format(index, gene))
        #ax = shap.summary_plot(deep_shap_values[0][index], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], plot_type = "bar", show = False)
        shap.summary_plot(deep_shap_values[0][index], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], plot_type = "bar", show = False)
        plt.title(f"SHAP summary for gene test index {index} : {gene}")
        plt.savefig(save_directory + '/' + gene + '_-_cross_patient_regression_SHAP_deep_explianer_bar_plot.png', dpi = 100, bbox_inches = 'tight')


        # Plot region of interest for gene in bar plot.
        ####plt.close()
        ####fig, ax = plt.subplots()
        ####fig.suptitle(f"SHAP summary for gene test index {index} : {gene}; over region {region_of_interest_start}:{region_of_interest_stop - 1}")
        #ax.set(title = "SHAP summary for gene test index {} : {}".format(index, gene))
        ####ax = shap.summary_plot(deep_shap_values[0][index][region_of_interest_start:region_of_interest_stop], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], plot_type = "bar", show = False)
        ####plt.savefig(save_directory + '/' + gene + '_-_regional_analysis_-_regression_SHAP_summary_plot - deep_explianer_bar_plot.png')


        #plt.close()
        #shap.summary_plot(gradient_shap_values[0], plot_type = 'bar', show=True)
        #plt.savefig(save_directory + '/Regression_SHAP_summary_plot - gradient.png')

        #plt.close()
        #shap.plots.bar(deep_shap_values)
        #plt.savefig(save_directory + '/Regression_SHAP_summary_plot - gradient-bar.png')

        # Summary plot 2
        #plt.close()
        #shap.plots.beeswarm(deep_explainer, show = False)    
        #plt.savefig(save_directory + 'Regression_SHAP_beeswarm_plot.png')

        # Waterfall plot
        # This plot type can take only one example at a time. It cannot plot groups or multiple 
        # indexes
        ####plt.close()
        ####fig, ax = plt.subplots()
        ####fig.suptitle(f"SHAP local analysis for gene test index {index} : {gene}; bin : {bin_of_interest}")
        #ax.set(title = "SHAP local analysis for gene test index {} : {}; bin : {}".format(index, gene, bin_of_interest))
        ####ax = shap.plots._waterfall.waterfall_legacy(deep_explainer.expected_value[0], deep_shap_values[0][index][bin_of_interest], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], show = False)
        ####plt.savefig(save_directory + '/' + gene + '_-_regression_SHAP_waterfall_plot.png')


        # Force plot
        #shap.initjs()
        #plt.close()
        #fig, ax = plt.subplots()
        #ax = shap.force_plot(deep_explainer.expected_value[0], deep_shap_values[0][index][bin_of_interest], features = ['H3K27aAc', 'CTCF', 'ATAC', 'RNAPII'], show = False)
        #plt.savefig(save_directory + '/' + gene + '_regression_SHAP_force_plot.png')


        # Decision plot

        # Plot region of interest
        ####plt.close()
        ####fig, ax = plt.subplots()
        ####fig.suptitle(f"SHAP local analysis for gene test index {index} : {gene}; bin : {region_of_interest_start}:{region_of_interest_stop - 1}")
        #ax.set(title = "SHAP summary for gene test index {} : {}; bin : {}".format(index, gene, bin_of_interest))
        ####ax = shap.decision_plot(deep_explainer.expected_value[0], deep_shap_values[0][index][region_of_interest_start:region_of_interest_stop], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], show = False)
        ####plt.savefig(save_directory + '/'+ gene + '_-_regional_analysis_-_regression_SHAP_decision_plot.png')

    return None


def visualize_aggregated_shap_analysis(shap_values, deep_explainer, all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15, prediction_dataframe):
    
    '''
    Function produces visualizations for groups
    of genes.

    return: Nothing
    '''
    gene_indecies = [list(all_zero_true_expression.index), list(true_expression_between_0_and_5.index), list(true_expression_between_5_and_10.index), list(true_expression_between_10_and_15.index), list(prediction_dataframe.index)]
    indices_names = ['SHAP Analysis - Zero Expression Genes','SHAP Analysis - Genes with true expression values between 0 and 5', 'SHAP Analysis - Genes with true expression values between 5 and 10','SHAP Analysis - Genes with true expression values from 10 to 15', 'SHAP Analysis - All genes in test set']
    bin_of_interest = 25
    region_of_interest_start = 20
    region_of_interest_stop = 31

    print('Saving expression catagory SHAP analysis images...')
    for i in tqdm(range(len(gene_indecies))):
        list_of_mean_group_shap_values = []
        #for l in i:
        #group_shap_values.append(shap_values[0][l])
        group_shap_values = np.array(shap_values[0])[gene_indecies[i]]
        #mean_group_shap_values = np.mean(np.array(group_shap_values), axis = 0).tolist()
        mean_group_shap_values = np.mean(group_shap_values, axis = 0)
        list_of_mean_group_shap_values.append(mean_group_shap_values)

        # Summary plots.

        # Plot gene catagory in dot plot.
        plt.close()
        sn.set_theme(style = 'whitegrid')
        #shap.image_plot(deep_shap_values, -X_test[0:5])   
        ####fig, ax = plt.subplots()
        ####fig.suptitle(f"SHAP summary for gene test {indices_names[i]}")
        ####ax = shap.summary_plot(list_of_mean_group_shap_values[0], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], plot_type = "dot", show = False)
        shap.summary_plot(list_of_mean_group_shap_values[0], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], plot_type = "dot", show = False)
        plt.title(f"{indices_names[i]}")
        plt.savefig(save_directory + '/' + indices_names[i] + '_-_regression_SHAP_summary_plot_-_deep_explainer_dot_plot.png', dpi = 100, bbox_inches = 'tight')


        # Plot region of interest for catagory in dot plot.
        ####plt.close()
        ####fig, ax = plt.subplots()
        ####fig.suptitle(f"SHAP summary for gene test {indices_names[i]}; over region {region_of_interest_start}:{region_of_interest_stop - 1}")
        ####ax = shap.summary_plot(list_of_mean_group_shap_values[0][region_of_interest_start:region_of_interest_stop], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], plot_type = "dot", show = False)
        ####plt.savefig(save_directory + '/' + indices_names[i] + '_-_regional_analysis_-_regression_SHAP_summary_plot - deep_explainer_dot_plot.png')


        # Plot gene catagory in bar plot.
        plt.close()
        sn.set_theme(style = 'whitegrid')
        ####fig, ax = plt.subplots()
        ####fig.suptitle(f"SHAP summary for gene test {indices_names[i]}")
        ####ax = shap.summary_plot(list_of_mean_group_shap_values[0], feature_names = ['H3K27ac', 'CTCF', 'ATAC', 'RNA Pol II'], plot_type = "bar", show = False)
        shap.summary_plot(list_of_mean_group_shap_values[0], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], plot_type = "bar", show = False)
        plt.title(f"{indices_names[i]}")
        plt.savefig(save_directory + '/' + indices_names[i] + 'SHAP_summary_plot_-_deep_explianer_bar_plot.png', dpi = 100, bbox_inches = 'tight')


        # Plot region of interest for catagory in bar plot.
        ####plt.close()
        ####fig, ax = plt.subplots()
        ####fig.suptitle(f"SHAP summary for gene test {indices_names[i]}; over region {region_of_interest_start}:{region_of_interest_stop - 1}")
        ####ax = shap.summary_plot(list_of_mean_group_shap_values[0][region_of_interest_start:region_of_interest_stop], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], plot_type = "bar", show = False)
        ####plt.savefig(save_directory + '/' + indices_names[i] + '_-_regional_analysis_-_regression_SHAP_summary_plot - deep_explianer_bar_plot.png')



        # Waterfall plot
        # This plot type can take only one example at a time. It cannot plot groups or multiple 
        # indexes
        #plt.close()
        #fig, ax = plt.subplots()
        #fig.suptitle(f"SHAP local analysis for gene test index {i}; bin : {bin_of_interest}")
        ##ax.set(title = "SHAP local analysis for gene test index {} : {}; bin : {}".format(index, gene, bin_of_interest))
        #ax = shap.plots._waterfall.waterfall_legacy(deep_explainer.expected_value[0], group_shap_values[0][bin_of_interest], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], show = False)
        #plt.savefig(save_directory + '/' + i + '_regression_SHAP_waterfall_plot.png')


        # Decision plot
        # Plot region of interest for catagory in decision plot.
        ####plt.close()
        ####fig, ax = plt.subplots()
        ####fig.suptitle(f"SHAP local analysis for gene test {indices_names[i]}; bins : {region_of_interest_start}:{region_of_interest_stop - 1}")
        #ax.set(title = "SHAP summary for gene test index {} : {}; bin : {}".format(index, gene, bin_of_interest))
        ####ax = shap.decision_plot(deep_explainer.expected_value[0], list_of_mean_group_shap_values[0][region_of_interest_start:region_of_interest_stop], feature_names = ['H3K27Ac', 'CTCF', 'ATAC', 'RNAPII'], show = False)
        ####plt.savefig(save_directory + '/'+ indices_names[i] + '_-_regional_analysis_-_regression_SHAP_decision_plot.png')

    #return list_of_mean_group_shap_values
    return None



def make_prediction(model, input_data_1, input_data_2):
    '''
    param model: a trained model
    param input_data: model inputs
    return: the model's predictions for the provided input data
    '''

    return np.asarray(model.predict({'epigenetic_features_input': input_data_1, 'reference_genome_sequence_features_input': input_data_2}), dtype='float')

def prediction_csv(se_value, y_true, y_pred, gene_names):
    '''
    Writes each gene name with it's true and 
    predicted RNAseq values along with the calculation 
    of the Squared Error into a csv file saved to the 
    model's output directory.
    
    param se_value: Array of the calculated Squared Error values per gene.
    param y_true: the true (observed) RNAseq values.
    param y_pred: the model's predicted values.
    param gene_names: list of each gene name in the test set.
    
    returns: Nothing
    '''
   # Create csv file to hold prediction information.
    with open(save_directory + '/mlp_cross_patient_regression_gsc_stem_standard_test_predictions.csv', 'w') as log:
            log.write(f'gene name, true RNAseq value, predicted RNAseq value, prediction Squared Error (SE)')

    for i in tqdm(range(len(gene_names))):
        #index = gene_indices[i]
        gn = gene_names[i]
        se = se_value[i]
        tv = y_true[i]
        pv = y_pred[i]
        with open(save_directory + '/mlp_cross_patient_regression_gsc_stem_standard_test_predictions.csv', 'a') as log:
            log.write('\n' f'{gn}, {tv[0]}, {pv[0]}, {se[0]}')

    return None

def load_csv_and_create_dataframes():
    
    prediction_dataframe = pd.read_csv(save_directory + '/mlp_cross_patient_regression_gsc_stem_standard_test_predictions.csv', float_precision = 'round_trip')
    
    # Define gene expression catagories for analysis.

    all_zero_true_expression = prediction_dataframe[prediction_dataframe[' true RNAseq value'] == 0]
    true_expression_between_0_and_5 = prediction_dataframe[(prediction_dataframe[' true RNAseq value'] > 0) & (prediction_dataframe[' true RNAseq value'] < 5)]
    true_expression_between_5_and_10 = prediction_dataframe[(prediction_dataframe[' true RNAseq value'] >= 5) & (prediction_dataframe[' true RNAseq value'] < 10)]
    true_expression_between_10_and_15 = prediction_dataframe[(prediction_dataframe[' true RNAseq value'] >= 10) & (prediction_dataframe[' true RNAseq value'] <= 15)]
    #print(len(all_zero_expression) + len(between_0_and_5) + len(between_5_and_10) + len(between_10_and_15))
    
    return prediction_dataframe, all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15


def visualize_model_test_results(results_list):
    '''
    Creates a bar graph visualization of the test metric 
    results for the current model run. The image is saved to the model's 
    image directory.
    
    param results_list: The output of model.evaluate .
    
    returns: Nothing
    '''
    
    results_dataframe = pd.DataFrame({"metric" : ["PCC", "SCC", "R2"],
                                      "test set results": [results_list[1], results_list[3], results_list[2]]})
    plt.close()
    sn.set_theme(style = 'whitegrid')
    fig, ax = plt.subplots(figsize = (8, 5))
    ax = sn.barplot(data = results_dataframe, x = "metric", y = "test set results")
    ax.set_xticklabels(ax.get_xticklabels())
    ax.set(title = 'MLP Cross Patient Regression Test Results')
    sn.set(font_scale=1)
    for i in ax.containers:
        ax.bar_label(i,)
    plt.savefig(save_directory + '/mlp_cross_patient_regression_gsc_stem_standard_test_metric_results.png', 
                bbox_inches='tight')
    
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
    
    ax = sn.heatmap(genes_for_vis.reshape(genes_for_vis.shape[0], 1), cmap = "YlGnBu", annot = True)
    
    # The number and range of genes presented can be adjusted by slicing as above.
    ax.set_yticklabels(gene_names_in_test_set[:50], rotation = 0)
    
    ax.tick_params(left=True, bottom=False)
    ax.set_xticklabels([])
    #ax.set_yticks(np.arange(len(gene_names_in_test_set[:10])), labels = gene_names_in_test_set[:10], rotation = 0)
    plt.savefig(save_directory + '/mlp_cross_patient_regression_gsc_stem_standard_test_gene_squared_error_heatmap.png', 
                bbox_inches = 'tight')
    #plt.show()
    
    return None

def visualize_testing_distributions(all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15):
    # Create dataframe for catagory counts. The dataframe will be used for visualization.
    expression_counts_dataframe = pd.DataFrame({"expression catagory" : ["all_zero_true_expression", "true_expression_between_0_and_5", 
                                                                    "true_expression_between_5_and_10", "true_expression_between_10_and_15" ],
                                                "count": [len(all_zero_true_expression), len(true_expression_between_0_and_5), 
                                                     len(true_expression_between_5_and_10), len(true_expression_between_10_and_15)]})

    # Visualize number of genes in each catagory.
    plt.close()
    sn.set_theme(style = "whitegrid")
    fig, ax = plt.subplots(figsize = (8, 5))
    ax = sn.barplot(data = expression_counts_dataframe, x = "expression catagory", y = "count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = "45")
    ax.set(title = 'Testing Set - MLP Cross Patient Regression Expression Catagory Counts')
    sn.set(font_scale=1)
    for i in ax.containers:
        ax.bar_label(i,)

    plt.savefig(save_directory + '/mlp_cross_patient_regression_gsc_stem_standard_test_expression_catagory_counts.png', 
                bbox_inches = 'tight')

    return None

def visualize_testing_set_mse_by_catagory(test_set_MSE, all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15):
    
    # Create dataframe for the calculation of the mean SE. 
    # The dataframe will be used for visualization.
    expression_mean_dataframe = pd.DataFrame({"expression catagory after log(2) transform" : ["all zero", "between 0 and 5", 
                                                                    "between 5 and 10", "between 10 and 15" ],
                                              "mean squared error (MSE)" : [all_zero_true_expression[' prediction Squared Error (SE)'].mean(),
                                                    true_expression_between_0_and_5[' prediction Squared Error (SE)'].mean(),
                                                    true_expression_between_5_and_10[' prediction Squared Error (SE)'].mean(),
                                                    true_expression_between_10_and_15[' prediction Squared Error (SE)'].mean()]})
    
    plt.close()
    sn.set_theme(style = "whitegrid")
    fig, ax = plt.subplots(figsize = (8, 5))
    ax = sn.barplot(data = expression_mean_dataframe, x = "expression catagory after log(2) transform", y = "mean squared error (MSE)")
    ax.set(title = 'Testing Set - MLP Cross Patient Regression MSE Per Expression Catagory')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = "45")
    ax.axhline(test_set_MSE, label = f'MSE of entire test set: {test_set_MSE}')
    plt.legend(loc = 'upper left')
    for i in ax.containers:
        ax.bar_label(i,)
    
    
    plt.savefig(save_directory + '/mlp_cross_patient_regression_gsc_stem_standard_test_expression_catagory_MSE.png', 
                bbox_inches = 'tight')

    return None




def main(loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, count, batch_size, learning_rate, epigenetic_branch_hidden_size_1, epigenetic_branch_hidden_size_2, epigenetic_branch_hidden_size_3, epigenetic_branch_dropout_rate, sequence_branch_hidden_size_1, sequence_branch_hidden_size_2, sequence_branch_hidden_size_3, sequence_branch_dropout_rate, combined_hidden_size_1, combined_hidden_size_2, combined_hidden_size_3, combined_dropout_rate):
    
    # Save directory - path where result files and figures are saved
    global save_directory

    now = datetime.datetime.now()
    
    if sys.argv[6:]:
        # Save path given by the user in the 6th argument to the global variable
        save_directory = str(sys.argv[6])
        # Create the given directory
        print('*'*25)
        print(f'Using {save_directory} as the save directory.')
        print('*'*25)
        os.makedirs(save_directory, exist_ok = True)

    else:
        save_directory = './cross_patient_regression_using_branched_mlp_-_results_and_figures_-_' + \
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
    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]
    file_path_3 = sys.argv[3]
    indices = sys.argv[4]
    random_state = int(sys.argv[5])
    print('*'*25)
    print('The random seed is set to: ')
    print(random_state)
    print('*'*25) 
    
    
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
    if validation == True:
        X_train, X_val, Y_train, Y_val, gene_dict, num_genes, X_Seq_train, X_Seq_val = get_data_patient_1(file_path_1, 
                                                                                                          file_path_3, 
                                                                                                          indices, 
                                                                                                          gene_dict, 
                                                                                                          num_genes, 
                                                                                                          preprocess = preprocess_bool, 
                                                                                                          validation = validation_bool)
    else:
        X_train, Y_train, gene_dict, num_genes, X_Seq_train = get_data_patient_1(file_path_1, 
                                                                                                          file_path_3, 
                                                                                                          indices, 
                                                                                                          gene_dict, 
                                                                                                          num_genes, 
                                                                                                          preprocess = preprocess_bool, 
                                                                                                          validation = validation_bool)
    
    # Processing data for patient 2 file to produce test set.
    X_test, Y_test, gene_dict, num_genes, X_Seq_test, test_set_indices = get_data_patient_2(file_path_2, file_path_3, indices, gene_dict, num_genes, preprocess = preprocess_bool)
    
    # Call train_model() to train the model
    print("Training model...")
    if validation == True:
        model, history = train_model(X_train, Y_train, X_Seq_train, epochs=150, validation = validation_bool, batch_size = batch_size, learning_rate = learning_rate, epigenetic_branch_hidden_size_1 = epigenetic_branch_hidden_size_1, epigenetic_branch_hidden_size_2 = epigenetic_branch_hidden_size_2, epigenetic_branch_hidden_size_3 = epigenetic_branch_hidden_size_3, epigenetic_branch_dropout_rate = epigenetic_branch_dropout_rate, 
                                 sequence_branch_hidden_size_1 = sequence_branch_hidden_size_1, sequence_branch_hidden_size_2 = sequence_branch_hidden_size_2, sequence_branch_hidden_size_3 = sequence_branch_hidden_size_3, sequence_branch_dropout_rate = sequence_branch_dropout_rate, 
                                combined_hidden_size_1 = combined_hidden_size_1, combined_hidden_size_2 = combined_hidden_size_2, combined_hidden_size_3 = combined_hidden_size_3, combined_dropout_rate = combined_dropout_rate, random_state = random_state, 
                                     X_val = X_val,
                                     Y_val = Y_val,
                                     X_Seq_val = X_Seq_val)
        
        min_val_loss = min(history.history['val_loss'])
        max_val_pcc = max(history.history['val_pearson_r'])
        max_val_scc = max(history.history['val_spearman_r'])
        max_val_r2_score = max(history.history['val_r_square'])
        
    else:
        model, history = train_model(X_train, Y_train, X_Seq_train, epochs = 150, validation = validation_bool, batch_size = batch_size, learning_rate = learning_rate, epigenetic_branch_hidden_size_1 = epigenetic_branch_hidden_size_1, epigenetic_branch_hidden_size_2 = epigenetic_branch_hidden_size_2, epigenetic_branch_hidden_size_3 = epigenetic_branch_hidden_size_3, epigenetic_branch_dropout_rate = epigenetic_branch_dropout_rate, 
                                 sequence_branch_hidden_size_1 = sequence_branch_hidden_size_1, sequence_branch_hidden_size_2 = sequence_branch_hidden_size_2, sequence_branch_hidden_size_3 = sequence_branch_hidden_size_3, sequence_branch_dropout_rate = sequence_branch_dropout_rate, 
                                combined_hidden_size_1 = combined_hidden_size_1, combined_hidden_size_2 = combined_hidden_size_2, combined_hidden_size_3 = combined_hidden_size_3, combined_dropout_rate = combined_dropout_rate, random_state = random_state, 
                                     X_val = None, Y_val = None, X_Seq_val = None)
        
        min_val_loss = 'TRAINING SET ONLY'
        max_val_pcc = 'TRAINING SET ONLY'
        max_val_scc = 'TRAINING SET ONLY'
        max_val_r2_score = 'TRAINING SET ONLY'

    min_loss = min(history.history['loss'])
    max_pcc = max(history.history['pearson_r'])
    max_scc = max(history.history['spearman_r'])
    max_r2_score = max(history.history['r_square'])


    loss_dict[count] = min_loss
    pcc_dict[count] = max_pcc
    scc_dict[count] = max_scc
    r2_score_dict[count] = max_r2_score
    
    val_loss_dict[count] = min_val_loss
    val_pcc_dict[count] = max_val_pcc
    val_scc_dict[count] = max_val_scc
    val_r2_score_dict[count] = max_val_r2_score
    
    print("CURRENT COUNT:", count, " learning rate: ", learning_rate, " batch size: ", batch_size, "\n epigenetic branch hidden layer size 1: ", epigenetic_branch_hidden_size_1, " epigenetic branch hidden layer size 2: ", epigenetic_branch_hidden_size_2, " epigenetic branch hidden layer size: ", epigenetic_branch_hidden_size_3, " epigenetic branch dropout rate: ", epigenetic_branch_dropout_rate, "\n sequence branch hidden layer size 1: ", sequence_branch_hidden_size_1, " sequence branch hidden layer size 2: ", sequence_branch_hidden_size_2, " sequence branch hidden layer size 3: ", sequence_branch_hidden_size_3, " sequence branch dropout rate: ", sequence_branch_dropout_rate, "\n combined hidden layer size 1: ", combined_hidden_size_1, " combined hidden layer size 2: ", combined_hidden_size_2, " combined hidden layer size: ", combined_hidden_size_3, " combined dropout rate: ", combined_dropout_rate, "\n Min training loss: ", min_loss, "\n Max training correlation: ", max_pcc, "\n Min val loss: ", min_val_loss, "\n Max val correlation: ", max_val_pcc)
    
    print('*'*25)
    print("Evaluating model...")
    print('*'*25)
    
    # One-hot encode genomic sequence values
    X_Seq_test_one_hot = tf.one_hot(X_Seq_test, depth = 5, axis = -1)
    
    # Evaluate model
    #results  = model.evaluate({'epigenetic_features_input': X_test, 'reference_genome_sequence_features_input': X_Seq_test_one_hot}, Y_test, batch_size = batch_size)
    
    results = test_model(model, X_test, Y_test, 
                         X_Seq_test_one_hot, batch_size)
    
    print("Test results:")
    print(f"loss,{results[0]}")
    print(f"PCC,{results[1]}")
    print(f"SCC,{results[2]}")
    print(f"R2,{results[3]}")
    print('*'*25)
    
    # Log file formatted with commas and one new line
    # to facilitate import into a spreadsheet or pandas
    # to sort values.
    #if os.path.exists('mlp_tuning_regression_baseline_gsc_stem_standard_log2.log'):
    with open(save_directory + '/mlp_cross_patient_regression_gsc_stem_sequence_standard_log2_info.csv', 'a') as log:
        log.write('\n' f'{now.strftime("%H:%M on %A, %B %d")},')
        log.write(f'CURRENT COUNT: {count},learning rate: {learning_rate},batch size: {batch_size},')
        log.write(f'epigenetic branch hidden layer 1 size: {epigenetic_branch_hidden_size_1}, epigenetic branch hidden layer 2 size: {epigenetic_branch_hidden_size_2}, epigenetic branch hidden layer 3 size: {epigenetic_branch_hidden_size_3}, epigenetic branch dropout_rate: {epigenetic_branch_dropout_rate},')
        log.write(f'sequence branch hidden layer 1 size: {sequence_branch_hidden_size_1}, sequence branch hidden layer 2 size: {sequence_branch_hidden_size_2}, sequence branch hidden layer 3 size: {sequence_branch_hidden_size_3}, sequence branch dropout_rate: {sequence_branch_dropout_rate},')
        log.write(f'combined hidden layer 1 size: {combined_hidden_size_1}, combined hidden layer 2 size: {combined_hidden_size_2}, combined hidden layer 3 size: {combined_hidden_size_3}, combined dropout_rate: {combined_dropout_rate},')        
        log.write(f'Min training loss: {min_loss},Max training PCC: {max_pcc},Max training SCC: {max_scc},Max training R2 Score: {max_r2_score},')
        log.write(f'Min val loss: {min_val_loss}, Max val PCC {max_val_pcc}, Max val SCC {max_val_scc}, Max val R2 Score: {max_val_r2_score},')
        log.write(f'Test loss: {results[0]}, Test PCC {results[1]},Test SCC {results[2]},Test R2 Score: {results[3]}')                                
    gene_names_in_test_set = get_gene_names(gene_dict, indices, X_test.shape[0], num_genes, test_set_indices)                                                                            
    
    visualize_metrics_together(history, validation_bool, count)

    # Visualize model training and validation loss.
    #visualize_loss(history, validation_bool, count)


    # Visualize model training and validation pcc.
    visualize_pcc(history, validation_bool, count)


    # Visualize model training and validation RSquare.
    #visualize_rsquare(history, validation_bool, count) 


    # Visualize model training and validation SCC.
    visualize_scc(history, validation_bool, count)

    # Make predictions given test data inputs
    Y_pred = make_prediction(model, X_test, X_Seq_test_one_hot)
       
    # Calculate the se of each gene in prediction results
    prediction_se = calculate_se_for_predictions(Y_test, Y_pred)
    #print(prediction_mse.shape)
    
    visualize_prediction_mse(prediction_se, Y_test, Y_pred)
    visualize_test_obs_pred(Y_test, Y_pred)
    # Get lists of "best" and "worst" predictions
    #low_mse_indexes, high_mse_indexes, sorted_mse_with_idx = calculate_prediction_high_low(prediction_se, Y_test, Y_pred)
    #stacked_arrays = calculate_prediction_high_low(prediction_mse, Y_test, Y_pred)
    
    #highly_expressed_low_mse_group, lowly_expessed_low_mse_group, highly_expressed_high_mse_group, lowly_expressed_high_mse_group = calculate_prediction_high_low(prediction_mse, Y_test, Y_pred)
    #print(highly_expressed_low_mse_group)
    #print(lowly_expessed_low_mse_group)
    
    prediction_csv(prediction_se, Y_test, Y_pred, gene_names_in_test_set)
    
    prediction_dataframe, all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15 = load_csv_and_create_dataframes()

    visualize_model_test_results(results)
    
    visualize_se_heatmap(prediction_se, gene_names_in_test_set)
    
    #visualize_prediction(highly_expressed_low_mse_group, lowly_expessed_low_mse_group, highly_expressed_high_mse_group, lowly_expressed_high_mse_group, prediction_se, Y_test, Y_pred, gene_names_in_test_set)
    
    visualize_aggregated_input_profiles(X_test, all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15,prediction_dataframe)
     
    ##visualize_correlation(X_test, highly_expressed_low_mse_group, lowly_expessed_low_mse_group, highly_expressed_high_mse_group, lowly_expressed_high_mse_group)

    visualize_training_distributions(Y_train)
    
    if validation_bool == True:
        visualize_validation_distributions(Y_val)
    
    visualize_testing_distributions(all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15)
    
    test_set_MSE = prediction_dataframe[' prediction Squared Error (SE)'].mean()
    
    visualize_testing_set_mse_by_catagory(test_set_MSE, all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15)
    
    
    #### NOTE: As of 01/05/23 SHAP analysis has been disabled in this script. It appears 
    #### as though the 'branched' architecture and the fact that there are two 
    #### distinct types of input features, is not compatible with the syntax used with SHAP to calculate the 
    #### feature importance. Model tuning and testing is being done without it until a solution is found.
   
    # Calculate SHAP feature importance values for analysis.    
    ####deep_shap_values, deep_explainer = calculate_shap_values(model, X_train, X_Seq_train, X_test, X_Seq_test_one_hot)
    
    # SHAP visualization for model interpretation
    ####visualize_shap_analysis(deep_shap_values, gene_names_in_test_set, ['MUC20', 'AIMP2', 'NFYB', 'CYP7B1', 'SLC35F5', 'TPM4', 'ITGB1', 'COL1A1', 'GNAS', 'POSTN'], visualize_full_dataset_results = False)
    

    # SHAP analysis at gene grouping level and visualization for model interpretation
    ####visualize_aggregated_shap_analysis(deep_shap_values, deep_explainer, all_zero_true_expression, true_expression_between_0_and_5, true_expression_between_5_and_10, true_expression_between_10_and_15, prediction_dataframe)    
    

    return loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, X_train, X_val, X_test, Y_train, Y_val, Y_test, X_Seq_train, X_Seq_val, X_Seq_test, indices, model, history, results, validation_bool

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

    # Hyperparameter combination that produces the best validation loss and metrics
    parameters = dict(learning_rate_vals = [1e-2], batch_size_vals = [512], epigenetic_branch_hidden_size_1 = [200], epigenetic_branch_hidden_size_2 = [200], epigenetic_branch_hidden_size_3 = [200], epigenetic_branch_dropout_rate = [0.3], sequence_branch_hidden_size_1 = [200], sequence_branch_hidden_size_2 = [200], sequence_branch_hidden_size_3 = [100], sequence_branch_dropout_rate = [0.3], combined_hidden_size_1 = [50], combined_hidden_size_2 = [200], combined_hidden_size_3 = [20], combined_dropout_rate = [0.3])

    # Hyperparameter grid search
    #parameters = dict(learning_rate_vals = [1e-2, 5e-2, 1e-3, 5e-3, 5e-4, 1e-4], batch_size_vals = [512, 1024], epigenetic_branch_hidden_size_1 = [200, 100, 50, 20], epigenetic_branch_hidden_size_2 = [200, 100, 50, 20], epigenetic_branch_hidden_size_3 = [200, 100, 50, 20], epigenetic_branch_dropout_rate = [0.3, 0.4, 0.5], sequence_branch_hidden_size_1 = [200, 100, 50, 20], sequence_branch_hidden_size_2 = [200, 100, 50, 20], sequence_branch_hidden_size_3 = [200, 100, 50, 20], sequence_branch_dropout_rate = [0.3, 0.4, 0.5], combined_hidden_size_1 = [200, 100, 50, 20], combined_hidden_size_2 = [200, 100, 50, 20], combined_hidden_size_3 = [200, 100, 50, 20], combined_dropout_rate = [0.3, 0.4, 0.5])
    
   
    param_values = [v for v in parameters.values()]

    count = 0

    for lr, bs, ebhs1, ebhs2, ebhs3, ebdr, sbhs1, sbhs2, sbhs3, sbdr, chs1, chs2, chs3, cdr  in product(*param_values): 
        loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, X_train, X_val, X_test, Y_train, Y_val, Y_test, X_Seq_train, X_Seq_val, X_Seq_test, indices, model, history, results, validation_bool = main(loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, count, batch_size=bs, learning_rate=lr, epigenetic_branch_hidden_size_1 = ebhs1, epigenetic_branch_hidden_size_2 = ebhs2, epigenetic_branch_hidden_size_3 = ebhs3, epigenetic_branch_dropout_rate = ebdr,
        sequence_branch_hidden_size_1 = sbhs1, sequence_branch_hidden_size_2 = sbhs2, sequence_branch_hidden_size_3 = sbhs3, sequence_branch_dropout_rate = sbdr, combined_hidden_size_1 = chs1, combined_hidden_size_2 = chs2, combined_hidden_size_3 = chs3, combined_dropout_rate = cdr)
        count+=1

    min_loss_count = min(loss_dict, key=loss_dict.get)
    max_pcc_count = max(pcc_dict, key=pcc_dict.get)
    max_r2_count = max(r2_score_dict, key=r2_score_dict.get)
    max_scc_count = max(scc_dict, key=scc_dict.get)
    min_val_loss_count = min(val_loss_dict, key=val_loss_dict.get)
    max_val_pcc_count = max(val_pcc_dict, key=val_pcc_dict.get)
    max_val_r2_count = max(val_r2_score_dict, key=val_r2_score_dict.get)
    max_val_scc_count = max(val_scc_dict, key=val_scc_dict.get)
    

    print("\n Min training loss and count: ", min(loss_dict.values()), min_loss_count, "\n Max training pcc and count: ", max(pcc_dict.values()), max_pcc_count, "\n Max training R2 and count: ", max(r2_score_dict.values()), max_r2_count, "\n Max training scc and count: ", max(scc_dict.values()), max_scc_count, "\n Min val loss and count: ", min(val_loss_dict.values()), min_val_loss_count, "\n Max val pcc and count: ", max(val_pcc_dict.values()), max_val_pcc_count, "\n Max val R2 and count: ", max(val_r2_score_dict.values()), max_val_r2_count, "\n Max val scc and count: ", max(val_scc_dict.values()), max_val_scc_count)
