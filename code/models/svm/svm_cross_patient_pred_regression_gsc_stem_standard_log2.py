'''
This is a Support Vector Regression version of a 
Support Vector Machine algorithm model composed
using the scikit-learn library.

Goal: Predict gene expression value (regression) 
from epigenetic signal inputs.

It is configured for cross-patient prediction.
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
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from itertools import product
import random
import datetime

def get_data_patient_1(file_path, indices, gene_dict, num_genes, 
                       preprocess, validation):

    '''
    Patient 1's data preperation for model input.
    param file_path: location of patient data
    param indices: location of file used to shuffle gene order
    param gene_dict: dictionary of gene names and their 
                     position in data.
    param num_genes: the number of genes in a patient's dataset
    param preprocess: boolean to determine if function should perform
                      processing
    param validation: boolean to determine if function shoud produce 
                      a useable validation dataset
    return: X_train, X_val, Y_train, Y_val; where X refers 
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

            # Set corresponding value to be first bin's RNAseq value
            # (since all 50 bins have the same value when using
            # the featureCounts utility and process).
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
            
            # List of all datasets after split operation.
            # Standardization ONLY on input variables.
            datasets = [X_train, X_val]

        
        else:
            # TESTING SPLITS
            # The training set will have 99% of the patient 1 data to train the model.
            train_ind = ind
            X_train = X[train_ind]
            Y_train = Y[train_ind]
            datasets = [X_train]
                


        # Perform calculation on each column of the seperate train, validation and test sets. 
        for dataset in datasets:
            for i in range(dataset.shape[2]):
                # Standardize the column values.
                dataset[:, :, i] = (dataset[:, :, i] - np.mean(dataset[:, :, i])) / np.std(dataset[:, :, i], ddof = 1) # The degrees of freedom is set t
                
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
    Patient 2's data preperation for model input.
    param file_path: location of patient data
    param indices: location of file used to shuffle gene order
    param gene_dict: dictionary of gene names and their 
                     position in data.
    param num_genes: the number of genes in a patient's dataset
    param preprocess: boolean to determine if function should perform                           processing
    return: X_test, Y_test; where X refers to inputs and Y refers to                   labels.
    return: gene_dict (gene name dictionary)
    return: num_genes (the number of genes in the test set)
    return: patient2_ind (the indices for the patient 2 dataset. This
            may be a subset of the indices shuffle index 
            if the number of genes in the test set is 
            lower than the 20,015 in the training set) 
    '''

    if preprocess:
        print("Loading patient 2 dataset...")
        # Col 1 = gene names, 2 = bin number, 3-6 = features, 7 = labels
        combined_diff = np.load(file_path, allow_pickle = True)
        
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

            # Each subset is of shape 100 x 6 
            # (number of 100bp bins x number of columns)
            subset = combined_diff[np.where(combined_diff[:, 0] == name)]

            # Create matrix of data. 
            gene_ind = gene_dict[name]
            data = subset[:, 2:]

            data_inputs = data[:, :-1]

            # Add to array at the unique id position
            X[gene_ind] = data_inputs
   
            # Set corresponding value to be first 
            # bin's RNAseq value (since all 50 bins
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
        #test_ind = ind[int(0.85*num_genes):]


        # NOTE: Use entire dataset for test set.
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

    return X_test, Y_test, gene_dict, num_genes, patient2_ind


def reset_random_seeds(seed):
    '''
    Takes a given number and assigns it
    as a random seed to various generators and the
    os environment.
    param seed: the number to be assigned as the random seed
    return: nothing
    '''

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)

    return None

def train_model(X_train, Y_train, 
                validation, C_value, epsilon_value, 
                random_state, X_val, Y_val):
    '''
    Train function for a SVR model.
    param X_train: the training epigenetic inputs.
    param Y_train: the training RNAseq values.
    param X_val: the validation epigenetic inputs if appropriate.
    param Y_val: the validation RNAseq values if appropriate.
    param validation: boolean to determine if function shoud make use of validation dataset
    param C_value: model hyperparemeter
    param epsilon_value: model hyperparameter
    param random_state: the random seed value for model run
    return: Trained model and validation metrics if applicable.
    '''

    
    random_state = random_state
    
    # Set random seed
    reset_random_seeds(random_state)
    
    
    # Reshape data into 2 dimensions.
    reshaped_X_train = X_train.reshape((X_train.shape[0], -1), 
                                       order = 'F')
    reshaped_Y_train = np.squeeze(Y_train)
    
    if validation == True:
        reshaped_X_val = X_val.reshape((X_val.shape[0], -1),
                                   order = 'F')
        reshaped_Y_val = np.squeeze(Y_val)
    
    regr = SVR(kernel = 'linear', C = C_value, 
               epsilon = epsilon_value)
    regr.fit(reshaped_X_train, reshaped_Y_train)
    
    if validation == True:
        Y_pred = regr.predict(reshaped_X_val)
        PCC = pearsonr(reshaped_Y_val, Y_pred)[0]
        SCC = spearmanr(reshaped_Y_val, Y_pred)[0]
        R2 = r2_score(reshaped_Y_val, Y_pred)
    
        return regr, PCC, SCC, R2
    
    else:
        
        return regr
    


def test_model(model, X_test, Y_test):
    '''
    Test/prediction function for the SVR model.
    param X_test: the testing epigenetic inputs.
    param Y_test: the testing RNAseq values
    return: metric results on test set
    '''
    
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


def main(pcc_dict, r2_score_dict, scc_dict, val_pcc_dict,
         val_r2_score_dict, val_scc_dict, gene_dict, 
         num_genes, count, C_value, epsilon_value):
    '''
    "Main" function for model script.
    param pcc_dict: dictionary of PCC metric values
    param r2_score_dict: dictionary of R2 metric values
    param scc_dict: dictionary of SCC metric values
    param val_pcc_dict: dictionary of PCC metric values
    param val_r2_dict: dictionary of R2 metric values
    param val_scc_dict: dictionary of SCC metric values
    param gene_dict: dictionary of gene names and their 
                     position in data.
    param num_genes: the number of genes in a patient's dataset
    param count: the script run count
    param C_value: model hyperparemeter
    param epsilon_value: model hyperparameter
    return: model, indices, metric dictionaries and number of genes 
    '''
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
        save_directory = './cross_patient_regression_using_svm_-_results_-_' + \
        str(now.month) + '-' + str(now.day) + '-' + str(now.year) + '_' + \
        'at_' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)
        print('*'*25)
        print('Using the default save directory:')
        print(f'{save_directory}')
        print('since a directory was not specified.')
        print('*'*25)
        os.makedirs(save_directory, exist_ok = True)
    
    # Indicate True or False for the creation of a validation set. 
    # The script will fit the model accordingly.
    validation = False
    
    # Get file path from command line
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
                                           C_value = C_value,
                                           epsilon_value = epsilon_value,
                                           random_state = random_state,
                                           X_val = X_val, Y_val = Y_val)

        max_val_pcc = PCC
        max_val_r2_score = R2
        max_val_scc = SCC
        

    else:
        model  = train_model(X_train, Y_train, 
                             validation = validation_bool, 
                             C_value = C_value,
                             epsilon_value = epsilon_value,
                             random_state = random_state,
                             X_val = None, Y_val = None)
        
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
    test_PCC, test_SCC, test_R2  = test_model(model, X_test, Y_test)
    print("Test results:")
    print(f"PCC,{test_PCC}")
    print(f"SCC,{test_SCC}")
    print(f"R2,{test_R2}")
    print('*'*25)

    # Create and populate log file
    with open(save_directory + '/svm_cross_patient_regression_gsc_stem_standard_log2_info.csv', 'a') as log:
        log.write('\n' f'{now.strftime("%H:%M on %A %B %d")},')
     
        log.write(f'CURRENT COUNT: {count},C value: {C_value},epsilon value: {epsilon_value},')
        log.write(f'Val PCC: {max_val_pcc},Val SCC: {max_val_scc}, Val R2: {max_val_r2_score},')
        log.write(f'Test PCC: {test_PCC},Test SCC: {test_SCC}, Test R2: {test_R2}')


    
    return pcc_dict, r2_score_dict, scc_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, indices, model

if __name__ == '__main__':

    pcc_dict = {}
    r2_score_dict = {}
    scc_dict = {}
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

    count= 0

    for Cv, ev in product(*param_values): 
        pcc_dict, r2_score_dict, scc_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, indices, model = main(pcc_dict, r2_score_dict, scc_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, count, C_value = Cv, epsilon_value = ev)
        count+=1
    
    
