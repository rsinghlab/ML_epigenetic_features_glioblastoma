'''
This is a Gradient Boosting Regression algorithm model. 
It is intended to be one of the baseline models 
for the cross patient regression analysis.
'''

import sys
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from itertools import product
import random
import datetime

def get_data_patient_1(file_path, indices, gene_dict, num_genes, preprocess, validation):
    '''
    Patient 1's data preperation for model input.
    param file_path: location of patient data
    param indices: location of file used to shuffle gene order
    param gene_dict: dictionary of gene names and their 
                     position in data.
    param num_genes: the number of genes in a patient's dataset
    param preprocess: boolean to determine if function should perform processing
    param validation: boolean to determine if function shoud produce useable validation dataset
    return: X_train, X_val, Y_train, Y_val; where X refers to inputs and Y refers to labels.
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
    Patient 2's data preperation for model input.
    param file_path: location of patient data
    param indices: location of file used to shuffle gene order
    param gene_dict: dictionary of gene names and their 
                     position in data.
    param num_genes: the number of genes in a patient's dataset
    param preprocess: boolean to determine if function should perform processing
    return: X_test, Y_test; where X refers to inputs and Y refers to labels.
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
    param seed: the number to be assigned as the random seed
    return: nothing
    '''

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)

    return None

def train_model(X_train, X_val, Y_train, Y_val, validation, learning_rates, n_estimators, max_depths):
    '''
    Implements and trains a Gradirnt Boosting Regression model.
    param X_train: the training inputs
    param Y_train: the training labels
    param X_val: the validation inputs
    param Y_val: the validation labels
    param validation: boolean to determine if function shoud make use of validation dataset
    param learning_rates: model hyperparemeter
    param n_estimators: model hyperparameter
    param max_depths: model hyperparameter    
    return: trained model and validation metrics if appropriate
    '''

    # Set random seed
    reset_random_seeds(10)
    
    # Reshape data into 2 dimensions.
    reshaped_X_train = X_train.reshape((X_train.shape[0], -1), order = 'F')
    reshaped_X_val = X_val.reshape((X_val.shape[0], -1), order = 'F')
    reshaped_Y_train = np.squeeze(Y_train)
    reshaped_Y_val = np.squeeze(Y_val)
    
    regr = GBR(learning_rate = learning_rates, n_estimators = n_estimators, max_depth = max_depths)
    regr.fit(reshaped_X_train, reshaped_Y_train)
    if validation == True:
        Y_pred = regr.predict(reshaped_X_val)
        PCC = pearsonr(reshaped_Y_val, Y_pred)[0]
        SCC = spearmanr(reshaped_Y_val, Y_pred)[0]
        R2 = r2_score(reshaped_Y_val, Y_pred)
    
        return regr, PCC, SCC, R2
    
    else:
        
        return regr
    


def test_model(model, X_test, Y_test, learning_rates, n_estimators, max_depths):
    '''
    Test/prediction function for Gradient Boosting Regression model.
    param model: trained model
    param X_test: the testing inputs
    param Y_test: the testing labels
    param learning_rates: model hyperparemeter
    param n_estimators: model hyperparameter
    param max_depths: model hyperparameter    
    return: metric results on test set
    '''
        
    # Reshape data into 2 dimensions.
    reshaped_X_test = X_test.reshape((X_test.shape[0], -1), order = 'F')
    reshaped_Y_test = np.squeeze(Y_test)
    
    Y_pred = model.predict(reshaped_X_test)
    PCC = pearsonr(reshaped_Y_test, Y_pred)[0]
    SCC = spearmanr(reshaped_Y_test, Y_pred)[0]
    R2 = r2_score(reshaped_Y_test, Y_pred)
    
    return PCC, SCC, R2

def main(pcc_dict, r2_score_dict, scc_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, count, learning_rates, n_estimators, max_depths):
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
    param learning_rates: model hyperparemeter
    param n_estimators: model hyperparameter
    param max_depths: model hyperparameter
    return: model, datasets, indices, metric dictionaries, gene names, and number of genes 
    '''
    # Save directory - path where result files and figures are saved
    global save_directory

    if sys.argv[4:]:
        # Save path given by the user in the 4th argument to the global variable
        save_directory = sys.argv[4]
        # Create the given directory
        print(f'Using {save_directory} as the save directory for experiment output.')
        os.makedirs(save_directory, exist_ok=True)

    else:
        save_directory = './cross_patient_regression_using_gbr_results_and_figures'
        print('Using the default save directory:')
        print('./cross_patient_regression_using_gbr_results_and_figures')
        print('since a directory was not specified.')
        os.makedirs(save_directory, exist_ok=True)
    
    # Indicate True or False for the creation of a validation set. The script will fit the model accordingly.
    validation = False
    
    # Get file path from command line
    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]
    indices = sys.argv[3]
        
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
        model, PCC, SCC, R2  = train_model(X_train, X_val, Y_train, Y_val, validation=validation_bool, learning_rates = learning_rates, n_estimators = n_estimators, max_depths = max_depths)

        max_val_pcc = PCC
        max_val_r2_score = R2
        max_val_scc = SCC
        

    else:
        model = train_model(X_train, X_val, Y_train, Y_val, validation=validation_bool, learning_rates = learning_rates, n_estimators = n_estimators, max_depths = max_depths)
        max_val_pcc = 'TRAINING SET ONLY'
        max_val_r2_score = 'TRAINING SET ONLY'
        max_val_scc = 'TRAINING SET ONLY'


    val_pcc_dict[count] = max_val_pcc
    val_r2_score_dict[count] = max_val_r2_score
    val_scc_dict[count] = max_val_scc


    print("CURRENT COUNT:", count, "\n learning rate: ", learning_rates, "\n n estimator: ", n_estimators, "\n max depth: ", max_depths,"\n Val PCC: ", max_val_pcc, "\n Val SCC: ", max_val_scc, "\n Val R2: ", max_val_r2_score)        

    print('*'*25)
    print("Evaluating model...")
    print('*'*25)
    test_PCC, test_SCC, test_R2  = test_model(model, X_test, Y_test, learning_rates = learning_rates, n_estimators = n_estimators, max_depths = max_depths)
    print("Test results:")
    print(f"PCC,{test_PCC}")
    print(f"SCC,{test_SCC}")
    print(f"R2,{test_R2}")
    print('*'*25)



    now = datetime.datetime.now()
    # Script log file.
    with open(save_directory + '/gbr_cross_patient_regression_gsc_stem_standard_log2_info.csv', 'a') as log:
        log.write('\n' f'{now.strftime("%H:%M on %A %B %d")},')
     
        log.write(f'CURRENT COUNT: {count},learning rate: {learning_rates},n estimator: {n_estimators}, max depth: {max_depths},')
        log.write(f'Val PCC: {max_val_pcc},Val SCC: {max_val_scc}, Val R2: {max_val_r2_score},')
        log.write(f'Test PCC: {test_PCC},Test SCC: {test_SCC}, Test R2: {test_R2}')


    
    return pcc_dict, r2_score_dict, scc_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, X_train, X_val, X_test, Y_train, Y_val, Y_test, indices, model

if __name__ == '__main__':

    pcc_dict = {}
    r2_score_dict = {}
    scc_dict = {}
    val_pcc_dict = {}
    val_r2_score_dict = {}
    val_scc_dict = {}
    gene_dict = {}
    num_genes = 0

    
    
    # Hyperparameter combination that produces the best validation metrics
    parameters = dict(learning_rates = [0.06], n_estimators = [90], max_depths = [4])
    
    
    
    # Hyperparameter grid search
    #parameters = dict(learning_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], n_estimators = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], max_depths = [1, 2, 3, 4])

    
       
    param_values = [v for v in parameters.values()]

    count=0

    for lr, ne, md in product(*param_values): 
        pcc_dict, r2_score_dict, scc_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, X_train, X_val, X_test, Y_train, Y_val, Y_test, indices, model = main(pcc_dict, r2_score_dict, scc_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, count, learning_rates = lr, n_estimators = ne, max_depths = md)
        count+=1
 
