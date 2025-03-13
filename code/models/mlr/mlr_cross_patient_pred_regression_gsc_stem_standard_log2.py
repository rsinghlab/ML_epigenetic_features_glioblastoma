'''
This is a multiple linear regression model 
script composed using the Statsmodels 
api and configured for cross-patient prediction.

The script input is two patient datafiles 
and an index file (all in numpy format). 

The model is trained and validated (if applicable) on 
the first position datafile. The model is then 
tested on the second position file.
'''
import sys
import numpy as np
import os
import statsmodels.api as sm
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from itertools import product
import random
import datetime
import math

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
        combined_diff = np.load(file_path, allow_pickle = True)
        
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
        
        # Shuffle the data
        # ind = np.arange(0, num_genes)
        # np.random.shuffle(ind)
        ind = np.load(indices, allow_pickle=True)

        
        if validation == True:
            # HYPERPARAMETER TUNING SPLITS
            # Create train (70%), validation (30%).
            train_ind = ind[0: int(0.7*num_genes)]
            val_ind = ind[int(0.7*num_genes):]
        
        else:
            # TESTING SPLITS
            # The training set will have 99% of the 
            # patient 1 data to train the model.
            # The validation set is reduced to 1% but 
            # still present to not break the function.
            train_ind = ind
            #train_ind = ind[0: int(0.99*num_genes)]
            val_ind = ind[int(0.99*num_genes):]

        
        X_train = X[train_ind]
        X_val = X[val_ind]
        

        Y_train = Y[train_ind]
        Y_val = Y[val_ind]
        

        # List of all datasets after split operation.
        # Standardization ONLY on input variables.
        datasets = [X_train, X_val]

        # Perform calculation on each column of the seperate train, validation and test sets. 
        for dataset in datasets:
            for i in range(dataset.shape[2]):
                # Standardize the column values.
                dataset[:, :, i] = (dataset[:, :, i] - np.mean(dataset[:, :, i])) / np.std(dataset[:, :, i], ddof = 1)
                
        np.save("X_cross_patient_regression_patient_1_stem_standard_log2_train",
                X_train, 
                allow_pickle = True)
        np.save("X_cross_patient_regression_patient_1_stem_standard_log2_val",
                X_val,
                allow_pickle = True)
        
        np.save("Y_cross_patient_regression_patient_1_stem_standard_log2_train",
                Y_train, 
                allow_pickle = True)
        np.save("Y_cross_patient_regression_patient_1_stem_standard_log2_val",
                Y_val, 
                allow_pickle = True)
        
    
    else:
        X_train = np.load("X_cross_patient_regression_patient_1_stem_standard_log2_train.npy", 
                          allow_pickle = True)
        X_val = np.load("X_cross_patient_regression_patient_1_stem_standard_log2_val.npy", 
                        allow_pickle = True)
        
        Y_train = np.load("Y_cross_patient_regression_patient_1_stem_standard_log2_train.npy", 
                          allow_pickle = True)
        Y_val = np.load("Y_cross_patient_regression_patient_1_stem_standard_log2_val.npy", 
                        allow_pickle = True)
        
    
        gene_dict = gene_dict
        num_genes = num_genes


    return X_train, X_val, Y_train, Y_val, gene_dict, num_genes

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

        # Shuffle the data
        #ind = np.arange(0, num_genes)
        # np.random.shuffle(ind)
        ind = np.load(indices, allow_pickle = True)

        # Collect the indices that need to be deleted from the array
        # because the number of genes is lower than the 20,015 due to 
        # experiments keeping only the expressed genes in combined_diff
        # or different numbers of genes in various test datasets. 
        print(combined_diff.shape)
        indexes = np.where(ind > X.shape[0] - 1)
        patient2_ind = np.delete(ind, indexes)
        print(patient2_ind.shape)

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
            #10/28/24
            #for i in [0]: ### NO standardization on columns 1,2,and 3 - CTCF, ATAC and RNA Pol II for Omnibus cell testing ONLY.

                # Standardize the column values.
                dataset[:, :, i] = (dataset[:, :, i] - np.mean(dataset[:, :, i])) / np.std(dataset[:, :, i], ddof = 1)
                

        np.save("X_cross_patient_regression_patient_2_stem_standard_log2_test",
                X_test, 
                allow_pickle=True)
        np.save("Y_cross_patient_regression_patient_2_stem_standard_log2_test",
                Y_test, 
                allow_pickle=True)

    else:
        X_test = np.load("X_cross_patient_regression_patient_2_stem_standard_log2_test.npy", 
                         allow_pickle=True)
        Y_test = np.load("Y_cross_patient_regression_patient_2_stem_standard_log2_test.npy", 
                         allow_pickle=True)

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
    
def train_model(X_train, X_val, Y_train, Y_val, 
                validation, alpha, L1_wt):
    """
    Implements and trains a MLR model.
    param X_train: the training inputs
    param Y_train: the training labels
    param X_val: the validation inputs
    param Y_val: the validation labels
    return: a trained model and training history
    """

    #random_state = count
    random_state = 10
    
    # Set random seed
    reset_random_seeds(random_state)
    
    # Reshape data into 2 dimensions.
    reshaped_X_train = X_train.reshape((X_train.shape[0], -1), 
                                       order = 'F')
    reshaped_X_val = X_val.reshape((X_val.shape[0], -1), 
                                   order = 'F')
    reshaped_Y_train = np.squeeze(Y_train)
    reshaped_Y_val = np.squeeze(Y_val)
        
    # Version of model with no regularization.
    # est = sm.OLS(reshaped_Y_train, reshaped_X_train).fit()
    # Version of model with regularization and hyperparameters tuned.
    est = sm.OLS(reshaped_Y_train,
                 reshaped_X_train).fit_regularized(method='elastic_net',
                                                   alpha = alpha,
                                                   L1_wt = L1_wt)
    if validation == True:
        Y_pred = est.predict(reshaped_X_val)
        PCC = pearsonr(reshaped_Y_val, Y_pred)[0]
        SCC = spearmanr(reshaped_Y_val, Y_pred)[0]
        R2 = r2_score(reshaped_Y_val, Y_pred)
    
        return est, PCC, SCC, R2
    
    else:
        
        return est
    
def test_model(model, X_test, Y_test):
    """
    Implements and trains a MLR model.
    param X_test: the testing inputs
    param Y_test: the testing labels
    return: testing metric results
    """
    
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

def main(loss_dict, pcc_dict, r2_score_dict, scc_dict, 
         val_loss_dict, val_pcc_dict, val_r2_score_dict, 
         val_scc_dict, gene_dict, 
         num_genes, count, alpha, L1_wt):

    # Save directory - path where result files and figures are saved
    global save_directory

    if sys.argv[4:]:
        # Save path given by the user in the 4th argument to the global variable
        save_directory = sys.argv[4]
        # Create the given directory
        print('*'*25)
        print(f'Using {save_directory} as the save directory for experiment output.')
        print('*'*25)
        os.makedirs(save_directory, exist_ok = True)

    else:
        save_directory = './cross_patient_regression_using_mlr_results/'
        print('*'*25)
        print('Using the default save directory:')
        print('./cross_patient_regression_using_mlr_results')
        print('since a directory was not provided.')
        print('*'*25)
        os.makedirs(save_directory, exist_ok = True)
    
    # Indicate True or False for the creation of a validation set. The script will fit the model accordingly.
    validation = False
    
    # Get file path from command line
    # NOTE: file_path_1 is the datafile for training and validating the model.    
    # NOTE: file_path_2 is the datafile for testing the model.
    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]
    indices = sys.argv[3]
        
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


    # Processing data for patient 1 file to produce train and validation sets.
    #X_train, X_val, X_test, Y_train, Y_val, Y_test, gene_dict, num_genes = get_data_patient_1(file_path, indices, gene_dict, num_genes, preprocess = preprocess_bool)
    X_train, X_val, Y_train, Y_val, gene_dict, num_genes = get_data_patient_1(file_path_1, indices, gene_dict, num_genes, preprocess = preprocess_bool, validation = validation_bool)

    # Processing data for patient 2 file to produce test set.
    X_test, Y_test, gene_dict, num_genes = get_data_patient_2(file_path_2, indices, gene_dict, num_genes, preprocess = preprocess_bool)

    # Call train_model() to train the model
    print("Training model...")
    
    if validation_bool == True:
        model, PCC, SCC, R2  = train_model(X_train, 
                                           X_val, 
                                           Y_train, 
                                           Y_val, 
                                           validation = validation_bool, 
                                           alpha = alpha, 
                                           L1_wt = L1_wt)
        max_val_pcc = PCC
        max_val_r2_score = R2
        max_val_scc = SCC
        

    else:
        model  = train_model(X_train, 
                             X_val, 
                             Y_train, 
                             Y_val, 
                             validation = validation_bool, 
                             alpha = alpha, 
                             L1_wt = L1_wt)
        max_val_pcc = 'TRAINING SET ONLY'
        max_val_r2_score = 'TRAINING SET ONLY'
        max_val_scc = 'TRAINING SET ONLY'


    val_pcc_dict[count] = max_val_pcc
    val_r2_score_dict[count] = max_val_r2_score
    val_scc_dict[count] = max_val_scc


    print("CURRENT COUNT:", count, "\n Val PCC: ", max_val_pcc , "\n Val SCC: ", max_val_scc, "\n Val R2: ", max_val_r2_score)    

    print('*'*25)
    print("Evaluating model...")
    print('*'*25)
    test_PCC, test_SCC, test_R2  = test_model(model, X_test, Y_test)
    print("Test results:")
    print(f"PCC,{test_PCC}")
    print(f"SCC,{test_SCC}")
    print(f"R2,{test_R2}")
    print('*'*25)



    now = datetime.datetime.now()
    with open(save_directory + '/mlr_cross_patient_regression_gsc_stem_standard_log2_info.csv', 'a') as log:
        log.write('\n' f'{now.strftime("%H:%M on %A %B %d")},')     
        log.write(f'CURRENT COUNT: {count}, alpha: {alpha}, L1 weight: {L1_wt},')
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
  
    
    # Hyperparameter combination that produces the best validation metrics
    parameters = dict(alpha = [5000], L1_wt = [0.0])
    

    # Hyperparameter grid search
    #parameters = parameters = dict(alpha = [0, 1, 5, 10, 25, 50, 75, 100, 200, 400, 800, 1600, 2500, 3200, 5000], L1_wt = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    
       
    param_values = [v for v in parameters.values()]

    count=0

    for al, l1 in product(*param_values): 
        loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict, gene_dict, num_genes, X_train, X_val, X_test, Y_train, Y_val, Y_test, indices, model = main(loss_dict, pcc_dict, r2_score_dict, 
        scc_dict, val_loss_dict, val_pcc_dict, 
        val_r2_score_dict, val_scc_dict, gene_dict, 
        num_genes, count, alpha = al, L1_wt = l1)
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
    