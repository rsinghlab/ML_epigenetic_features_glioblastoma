import sys
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
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Input, Bidirectional, GRU, LSTM, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn import preprocessing
from tqdm import tqdm
from tensorflow.keras import backend as K
#from scipy.stats import pearsonr
from scipy.stats import spearmanr
from itertools import product
import random
import datetime
import math
import shap
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Run the model with additional system arguments values listed below:
# train_val_patient = sys.argv[1] # GSC1 / GSC2
# test_patient = sys.argv[2] # GSC1 / GSC2
# model_structure = sys.argv[3] # LSTM / biLSTM / LSTM-concatenated / GRU / biGRU
# preprocess = sys.argv[4] # true / false
# mode = sys.argv[5] # hyperparameter-tuning / testing / visualizing
# seed = sys.argv[6] # random seed
# attention = sys.argv[7] # attention


def get_data_train_val(data_path, index_path, gene_dict, num_genes, preprocess):
    '''
    Returns X_train, X_val, Y_train, Y_val; where X refers to inputs and Y refers to labels.
    '''
    
    if preprocess:
       
        print("Loading train/val dataset...")
        # Col 1 = gene names, 2 = bin number, 3-6 = features, 7 = labels
        combined_diff = np.load(data_path, allow_pickle=True)
        
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
        
        # Shuffle the data
        # ind = np.arange(0, num_genes)
        # np.random.shuffle(ind)
        ind = np.load(index_path, allow_pickle=True)        
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

        if sys.argv[5] == "hyperparameter-tuning":
            # HYPERPARAMETER TUNING SPLITS
            # Create train (70%), validation (30%).
            train_ind = patient1_ind[0: int(0.7*num_genes)]
            val_ind = patient1_ind[int(0.7*num_genes):]
        elif sys.argv[5] == "testing":
            # TESTING SPLITS
            # The training set will have 100% of the patient 1 data to train the model.
            # The validation set is reduced to 1% but ket to not break the script.
            train_ind = patient1_ind
            val_ind = patient1_ind[int(0.99*num_genes):]
            
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
                
        np.save("X_cross_patient_regression_standard_log2_"+sys.argv[1]+"_"+sys.argv[5]+"_train", X_train, allow_pickle=True)
        np.save("X_cross_patient_regression_standard_log2_"+sys.argv[1]+"_"+sys.argv[5]+"_val", X_val, allow_pickle=True)
        
        np.save("Y_cross_patient_regression_standard_log2_"+sys.argv[1]+"_"+sys.argv[5]+"_train", Y_train, allow_pickle=True)
        np.save("Y_cross_patient_regression_standard_log2_"+sys.argv[1]+"_"+sys.argv[5]+"_val", Y_val, allow_pickle=True)
        
    
    else:
        if sys.argv[5] == "visualizing":
            X_train = np.load("X_cross_patient_regression_standard_log2_"+sys.argv[1]+"_testing_train.npy", allow_pickle=True)
            X_val = np.load("X_cross_patient_regression_standard_log2_"+sys.argv[1]+"_testing_val.npy", allow_pickle=True)
        
            Y_train = np.load("Y_cross_patient_regression_standard_log2_"+sys.argv[1]+"_testing_train.npy", allow_pickle=True)
            Y_val = np.load("Y_cross_patient_regression_standard_log2_"+sys.argv[1]+"_testing_val.npy", allow_pickle=True)
        else:
            X_train = np.load("X_cross_patient_regression_standard_log2_"+sys.argv[1]+"_"+sys.argv[5]+"_train.npy", allow_pickle=True)
            X_val = np.load("X_cross_patient_regression_standard_log2_"+sys.argv[1]+"_"+sys.argv[5]+"_val.npy", allow_pickle=True)
        
            Y_train = np.load("Y_cross_patient_regression_standard_log2_"+sys.argv[1]+"_"+sys.argv[5]+"_train.npy", allow_pickle=True)
            Y_val = np.load("Y_cross_patient_regression_standard_log2_"+sys.argv[1]+"_"+sys.argv[5]+"_val.npy", allow_pickle=True)
        
        gene_dict = gene_dict
        num_genes = num_genes


    return X_train, X_val, Y_train, Y_val, gene_dict, num_genes


def get_data_test(data_path, index_path, gene_dict, num_genes, preprocess):
    '''
    Returns X_test, Y_test; where X refers to inputs and Y refers to labels.
    '''

    if preprocess:
        print("Loading test dataset...")
        # Col 1 = gene names, 2 = bin number, 3-6 = features, 7 = labels
        combined_diff = np.load(data_path, allow_pickle=True)
        
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
   
            # Set corresponding value to be first bin's RNAseq value (since all 100 bins
            # have the same value)
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

        # Shuffle the data
        #ind = np.arange(0, num_genes)
        # np.random.shuffle(ind)
        ind = np.load(index_path, allow_pickle=True)
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

        # Splits for this patient data can be adjusted here.
        #train_ind = ind[0: int(0.7*num_genes)]
        #val_ind = ind[int(0.7*num_genes):int(0.85*num_genes)]
        #test_ind = ind[int(0.85*num_genes):]


        # NOTE: For now use entire dataset for test set.
        test_ind = patient2_ind

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
            for i in range(dataset.shape[2]):
                # Standardize the column values.
                dataset[:, :, i] = (dataset[:, :, i] - np.mean(dataset[:, :, i])) / np.std(dataset[:, :, i], ddof = 1)


        # sys.argv[5] should not matter
        np.save("X_cross_patient_regression_standard_log2_"+sys.argv[2]+"_test", X_test, allow_pickle=True)
        np.save("Y_cross_patient_regression_standard_log2_"+sys.argv[2]+"_test", Y_test, allow_pickle=True)

    else:
        X_test = np.load("X_cross_patient_regression_standard_log2_"+sys.argv[2]+"_test.npy", allow_pickle=True)
        Y_test = np.load("Y_cross_patient_regression_standard_log2_"+sys.argv[2]+"_test.npy", allow_pickle=True)

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
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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
    Calculate Spearman Correlation Rank Coefficient (SCRC) as a metric for the model.
    '''
    spearman_value = tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout = tf.float32)

    return spearman_value


def train_model(X_train, X_val, Y_train, Y_val, params):
    """
    Implements and trains the model
    param X_train: the training inputs
    param Y_train: the training labels
    param X_val: the validation inputs
    param Y_val: the validation labels
    return: a trained model and training history
    """

    batch_size=params[0]
    learning_rate=params[1]
    unit_size=params[2]
    dropout_rate=params[3]
    hidden_size=params[4]
    epochs = params[5]
    metrics = [pearson_r, tfa.metrics.r_square.RSquare(), spearman_r]

    if sys.argv[3] == "LSTM-concatenated":

        class LSTM_with_sum_steps(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.lstm = LSTM(unit_size, dropout=dropout_rate, input_shape=(50, 4), return_state=True, return_sequences=True)
                self.dense_1 = Dense(hidden_size, activation='relu')
                self.dense_2 = Dense(1)

            def call(self, inputs, training=False):
                all_state_h_train, state_h_train, state_c_train = self.lstm(inputs)
                concate_h_train = tf.reshape(all_state_h_train, [-1,all_state_h_train.shape[1]*all_state_h_train.shape[2]])
                concate_h_train = self.dense_1(concate_h_train)
                return self.dense_2(concate_h_train)
        
        model = LSTM_with_sum_steps()
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate), metrics=metrics)

    else: # LSTM / biLSTM/ GRU / biGRU

        if sys.argv[7] == "attention":
            class rec_attention(tf.keras.Model): # iput: bx50xr / bx4xr; oput: byr
                # attention with bin context vector per HM and HM context vector
                def __init__(self, hm):
                    super(rec_attention, self).__init__()
                    if "bi" in sys.argv[3]:
                        self.num_directions = 2
                    else:
                        self.num_directions = 1
                    self.bin_rep_size = unit_size * self.num_directions
                    # if (hm == False): # bin-level
                        # self.bin_rep_size = unit_size * self.num_directions
                    # else: # HM-level
                        # self.bin_rep_size = unit_size
                    self.bin_context_vector = tf.Variable(tf.random.uniform([self.bin_rep_size,1],-0.1,0.1),trainable=True)
                    self.softmax = tf.keras.layers.Softmax(axis=1)

                def call(self, iput):
                    alpha = self.softmax(tf.einsum('bsr,ry->bs', iput, self.bin_context_vector))
                    repres = tf.einsum("bys,bsr->byr", tf.expand_dims(alpha, 1), iput)
                    return repres, alpha

            class recurrent_encoder(tf.keras.Model): # iput: bx50xr / bx4xr; oput: byr
                # modular LSTM encoder
                def __init__(self, hm, n_bins, bin_size):
                    super(recurrent_encoder,self).__init__()
                    self.iput_shape = (n_bins, bin_size) # 50x1 or 4xr
                    if "bi" in sys.argv[3]:
                        self.num_directions = 2
                    else:
                        self.num_directions = 1
                    # if (hm == False): # bin-level
                        # self.bin_rnn_size = unit_size
                        # self.bin_rep_size = unit_size * self.num_directions
                    # else: # HM-level
                        # self.bin_rnn_size = unit_size / 2
                        # self.bin_rep_size = unit_size
                    # self.bin_rep_size = int(self.bin_rnn_size * self.num_directions)
                    self.bin_rep_size = unit_size * self.num_directions
                    if sys.argv[3] == "LSTM":
                        self.rnn = LSTM(unit_size, dropout=dropout_rate, input_shape=self.iput_shape, return_state=True, return_sequences=True)
                    elif sys.argv[3] == "biLSTM":
                        self.rnn = Bidirectional(LSTM(unit_size, dropout=dropout_rate, input_shape=self.iput_shape, return_state=True, return_sequences=True))
                    elif sys.argv[3] == "GRU":
                        self.rnn = GRU(unit_size, dropout=dropout_rate, input_shape=self.iput_shape, return_state=True, return_sequences=True)
                    elif sys.argv[3] == "biGRU":
                        self.rnn = Bidirectional(GRU(unit_size, dropout=dropout_rate, input_shape=self.iput_shape, return_state=True, return_sequences=True))
                    self.bin_attention = rec_attention(hm)
                
                def outputlength(self):
                    return self.bin_rep_size
                
                def call(self, single_hm):
                    # bin_output, hidden = self.rnn(single_hm,hidden)
                    whole_seq_out = self.rnn(single_hm)[0] # bx50xr / bx4xr
                    hm_rep, bin_alpha = self.bin_attention(whole_seq_out) # byr
                    return hm_rep, bin_alpha
                
            class att_chrome(tf.keras.Model): # iput: bsh; oput: bx
                def __init__(self):
                    super(att_chrome,self).__init__()
                    self.rnn_hms = [recurrent_encoder(False, 50, 1) for i in range(4)]
                    self.opsize = self.rnn_hms[0].outputlength()
                    self.hm_level_rnn = recurrent_encoder(True, 4, self.opsize)
                    self.linear = Dense(1)

                def call(self, iput):

                    bin_level_rep = [] # bx4xr
                    bin_level_atten = []

                    for hm, hm_encdr in enumerate(self.rnn_hms):

                        hm_feature= tf.expand_dims(iput[:,:,hm],2) # bx50x1
                        rep, atten = hm_encdr(hm_feature) # bx1xr

                        if hm == 0:
                            bin_level_rep = rep
                            bin_level_atten = atten
                        else:
                            bin_level_rep = tf.concat([bin_level_rep, rep], 1)
                            bin_level_atten = tf.concat([bin_level_atten, atten], 1)
                    
                    hm_level_rep, hm_level_atten = self.hm_level_rnn(bin_level_rep)
                    hm_level_rep = tf.squeeze(hm_level_rep, 1)
                    pred = self.linear(hm_level_rep)
                    return pred
            model = att_chrome()
            model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate), metrics=metrics)

        elif sys.argv[7] == "attention-alt":
            class rec_attention(tf.keras.Model): # iput: bx50xr / bx4xr; oput: byr
                # attention with bin context vector per HM and HM context vector
                def __init__(self, hm):
                    super(rec_attention, self).__init__()
                    if "bi" in sys.argv[3]:
                        self.num_directions = 2
                    else:
                        self.num_directions = 1
                    self.bin_rep_size = unit_size * self.num_directions
                    # if (hm == False): # bin-level
                        # self.bin_rep_size = unit_size * self.num_directions
                    # else: # HM-level
                        # self.bin_rep_size = unit_size
                    self.bin_context_vector = tf.Variable(tf.random.uniform([self.bin_rep_size,1],-0.1,0.1),trainable=True)
                    self.softmax = tf.keras.layers.Softmax(axis=1)

                def call(self, iput):
                    alpha = self.softmax(tf.einsum('bsr,ry->bs', iput, self.bin_context_vector))
                    repres = tf.einsum("bys,bsr->byr", tf.expand_dims(alpha, 1), iput)
                    return repres, alpha
                
            class recurrent_encoder(tf.keras.Model): # iput: bx50xr / bx4xr; oput: byr
                # modular LSTM encoder
                def __init__(self, hm, n_bins, bin_size):
                    super(recurrent_encoder,self).__init__()
                    self.iput_shape = (n_bins, bin_size) # 50x4
                    if "bi" in sys.argv[3]:
                        self.num_directions = 2
                    else:
                        self.num_directions = 1
                    self.bin_rep_size = unit_size * self.num_directions
                    if sys.argv[3] == "LSTM":
                        self.rnn = LSTM(unit_size, dropout=dropout_rate, input_shape=self.iput_shape, return_state=True, return_sequences=True)
                    elif sys.argv[3] == "biLSTM":
                        self.rnn = Bidirectional(LSTM(unit_size, dropout=dropout_rate, input_shape=self.iput_shape, return_state=True, return_sequences=True))
                    elif sys.argv[3] == "GRU":
                        self.rnn = GRU(unit_size, dropout=dropout_rate, input_shape=self.iput_shape, return_state=True, return_sequences=True)
                    elif sys.argv[3] == "biGRU":
                        self.rnn = Bidirectional(GRU(unit_size, dropout=dropout_rate, input_shape=self.iput_shape, return_state=True, return_sequences=True))
                    self.bin_attention = rec_attention(hm)
                
                def outputlength(self):
                    return self.bin_rep_size
                
                def call(self, single_hm):
                    # bin_output, hidden = self.rnn(single_hm,hidden)
                    whole_seq_out = self.rnn(single_hm)[0] # bx50xr / bx4xr
                    hm_rep, bin_alpha = self.bin_attention(whole_seq_out) # byr
                    return hm_rep, bin_alpha
            
            class alt_chrome(tf.keras.Model): # iput: bsh; oput: bx
                def __init__(self):
                    super(alt_chrome,self).__init__()
                    self.rnn = recurrent_encoder(True, 50, 4)
                    self.linear = Dense(1)

                def call(self, iput):
                    all_rep, all_atten = self.rnn(iput)
                    all_rep = tf.squeeze(all_rep, 1)
                    pred = self.linear(all_rep)
                    return pred
            model = alt_chrome()
            model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate), metrics=metrics)

        else:

            if sys.argv[3] == "biGRU":
                model = Sequential()
                model.add(Bidirectional(GRU(unit_size, dropout=dropout_rate, input_shape=(50, 4))))
            elif sys.argv[3] == "biLSTM":
                model = Sequential()
                model.add(Bidirectional(LSTM(unit_size, dropout=dropout_rate, input_shape=(50, 4))))
            elif sys.argv[3] == "GRU":
                model = Sequential()
                model.add(GRU(unit_size, dropout=dropout_rate, input_shape=(50, 4)))
            elif sys.argv[3] == "LSTM":
                model = Sequential()
                model.add(LSTM(unit_size, dropout=dropout_rate, input_shape=(50, 4)))
            
            model.add(Dense(hidden_size, activation='relu'))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate), metrics=metrics)

    if sys.argv[5] == "hyperparameter-tuning": # hyperparameter tuning
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, shuffle=False)
    elif sys.argv[5] == "testing" or "visualizing": # testing
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
    
    return model, history


def visualize_pcc(history, validation, count, random_seed):
    '''
    Visualization of the Pearson Correlation Coefficient (PCC)
    values over all epochs.
    returns: Nothing. A visualization will appear and/or be saved.
    '''
    plt.close()
    plt.plot(history.history['pearson_r'], color = 'peru')
    if validation:
        plt.plot(history.history['val_pearson_r'], color = 'aqua')
    #plt.title(name)
    plt.title('PCC - Cross Patient Regression CNN (Standardized input/Log2 target)')
    plt.ylabel('PCC')
    plt.xlabel('epoch')
    if validation:
        plt.legend(['train pcc', 'validation pcc'], loc='upper left')
    else:
        plt.legend(['train pcc'], loc='upper left')
    if (sys.argv[5] == "hyperparameter-tuning" or sys.argv[5] == "visualizing"):
        plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + sys.argv[6] + "_" + sys.argv[7] + '/pcc_' + str(count) + '.png')
    else:
        plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_" + sys.argv[7] + '/pcc_' + str(count) + '.png')


def visualize_loss(history, validation, count, random_seed):
    '''
    Visualization of model loss over all epochs.
    returns: Nothing. A visualization will appear and/or be saved.
    '''
    plt.close()
    plt.plot(history.history['loss'], color = 'darkgoldenrod')
    if validation:
        plt.plot(history.history['val_loss'], color = 'dodgerblue')
    plt.title("Loss - Cross Patient Regression CNN (Standardized input/Log2 target)")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if validation:
        plt.legend(['train loss', 'validation loss'], loc='upper left')
    else:
        plt.legend(['train loss'], loc='upper left')
    if (sys.argv[5] == "hyperparameter-tuning" or sys.argv[5] == "visualizing"):
        plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + sys.argv[6] + "_" + sys.argv[7] + '/loss_' + str(count) + '.png')
    else:
        plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_" + sys.argv[7] + '/loss_' + str(count) + '.png')


def visualize_scc(history, validation, count, random_seed):
    '''
    Visualization of the Spearman Correlation Rank Coefficient (SCC)
    values over all epochs.
    returns: Nothing. A visualization will appear and/or be saved.
    '''
    plt.close()
    plt.plot(history.history['spearman_r'], color = 'plum')
    if validation:
        plt.plot(history.history['val_spearman_r'], color = 'tan')
    plt.title("SCC - Regression CNN (Standardized input/Log2 target) Realigned GSC Stem")
    plt.ylabel('SCC')
    plt.xlabel('epoch')
    if validation:
        plt.legend(['train scc', 'validation scrc'], loc='upper left')
    else:
        plt.legend(['train scc'], loc='upper left')
    if (sys.argv[5] == "hyperparameter-tuning" or sys.argv[5] == "visualizing"):
        plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + sys.argv[6] + "_" + sys.argv[7] + '/scc_' + str(count) + '.png')
    else:
        plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_" + sys.argv[7] + '/scc_' + str(count) + '.png')

        
def visualize_rsquare(history, validation, count, random_seed):
    '''
    Visualize the RSquare metric accross all epochs.

    returns: Nothing. The plot will appear and/or be saved.
    '''
    plt.close()
    plt.plot(history.history['r_square'], color = 'forestgreen')
    if validation:
        plt.plot(history.history['val_r_square'], color = 'gainsboro')
    plt.title('R2 - Cross Patient Regression CNN (Standardized input/Log2 target)')
    plt.ylabel('R2')
    plt.xlabel('epoch')
    if validation:
        plt.legend(['train R2', 'validation R2'], loc = 'upper left')
    else:
        plt.legend(['train R2'], loc = 'upper left')
    if (sys.argv[5] == "hyperparameter-tuning" or sys.argv[5] == "visualizing"):
        plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + sys.argv[6] + "_" + sys.argv[7] + '/rsquare_' + str(count) + '.png')
    else:
        plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_" + sys.argv[7] + '/rsquare_' + str(count) + '.png')
    

def visualize_bin_attention(bin_attens, random_seed, gene_index):
  plt.close()
  fig = plt.figure(figsize=(25,2))
  axes = plt.subplot()
  
  # using the matshow() function
  caxes = axes.matshow(bin_attens, cmap='bone')
  plt.xticks(range(50), np.arange(50))
  plt.yticks(range(4), ["H2K23ac", "CTCF", "ATAC", "RNAPol2"])

  divider = make_axes_locatable(axes)
  cax = divider.append_axes("right", size="2%", pad="2%")

  plt.colorbar(caxes, cax=cax)
  plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + random_seed + "_" + sys.argv[7] + '/bin-attention_gene-' + str(gene_index) + '.png')


def visualize_hm_attention(hm_atten, random_seed, gene_index):
  plt.close()
  fig = plt.figure(figsize=(0.5,2))
  axes = plt.subplot()
  
  # using the matshow() function
  caxes = axes.matshow(np.transpose(hm_atten), cmap='bone')
  plt.xticks(range(1), [])
  plt.yticks(range(4), ["H2K23ac", "CTCF", "ATAC", "RNAPol2"])

  divider = make_axes_locatable(axes)
  cax = divider.append_axes("right", size="100%", pad="30%")

  plt.colorbar(caxes, cax=cax)
  plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + random_seed + "_" + sys.argv[7] + '/hm-attention_gene-' + str(gene_index) + '.png')


def main(dicts, count, params, gene_dict, num_genes, random_seed):

    # Call get_data() to process the data, preprocess = True will read in processed .npy files,
    # if false then will re-preprocess data
    print("Processing data")
    if count == 0 and sys.argv[4] == "true":
        preprocess_bool = True
    else:
        preprocess_bool = False

    if sys.argv[1] == "GSC1":
        data_path_1 = '../../../data/latest_versions_of_all_raw/gsc1_stem_with_featurecounts_RNAseq_entire_gene.npy'
        
        data_path_2 = '../../../data/latest_versions_of_all_raw/gsc2_stem_with_featurecounts_RNAseq_entire_gene.npy'
        
        
    else:
        data_path_1 = '../../../data/latest_versions_of_all_raw/gsc2_stem_with_featurecounts_RNAseq_entire_gene.npy'
        
        data_path_2 = '../../../data/latest_versions_of_all_raw/gsc1_stem_with_featurecounts_RNAseq_entire_gene.npy'
        
    index_path = '../../../data/ind_shuffle.npy'


    X_train, X_val, Y_train, Y_val, gene_dict, num_genes = get_data_train_val(data_path_1, index_path, gene_dict, num_genes, preprocess = preprocess_bool)
    X_test, Y_test, gene_dict, num_genes = get_data_test(data_path_2, index_path, gene_dict, num_genes, preprocess = preprocess_bool)


    print("Training model")
        
    if sys.argv[5] == "hyperparameter-tuning":

        if not random_seed == "none":
            reset_random_seeds(int(random_seed))

        model, history = train_model(X_train, X_val, Y_train, Y_val, params)

        min_loss = min(history.history['loss'])
        max_pcc = max(history.history['pearson_r'])
        max_r2_score = max(history.history['r_square'])
        max_scc = max(history.history['spearman_r'])
        min_val_loss = min(history.history['val_loss'])
        max_val_pcc = max(history.history['val_pearson_r'])
        max_val_r2_score = max(history.history['val_r_square'])
        max_val_scc = max(history.history['val_spearman_r'])

        dicts[0][count] = min_loss
        dicts[1][count] = max_pcc
        dicts[2][count] = max_r2_score
        dicts[3][count] = max_scc
        dicts[4][count] = min_val_loss
        dicts[5][count] = max_val_pcc
        dicts[6][count] = max_val_r2_score
        dicts[7][count] = max_val_scc

        print("Evaluating model")
        results  = model.evaluate(X_test, Y_test, batch_size=params[0])
        print("Test loss, test PCC, test R2, test SCC: ", results)

        now = datetime.datetime.now()
        # if os.path.exists(sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + sys.argv[6] + ".log"):
        with open(sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + random_seed + "_" + sys.argv[7] + "_" + ".log", 'a') as log:
            log.write('\n' f'{now.strftime("%H:%M on %A, %B %d")}, ')
            log.write(f'CURRENT COUNT: {count}, batch size: {params[0]}, learning rate: {params[1]}, unit size: {params[2]}, dropout rate: {params[3]}, ')
            log.write(f'hidden size: {params[4]}, epoch: {params[5]}, ')
            log.write(f'Min training loss: {min_loss}, Max training PCC: {max_pcc}, Max training SCC: {max_scc}, Max training R2 Score: {max_r2_score}, ')
            log.write(f'Min val loss: {min_val_loss}, Max val PCC: {max_val_pcc}, Max val SCC: {max_val_scc}, Max val R2 Score: {max_val_r2_score}, ')
            log.write(f'test loss: {results[0]}, test PCC: {results[1]}, test SCC: {results[3]}, test R2 Score: {results[2]}')                                                              
    
        if not os.path.isdir("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + sys.argv[6] + "_" + sys.argv[7]):
            os.mkdir("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + sys.argv[6] + "_" + sys.argv[7])
        
        # Visualize model training and validation loss/pcc/r2/scc
        visualize_loss(history, True, count, random_seed)
        visualize_pcc(history, True, count, random_seed)
        visualize_rsquare(history, True, count, random_seed)
        visualize_scc(history, True, count, random_seed)

    elif sys.argv[5] == "testing": # testing # TODO
        
        reset_random_seeds(int(random_seed))

        model, history = train_model(X_train, X_val, Y_train, Y_val, params)

        min_loss = min(history.history['loss'])
        max_pcc = max(history.history['pearson_r'])
        max_r2_score = max(history.history['r_square'])
        max_scc = max(history.history['spearman_r'])

        dicts[0][count] = min_loss
        dicts[1][count] = max_pcc
        dicts[2][count] = max_r2_score
        dicts[3][count] = max_scc

        print("Evaluating model")
        results = model.evaluate(X_test, Y_test, batch_size=params[0])
        print("Test loss, test PCC, test R2, test SCC: ", results)

        now = datetime.datetime.now()
        # if os.path.exists(sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + sys.argv[6] + ".log"):
        with open(sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_" + sys.argv[7] + "_" + ".log", 'a') as log:
            log.write('\n' f'{now.strftime("%H:%M on %A, %B %d")}, ')
            log.write(f'CURRENT COUNT: {count}, batch size: {params[0]}, learning rate: {params[1]}, unit size: {params[2]}, dropout rate: {params[3]}, ')
            log.write(f'hidden size: {params[4]}, epoch: {params[5]}, random seed: {random_seed}, ')
            log.write(f'Min training loss: {min_loss}, Max training PCC: {max_pcc}, Max training SCC: {max_scc}, Max training R2 Score: {max_r2_score}, ')
            log.write(f'test loss: {results[0]}, test PCC: {results[1]}, test SCC: {results[3]}, test R2 Score: {results[2]}')                                                              
    
        if not os.path.isdir("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_" + sys.argv[7]):
            os.mkdir("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_" + sys.argv[7])
        
        # Visualize model training and validation loss/pcc/r2/scc
        visualize_loss(history, False, count, random_seed)
        visualize_pcc(history, False, count, random_seed)
        visualize_rsquare(history, False, count, random_seed)
        visualize_scc(history, False, count, random_seed)
    
    elif sys.argv[5] == "visualizing": # visualize attention map
        
        reset_random_seeds(int(random_seed))

        model, history = train_model(X_train, X_val, Y_train, Y_val, params)

        min_loss = min(history.history['loss'])
        max_pcc = max(history.history['pearson_r'])
        max_r2_score = max(history.history['r_square'])
        max_scc = max(history.history['spearman_r'])

        dicts[0][count] = min_loss
        dicts[1][count] = max_pcc
        dicts[2][count] = max_r2_score
        dicts[3][count] = max_scc

        print("Evaluating model")
        results = model.evaluate(X_test, Y_test, batch_size=params[0])
        print("Test loss, test PCC, test R2, test SCC: ", results)

        now = datetime.datetime.now()
        # if os.path.exists(sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + sys.argv[6] + ".log"):
        with open(sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + random_seed + "_" + sys.argv[7] + "_" + ".log", 'a') as log:
            log.write('\n' f'{now.strftime("%H:%M on %A, %B %d")}, ')
            log.write(f'CURRENT COUNT: {count}, batch size: {params[0]}, learning rate: {params[1]}, unit size: {params[2]}, dropout rate: {params[3]}, ')
            log.write(f'hidden size: {params[4]}, epoch: {params[5]}, random seed: {random_seed}, ')
            log.write(f'Min training loss: {min_loss}, Max training PCC: {max_pcc}, Max training SCC: {max_scc}, Max training R2 Score: {max_r2_score}, ')
            log.write(f'test loss: {results[0]}, test PCC: {results[1]}, test SCC: {results[3]}, test R2 Score: {results[2]}')                                                              
    
        if not os.path.isdir("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + sys.argv[6] + "_" + sys.argv[7]):
            os.mkdir("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + sys.argv[6] + "_" + sys.argv[7])
        
        # Visualize model training and validation loss/pcc/r2/scc
        visualize_loss(history, False, count, random_seed)
        visualize_pcc(history, False, count, random_seed)
        visualize_rsquare(history, False, count, random_seed)
        visualize_scc(history, False, count, random_seed)

        if sys.argv[7] == "attention-alt":
            for i in range(len(X_test)):
                rep, atten = model.rnn(np.expand_dims(X_test[i,:,:], 0))

                plt.close()
                fig = plt.figure(figsize=(25,2))
                axes = plt.subplot()
                caxes = axes.matshow(atten, cmap='bone')
                plt.xticks(range(50), np.arange(50))
                plt.yticks(range(1), [""])
                divider = make_axes_locatable(axes)
                cax = divider.append_axes("right", size="2%", pad="2%")
                plt.colorbar(caxes, cax=cax)
                plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + random_seed + "_" + sys.argv[7] + '/bin-attention_gene-' + str(i) + '.png')


        elif sys.argv[7] == "attention":
            hist = {i:{j:0 for j in range(4)} for i in range(4)}
            scores = np.zeros(4) 
            for i in range(len(X_test)):
                hm1_rep, hm1_atten = model.rnn_hms[0](np.expand_dims(X_test[i,:,0], (0,2)))
                hm2_rep, hm2_atten = model.rnn_hms[1](np.expand_dims(X_test[i,:,1], (0,2)))
                hm3_rep, hm3_atten = model.rnn_hms[2](np.expand_dims(X_test[i,:,2], (0,2)))
                hm4_rep, hm4_atten = model.rnn_hms[3](np.expand_dims(X_test[i,:,3], (0,2)))
                hm_rep = np.concatenate([hm1_rep, hm2_rep, hm3_rep, hm4_rep], axis=1)
                hm_atten = np.squeeze(np.stack([hm1_atten, hm2_atten, hm3_atten, hm4_atten], axis=0))
                rep, atten = model.hm_level_rnn(hm_rep)
                scores += np.array(atten[0])
                atten_argsort = np.argsort(atten)
                for j in range(4):
                    hist[j][3-atten_argsort[0][j]] += 1
                # visualize_bin_attention(hm_atten, sys.argv[6], i)
                # visualize_hm_attention(atten, sys.argv[6], i)
            scores /= len(X_test)

            width = 0.4
            features = ["H2K23ac", "CTCF", "ATAC", "RNAPol2"]
            ranks = [[hist[i][j] for i in range(4)] for j in range(4)]

            plt.close()
            fig = plt.subplots(figsize =(6, 8))
            plt.bar(features, ranks[0], width, color="maroon")
            # for i in range(4):
            #     plt.text(i, ranks[0][i]+0.005, f"{ranks[0][i]}", fontsize=8)
            plt.xlabel("Epigenetic features")
            plt.ylabel("Number of genes")
            plt.title("Attention Distribution on Epigenetic Features - Highest Attention")
            plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + random_seed + "_" + sys.argv[7] + '/hm-attention_x_highest-bar-plot.png')

            plt.close()
            fig = plt.subplots(figsize =(6, 8))
            plt.bar(features, scores, width, color="rosybrown")
            plt.xlabel("Epigenetic features")
            plt.ylabel("Attention Weights")
            plt.title("Average Attention Weights of Epigenetic Features")
            plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + random_seed + "_" + sys.argv[7] + '/hm-attention_x_avg-bar-plot.png')

            plt.close()
            weighted_ranks = np.array([np.array(ranks[i])*(i+1) for i in range(4)])
            print(weighted_ranks)
            print(np.sum(weighted_ranks, axis=0))
            print(np.sum(weighted_ranks, axis=0)/len(X_test))
            fig = plt.subplots(figsize =(6, 8))
            plt.bar(features, np.sum(weighted_ranks, axis=0)/len(X_test), width, color="rosybrown")
            plt.xlabel("Epigenetic features")
            plt.ylabel("Rank")
            plt.title("Average Attention Rank of Epigenetic Features")
            plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + random_seed + "_" + sys.argv[7] + '/hm-attention_x_rank-bar-plot.png')
            print(np.sum(weighted_ranks, axis=0)/len(X_test))


            plt.close()
            width = 0.2
            inds = np.arange(4)
            plt.bar(inds, ranks[0], width, color="maroon", label = "highest attention")
            plt.bar(inds+width, ranks[1], width, color="firebrick", label = "second-highest attention")
            plt.bar(inds+2*width, ranks[2], width, color="indianred", label = "third-highest attention")
            plt.bar(inds+3*width, ranks[3], width, color="lightcoral", label = "lowest attention")
            plt.xlabel("Epigenetic features")
            plt.ylabel("Number of genes")
            plt.title("Attention Distribution on Epigenetic Features")
            plt.xticks(inds+width*1.5, features)
            plt.legend(loc='best')
            plt.savefig("pngs_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[1] + "-to-" + sys.argv[2] + "_seed-" + random_seed + "_" + sys.argv[7] + '/hm-attention_x_bar-plot.png')







    return dicts #, gene_dict, num_genes

if __name__ == '__main__':

    if sys.argv[5] == "hyperparameter-tuning": 
        loss_dict = {}
        val_loss_dict = {}
        pcc_dict = {}
        r2_score_dict = {}
        scc_dict = {}
        val_pcc_dict = {}
        val_r2_score_dict = {}
        val_scc_dict = {}
        dicts = [loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict]

        parameters = dict(batch_size_vals = [512, 1024], learning_rate_vals = [1e-2, 5e-2, 1e-3, 1e-4], 
                          unit_size = [5, 10, 20, 30, 50, 80], dropout_rate = [0.1, 0.3, 0.5], 
                          hidden_layer_size = [200, 100, 50, 20], epoch = [500]) # num_epoch does not matter for tuning
        param_values = [v for v in parameters.values()]

        count = 0

        gene_dict = {}
        num_genes = 0


        for params in product(*param_values): 
            dicts = main(dicts, count, params, gene_dict, num_genes, sys.argv[6])
            count+=1


    elif sys.argv[5] == "testing": # here we hard coded the best-performing hyperparameters
        loss_dict = {}
        pcc_dict = {}
        r2_score_dict = {}
        scc_dict = {}
        gene_dict = {}
        num_genes = 0
        dicts = [loss_dict, pcc_dict, r2_score_dict, scc_dict]

        if sys.argv[7] == "attention":
            if sys.argv[3] == 'GRU':
                if sys.argv[1] == "GSC1": 
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [30], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [80], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500])
                                      #epoch = [100])# use 100 epochs for results collection on new split method  
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [30], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
                                    #epoch = [100]) # use 100 epochs for results collection on new split method  
                    # best params by tuning using GSC2
                    # parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [80], 
                                    # dropout_rate = [0.1], hidden_layer_size = [200], 
                                    # epoch = [100, 200, 300, 500]) 
            elif sys.argv[3] == 'biGRU':
                if sys.argv[1] == "GSC1": 
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [50], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.01], unit_size = [20], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [50], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    # parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.01], unit_size = [20], 
                                    # dropout_rate = [0.1], hidden_layer_size = [200], 
                                    # epoch = [100, 200, 300, 500]) # try different number of epochs
            elif sys.argv[3] == 'LSTM': 
                if sys.argv[1] == "GSC1":
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.01], unit_size = [20], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [10], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.01], unit_size = [20], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [10], 
                                    # dropout_rate = [0.1], hidden_layer_size = [200], 
                                    # epoch = [100, 200, 300, 500]) # try different number of epochs
            elif sys.argv[3] == 'biLSTM':
                if sys.argv[1] == "GSC1": 
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.01], unit_size = [30], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [80], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.01], unit_size = [30], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [80], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
        
        elif sys.argv[7] == "attention-alt":
            if sys.argv[3] == "GRU":
                if sys.argv[1] == "GSC1": 
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [20], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500])
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [50], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) 
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [20], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500])
                    # best params by tuning using GSC2
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [50], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) 
            elif sys.argv[3] == 'biGRU':
                if sys.argv[1] == "GSC1": 
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [80], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [50], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [80], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [50], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
            elif sys.argv[3] == 'LSTM': 
                if sys.argv[1] == "GSC1":
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [10], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.01], unit_size = [30], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [10], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.01], unit_size = [30], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
            elif sys.argv[3] == 'biLSTM': 
                if sys.argv[1] == "GSC1":
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.01], unit_size = [50], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.01], unit_size = [80], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.01], unit_size = [50], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) # try different number of epochs
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.01], unit_size = [80], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) # try different number of epochs
                    
        else:
            if sys.argv[3] == 'GRU':
                if sys.argv[1] == "GSC1": 
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.01], unit_size = [80], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [50], 
                    #                 epoch = [100, 200, 300, 500]) 
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [30], 
                                    dropout_rate = [0.1], hidden_layer_size = [50], 
                                    epoch = [100, 200, 300, 500]) 
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.01], unit_size = [80], 
                                    dropout_rate = [0.1], hidden_layer_size = [50], 
                                    epoch = [100, 200, 300, 500]) 
                    # best params by tuning using GSC2
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [30], 
                                    # dropout_rate = [0.1], hidden_layer_size = [50], 
                                    # epoch = [100, 200, 300, 500]) 
            elif sys.argv[3] == 'biGRU':
                if sys.argv[1] == "GSC1":
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [10], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [50], 
                    #                 epoch = [100, 200, 300, 500]) 
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [50], 
                                    dropout_rate = [0.1], hidden_layer_size = [20], 
                                    epoch = [100, 200, 300, 500])
                elif sys.argv[1] == "GSC2":
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [10], 
                                    dropout_rate = [0.1], hidden_layer_size = [50], 
                                    epoch = [100, 200, 300, 500]) 
                    # best params by tuning using GSC2
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [50], 
                                    # dropout_rate = [0.1], hidden_layer_size = [20], 
                                    # epoch = [100, 200, 300, 500])
            elif sys.argv[3] == 'LSTM':
                if sys.argv[1] == "GSC1":
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [10], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) 
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [80], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) 
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [10], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) 
                    # best params by tuning using GSC2
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.001], unit_size = [80], 
                                    # dropout_rate = [0.1], hidden_layer_size = [200], 
                                    # epoch = [100, 200, 300, 500]) 
            elif sys.argv[3] == 'biLSTM':
                if sys.argv[1] == "GSC1":
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.01], unit_size = [80], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [20], 
                    #                 epoch = [100, 200, 300, 500]) 
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.01], unit_size = [30], 
                                    dropout_rate = [0.1], hidden_layer_size = [20], 
                                    epoch = [100, 200, 300, 500]) 
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.01], unit_size = [80], 
                                    dropout_rate = [0.1], hidden_layer_size = [20], 
                                    epoch = [100, 200, 300, 500]) 
                    # best params by tuning using GSC2
                    # parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.01], unit_size = [30], 
                                    # dropout_rate = [0.1], hidden_layer_size = [20], 
                                    # epoch = [100, 200, 300, 500]) 
            elif sys.argv[3] == 'LSTM-concatenated': # wait
                if sys.argv[1] == "GSC1": 
                    # best params by tuning using GSC1
                    # parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [80], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [20], 
                    #                 epoch = [100, 200, 300, 500])
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.0001], unit_size = [30], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100, 200, 300, 500]) 
                elif sys.argv[1] == "GSC2":
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [80], 
                                    dropout_rate = [0.1], hidden_layer_size = [20], 
                                    epoch = [100, 200, 300, 500])
                    # best params by tuning using GSC2
                    # parameters = dict(batch_size_vals = [512], learning_rate_vals = [0.0001], unit_size = [30], 
                    #                 dropout_rate = [0.1], hidden_layer_size = [200], 
                    #                 epoch = [100, 200, 300, 500]) 
        
        param_values = [v for v in parameters.values()]
        #seeds = list(range(10))
        seeds = [10] # used to facilitate one seed per sub-dataset with review suggested split testing approach
        count = 0
        for seed in seeds:
            for params in product(*param_values): 
                dicts = main(dicts, count, params, gene_dict, num_genes, str(seed))
                #count+=1 #commented out to facilitate one run per seed with review suggested split testing approach


    elif sys.argv[5] == "visualizing": 
        # for best-performing model only
        if sys.argv[7] == "attention":
            if sys.argv[3] == 'GRU':
                if sys.argv[1] == "GSC1": 
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [30], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [200])
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [80], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100])
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [30], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [200])
                    # best params by tuning using GSC2
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [80], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100])
        elif sys.argv[7] == "attention-alt":
            if sys.argv[3] == "GRU":
                if sys.argv[1] == "GSC1": 
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [20], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100])
                elif sys.argv[1] == "GSC2": 
                    # best params by tuning using GSC1
                    parameters = dict(batch_size_vals = [1024], learning_rate_vals = [0.001], unit_size = [20], 
                                    dropout_rate = [0.1], hidden_layer_size = [200], 
                                    epoch = [100])
        param_values = [v for v in parameters.values()]

        loss_dict = {}
        val_loss_dict = {}
        pcc_dict = {}
        r2_score_dict = {}
        scc_dict = {}
        val_pcc_dict = {}
        val_r2_score_dict = {}
        val_scc_dict = {}
        dicts = [loss_dict, pcc_dict, r2_score_dict, scc_dict, val_loss_dict, val_pcc_dict, val_r2_score_dict, val_scc_dict]

        gene_dict = {}
        num_genes = 0

        count = 0
        for params in product(*param_values): 
                dicts = main(dicts, count, params, gene_dict, num_genes, sys.argv[6])
                count+=1
            
