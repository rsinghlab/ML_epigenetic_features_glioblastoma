## Machine learning on multiple epigenetic features reveals H3K27Ac as a driver of gene expression prediction across patients with glioblastoma.

## Datasets:
Our study's two patient epigenetic marker files are provided in the [data](data) folder. Specifically, the primary versions of this study's glioblastoma stem cell measurements ([GSC1](data/latest_versions_of_all_raw/gsc1_stem_with_featurecounts_RNAseq_entire_gene.npy) and [GSC2](data/latest_versions_of_all_raw/gsc2_stem_with_featurecounts_RNAseq_entire_gene.npy)) are located in the [latest_versions_of_all_raw](data/latest_versions_of_all_raw) subfolder. Additionally, to investigate gene expression across subsets of genes, there are subset files of both GSC1 and GSC2 where each primary dataset is split into 10 individual files whose genes do not overlap among the parts. The [ind_shuffle.npy](data/ind_shuffle.npy) file used in all the study's experiments is provided in the data folder. This file was used to create consistent dataset splits for train, validation, and test sets.

This study evaluates the cross-patient prediction methodology with all of the model scripts using this project's datasets. As a result of the relative performance this study observes among the model architectures, extended evaluation and analysis is performed with the XGBoost-based model using both this study's datasets and those created by adapting data from the study "Chromatin landscapes reveal developmentally encoded transcriptional states that define human glioblastoma" (https://doi.org/10.1084/jem.20190196). (1) Their study's data is avaliable from the Gene ExpressionOmnibus under accessions [GSE119755](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119755) and [GSE119834](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119834).  

## Cross-patient prediction models:

The project's cross-patient prediction models are avaliable in the following locations:

XGBoost (XGBR) [code/models/xgboost/xgboost_cross_patient_pred_regression_gsc_stem_standard_log2.py](code/models/xgboost/xgboost_cross_patient_pred_regression_gsc_stem_standard_log2.py)

"Branched" Multi-layered Perceptron ("Branched" MLP)
[code/models/mlp/mlp_cross_patient_regression_gsc_stem_sequence_standard_log2.py](code/models/mlp/mlp_cross_patient_regression_gsc_stem_sequence_standard_log2.py)

Multi-layered Perceptron (MLP)
[code/models/mlp/mlp_cross_patient_regression_gsc_stem_standard_log2.py](code/models/mlp/mlp_cross_patient_regression_gsc_stem_standard_log2.py)

Convolutional Neural Network (CNN) 
[code/models/cnn/cnn_cross_patient_pred_regression_gsc_stem_standard_log2.py](code/models/cnn/cnn_cross_patient_pred_regression_gsc_stem_standard_log2.py)

Recurrent Neural Network (RNN)
[code/models/rnn/rnn_models.py](code/models/rnn/rnn_models.py)

Gradient Boosting Regression (GBR)
[code/models/gbr/gbr_cross_patient_pred_regression_gsc_stem_standard_log2.py](code/models/gbr/gbr_cross_patient_pred_regression_gsc_stem_standard_log2.py)

Support Vector Machine (SVR)
[code/models/svm/svm_cross_patient_pred_regression_gsc_stem_standard_log2.py](code/models/svm/svm_cross_patient_pred_regression_gsc_stem_standard_log2.py)

Multiple Linear Regression (MLR)
[code/models/mlr/mlr_cross_patient_pred_regression_gsc_stem_standard_log2.py](code/models/mlr/mlr_cross_patient_pred_regression_gsc_stem_standard_log2.py)

## Setup process:

Clone this repository to the local filesystem using the link provided by the "Code" dropdown button above. For example:

```
git clone https://github.com/rsinghlab/ML_epigenetic_features_glioblastoma.git
```

Change the current working directory to the folder created by the clone process:

```
cd ./ML_epigenetic_features_glioblastoma
```

We recommend that a virtual environment be created to allow for the installation of the required packages and libraries in a without potential conflict with other packages already installed on the system. In the example here the virtual environment is given the same name as the project folder.

```
python3 -m venv ML_epigenetic_features_glioblastoma
```

Activate the new python environment.

```
source ./ML_epigenetic_features_glioblastoma/bin/activate
```

you can now install packages into the new environment using the included [requirements.txt](requirements.txt) file.

```
pip3 install -r requirements.txt
```

## How run the model scripts:

### Script arguments
### XGBoost, Gradient Boosting Regression, Convolutional Neural Network, Multi-layered Perceptron, Support Vector Machine, and Multiple Linear Regression
![script argument arrangement](assets/script_usage_image_1.jpeg)

A) The script's path and filename.

B) The first data file's path and filename. This script creates the model's training and validation (or training only) sets from this file. 



NOTE: The creation of a validation set is controlled by the ```validation = True``` or ```False``` statement in the script's ```main``` function. The proportions given to each set specified in the ```get_data_patient_1``` function under the comment ```#HYPERPARAMETER TUNING SPLITS``` and ```#TESTING SPLITS```.

C) The second data file's path and filename.

D) The [ind_shuffle.npy](data/ind_shuffle.npy) file (or equivalent) mentioned in the "Datasets" section above.

E) The integer to be used as the random seed.

F) The absolute or relative directory path where the various script functions will direct model output, predictions and visualizations. If no directory is specified, a directory name will be automatically generated and the directory created in the same directory where the script resides. The save directory's name will include the date and time the script was run.


### "Branched" Multi-layered Perceptron
![script argument arrangement 2](assets/script_usage_image_2.jpeg)

A) The script's path and filename.

B) The first data file's path and filename. This script creates the model's training and validation (or training only) sets from this file. 



NOTE: The creation of a validation set is controlled by the ```validation = True``` or ```False``` statement in the script's ```main()``` function. The proportions given to each set specified in the ```get_data_patient_1``` function under the comment ```#HYPERPARAMETER TUNING SPLITS``` and ```#TESTING SPLITS```.

C) The second data file's path and filename.

D) The script requires a file containing the genomic sequences for the corresponding genes in the first and second input files. Only one file is necessary since the script will use the sequences for the first and second inputs with the expectation that the sequence file includes data for all the genes contained in both. 

The [HG38_reference_genome_sequence_input_numerical_encoding.npy](data/HG38_reference_genome_data_files/HG38_reference_genome_sequence_input_numerical_encoding.npy) in the [data/HG38_reference_genome_data_files](data/HG38_reference_genome_data_files) folder was the primary version used for the study's experiments. 

E) The [ind_shuffle.npy](data/ind_shuffle.npy) file (or equivalent) mentioned in the "Datasets" section above.

F) The integer to be used as the random seed.

G) The absolute or relative directory path where the various script functions will direct model output, predictions and visualizations. If no directory is specified, a directory name will be automatically generated and the directory created in the same directory where the script resides. The save directory's name will include the date and time the script was run.


### Recurrent Neural Network
![script argument arrangement 3](assets/script_usage_image_3.jpeg)

A) The script's path with the rnn_models.py filename.

B) The designation of the train and validation dataset. Either ```GSC1``` or ```GSC2``` PLEASE NOTE: This designation is not the path and filename for the dataset.

B) The designation of the test dataset. Either ```GSC1``` or ```GSC2``` PLEASE NOTE: This designation is not the path and filename for the dataset.

C) The designation of which model architecture to be used: ```LSTM```, ```biLSTM```, ```LSTM-concatenated```, ```GRU```, or ```biGRU```.

D) A ```true``` or ```false``` for this argument will activate or de-activate the script's dataset preprocessing. If the script has been run previously with the chosen datasets, indicating ```false``` will allow the script to load the previously saved preprocessed files. 

E) The script will either be run for ```hyperparameter-tuning```, ```testing```, or ```visualizing```. 

F) The random seed integer.

G) Indicating ```attention``` will be used. If this is left blank attention will not be used.


PLEASE NOTE: The path and filename for each dataset and the indices file should be entered in the under GSC1 and GSC2 in the script's ```main``` function.

### Citations
1. Mack SC, Singh I, Wang X, Hirsch R, Wu Q, Villagomez R, et al. Chromatin landscapes reveal developmentally encoded transcriptional states that define human glioblastoma. J Exp Med. 20190404th ed. 2019 May 6;216(5):1071–90
