## Machine learning on multiple epigenetic features reveals H3K27Ac as a driver of gene expression prediction across patients with glioblastoma.

## Datasets:
Our study's two patient epigenetic marker files are provided in the [data](data) folder. Specifically, the latest versions of these GSC measurements are located in the [latest_versions_of_all_raw](data/latest_versions_of_all_raw) subfolder. Additionally, the [ind_shuffle.npy](data/ind_shuffle.npy) file is provided. This file was used to create consistent dataset splits for train, validation, and test sets.

We evaluated the cross-patient prediction methodology with all of the model scripts using this project's datasets. We performed extended evaluation and analysis with the XGBoost-based model using both this study's datasets and those created by adapting data from the study "Chromatin landscapes reveal developmentally encoded transcriptional states that define human glioblastoma" (https://doi.org/10.1084/jem.20190196). (1) Their study's data is avaliable from the Gene ExpressionOmnibus under accessions [GSE119755](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119755) and [GSE119834](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119834).  

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

### Cross-patient prediction models:

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

### Script arguments
### XGBoost, Gradient Boosting Regression, Multi-layered Perceptron, Support Vector Machine, Multiple Linear Regression
![script argument arrangement](assets/script_usage_image_1.jpeg)

A) The script's path and filename.

B) The first data file's path and filename. This script creates the model's training and validation (or training only) sets from this file. 



NOTE: The creation of a validation set is controlled by the ```validation = True``` or ```False``` statement in the script's ```main()``` function. The proportions given to each set specified in the ```get_data_patient_1``` function under the comment ```#HYPERPARAMETER TUNING SPLITS``` and ```#TESTING SPLITS```.

C) The second data file's path and filename.

D) The [ind_shuffle.npy](data/ind_shuffle.npy) file (or equivalent) mentioned in the "Datasets" section above.

E) The integer to be used as the random seed.

F) The absolute or relative directory path where the various script functions will direct model output, predictions and visualizations. If no directory is specified, a directory name will be automatically generated and the directory created in the same directory where the script resides. The save directory's name will include the date and time the script was run.

NOTE: 3/13/25 The 'script output save directory' argument and functionality is specific to the **XGBoost, Multi-layered Perceptron, Support Vector Machine, Gradient Boosting Regression and Multiple Linear Regression** model scripts. This functionality is planned for implementation in the other scripts. For now, arguments **A-D** are active for those scripts.

1. Mack SC, Singh I, Wang X, Hirsch R, Wu Q, Villagomez R, et al. Chromatin landscapes reveal developmentally encoded transcriptional states that define human glioblastoma. J Exp Med. 20190404th ed. 2019 May 6;216(5):1071–90
