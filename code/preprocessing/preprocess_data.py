import numpy as np
import csv
import pandas as pd
from pathlib import Path
import sys

# add filepath
data_filepath = ' '
output_filepath = ' '

# This function creates a dataframe representing the read counts of each gene split into 100bp.
# Each row of the dataframe is a different gene, and each column represents the counts of a 100bp window in the 10kbp sequence 
# The shape of the dataframe is (num_genes, 100)
def create_dataframes(filepath, cell_line, normalize=False, standardize=False, log=False):

    df = pd.read_csv(filepath)

    norm_count_col_name = cell_line + '_count'

    df.rename(columns={'Count': norm_count_col_name}, inplace=True)

    if standardize:
	    #define predictor variable columns
        df_x = df[[norm_count_col_name]]

        #standardize the values for each predictor variable
        df[[norm_count_col_name]] = (df_x-df_x.mean()) / df_x.std()

    if normalize:
        df[norm_count_col_name] = df[norm_count_col_name] / df[norm_count_col_name].max() #this line normalizes the counts between 0 and 1
    
    if log:
        df[norm_count_col_name] = np.log2(df[norm_count_col_name] + 1)    

    return df

# This function converts a dataframe to a numpy array with the shape (num_genes, 100) where 100 is the number of 100bp long reads for each gene
def dataframe_to_np(df):
    return df.to_numpy()
    
# Will implement this when we have multiple datasets
def combine_dataframes(dataset_list):
    combined = dataset_list[0]

    for i in range(1, len(dataset_list)):
        dataset = dataset_list[i]
        combined = combined.merge(dataset, on=['GeneId', 'BinId'], how='inner')
    
    return combined


if __name__ == "__main__":
    # pd.set_option("display.max_cols", None)

    # colums for combined_np are geneid, binid, h3k27ac, ctcf, atac, rnapol2, rnaseq

    # Create GSC1 Stem Dataframes
    gsc1_stem_H3K27ac_df = create_dataframes(data_filepath + 'H3K27ac/GSC1_Stem_H3K27ac_100bp.csv', 'H3K27ac')
    gsc1_stem_CTCF_df = create_dataframes(data_filepath + 'CTCF/GSC1_Stem_CTCF_100bp.csv', 'CTCF')
    gsc1_stem_ATAC_df = create_dataframes(data_filepath + 'ATAC/GSC1_Stem_ATAC_100bp.csv', 'ATAC')
    gsc1_stem_RNApol2_df = create_dataframes(data_filepath + 'RNApol2/GSC1_Stem_RNApol2_100bp.csv', 'RNApol2')
    #gsc1_stem_RNAseq_df = create_dataframes(data_filepath + 'RNAseq/GSC1_Stem_RNAseq.csv', 'RNAseq', False, False, False)
    gsc1_stem_RNAseq_df = create_dataframes(data_filepath + 'RNAseq/GSC1_Stem_RNAseq_100bp.csv', 'RNAseq', False, False, False)

    # Create GSC1 Diff Dataframes
    gsc1_diff_H3K27ac_df = create_dataframes(data_filepath + 'H3K27ac/GSC1_Diff_H3K27ac_100bp.csv', 'H3K27ac')
    gsc1_diff_CTCF_df = create_dataframes(data_filepath + 'CTCF/GSC1_Diff_CTCF_100bp.csv', 'CTCF')
    gsc1_diff_ATAC_df = create_dataframes(data_filepath + 'ATAC/GSC1_Diff_ATAC_100bp.csv', 'ATAC')
    gsc1_diff_RNApol2_df = create_dataframes(data_filepath + 'RNApol2/GSC1_Diff_RNApol2_100bp.csv', 'RNApol2')  
    #gsc1_diff_RNAseq_df = create_dataframes(data_filepath + 'RNAseq/GSC1_Diff_RNAseq.csv', 'RNAseq', False, False, False)  
    gsc1_diff_RNAseq_df = create_dataframes(data_filepath + 'RNAseq/GSC1_Diff_RNAseq_100bp.csv', 'RNAseq', False, False, False) 

    # Create GSC2 Stem Dataframes
    gsc2_stem_H3K27ac_df = create_dataframes(data_filepath + 'H3K27ac/GSC2_Stem_H3K27ac_100bp.csv', 'H3K27ac')
    gsc2_stem_CTCF_df = create_dataframes(data_filepath + 'CTCF/GSC2_Stem_CTCF_100bp.csv', 'CTCF')
    gsc2_stem_ATAC_df = create_dataframes(data_filepath + 'ATAC/GSC2_Stem_ATAC_100bp.csv', 'ATAC')
    gsc2_stem_RNApol2_df = create_dataframes(data_filepath + 'RNApol2/GSC2_Stem_RNApol2_100bp.csv', 'RNApol2')
    #gsc2_stem_RNAseq_df = create_dataframes(data_filepath + 'RNAseq/GSC2_Stem_RNAseq.csv', 'RNAseq', False, False, False)
    gsc2_stem_RNAseq_df = create_dataframes(data_filepath + 'RNAseq/GSC2_Stem_RNAseq_100bp.csv', 'RNAseq', False, False, False)

    # Create GSC2 Diff Dataframes
    gsc2_diff_H3K27ac_df = create_dataframes(data_filepath + 'H3K27ac/GSC2_Diff_H3K27ac_100bp.csv', 'H3K27ac')
    gsc2_diff_CTCF_df = create_dataframes(data_filepath + 'CTCF/GSC2_Diff_CTCF_100bp.csv', 'CTCF')
    gsc2_diff_ATAC_df = create_dataframes(data_filepath + 'ATAC/GSC2_Diff_ATAC_100bp.csv', 'ATAC')
    gsc2_diff_RNApol2_df = create_dataframes(data_filepath + 'RNApol2/GSC2_Diff_RNApol2_100bp.csv', 'RNApol2')
    #gsc2_diff_RNAseq_df = create_dataframes(data_filepath + 'RNAseq/GSC2_Diff_RNAseq.csv', 'RNAseq', False, False, False)
    gsc2_diff_RNAseq_df = create_dataframes(data_filepath + 'RNAseq/GSC2_Diff_RNAseq_100bp.csv', 'RNAseq', False, False, False)  

    print("[1] done")

    # Create dataset lists for combinations
    gsc1_stem_datasets = [gsc1_stem_H3K27ac_df, gsc1_stem_CTCF_df, gsc1_stem_ATAC_df, gsc1_stem_RNApol2_df, gsc1_stem_RNAseq_df]
    gsc1_diff_datasets = [gsc1_diff_H3K27ac_df, gsc1_diff_CTCF_df, gsc1_diff_ATAC_df, gsc1_diff_RNApol2_df, gsc1_diff_RNAseq_df]
    gsc2_stem_datasets = [gsc2_stem_H3K27ac_df, gsc2_stem_CTCF_df, gsc2_stem_ATAC_df, gsc2_stem_RNApol2_df, gsc2_stem_RNAseq_df]
    gsc2_diff_datasets = [gsc2_diff_H3K27ac_df, gsc2_diff_CTCF_df, gsc2_diff_ATAC_df, gsc2_diff_RNApol2_df, gsc2_diff_RNAseq_df]

    print("[2] done")

    # Combine datasets for each cell line
    gsc1_stem = combine_dataframes(gsc1_stem_datasets)
    gsc1_diff = combine_dataframes(gsc1_diff_datasets)
    gsc2_stem = combine_dataframes(gsc2_stem_datasets)
    gsc2_diff = combine_dataframes(gsc2_diff_datasets)

    print("[3] done")

    # Turn dataframes into numpy arrays
    gsc1_stem_np = dataframe_to_np(gsc1_stem)
    gsc1_diff_np = dataframe_to_np(gsc1_diff)
    gsc2_stem_np = dataframe_to_np(gsc2_stem)
    gsc2_diff_np = dataframe_to_np(gsc2_diff)

    print("[4] done")

    # Save to npy files
    np.save(output_filepath + 'gsc1_stem.npy', gsc1_stem_np)
    np.save(output_filepath + 'gsc1_diff.npy', gsc1_diff_np)
    np.save(output_filepath + 'gsc2_stem.npy', gsc2_stem_np)
    np.save(output_filepath + 'gsc2_diff.npy', gsc2_diff_np)

    print("[5] done")
