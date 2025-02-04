import numpy as np
import csv

data_filepath = ' '

cell_line = ['RNAseq']

patients = ['GSC1_RNA_stem', 'GSC1_RNA_diff', 'GSC2_RNA_stem', 'GSC2_RNA_diff']
patients2 = ['GSC1_Stem', 'GSC1_Diff', 'GSC2_Stem', 'GSC2_Diff']

header = ['GeneId', 'BinId', 'Count']
for i in range(1):
    #in_file = data_filepath + cell_line + '/' + patients[i] + '_10kb_featureCounts'
    #in_file = data_filepath + 'count_files/' cell_line + '/' + patients[i] + '_upanddown2.5kb.counts'
    #in_file = data_filepath + cell_line + '/' + 'count_files/' + patients[i] + '_entiregene_TPM_ver2.counts'

    in_file = data_filepath + cell_line + '/' + patients[i] + '_' + cell_line + '_entiregene_TPM.counts'
    outfile = data_filepath + cell_line + '/' + patients2[i] + '_' + cell_line + '_100bp.csv'
    with open(in_file) as fin, open(outfile, 'w') as fout:
        o=csv.writer(fout)
        o.writerow(header)
        skip_head = False
        line_num = 0
        for line in fin:
            if line_num > 1:
                cols = line.split()
                geneid = cols[0]
                count = cols[-1]
                curr_bin_id = 0
                for i in range(50):
                    o.writerow([geneid, curr_bin_id, count])
                    curr_bin_id += 1
            else:
                line_num += 1
