import numpy as np
import csv

data_filepath = ' '
cell_lines = ['H3K27ac', 'CTCF', 'ATAC', 'RNApol2', 'RNAseq']

patients = ['GSC1_Stem', 'GSC1_Diff', 'GSC2_Stem', 'GSC2_Diff']

for cell_line in cell_lines:
    for patient in patients:
        in_file = data_filepath + cell_line + '/combined/' + patient + '_' + cell_line + '.100bp.count'
        outfile = data_filepath + cell_line + '/' + patient + '_' + cell_line + '_100bp.csv'

        header = ['GeneId', 'BinId', 'Count']

        with open(in_file) as fin, open(outfile, 'w') as fout:
            o=csv.writer(fout)
            o.writerow(header)
            prev_geneid = None
            curr_bin_id = -1
            for line in fin:
                cols = line.split()

                geneid = cols[3]
                count = float(cols[4])

                if geneid != prev_geneid:
                    prev_geneid = geneid
                    curr_bin_id = -1

                curr_bin_id += 1

                o.writerow([geneid, curr_bin_id, count])