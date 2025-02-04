#!/bin/bash
#SBATCH -N 1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
# Specify a job name:
#SBATCH -J data_preprocessing
# Specify an output file
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=

## mandatory
# 'f', --file_name, default='H3K27ac', type=str'
# 'c', '--cell_line', default='GSC1_Stem', type=str
# 's', --sign, type=str'

# remove 200b with :%s/[^[:print:]]//g


module load python/3.9.16s-x3wdtvt
module load bedtools2/2.31.0-lsohc7s
module load samtools/1.16.1-txuglks
module load anaconda/2023.09-0-7nso27y

# cd to path where script is stored on filesystem
cd 

# samtools sort /bam_files/${f}/${c}_${f}.bam > /bam_files/${f}/${c}_${f}_sorted.bam
# echo "[5a] complete"
# index bam file
# samtools index ../bam_files/${f}/${c}_${f}_sorted.bam 
# echo "[5b] complete"
# [6] Use bedtools multicov to convert bam files to .count files

bedtools multicov -bams /${f}/${c}.bam -bed hg38_${s}.100bp.windows.bed > /${f}/seperate/${c}_${f}_${s}.100bp.count

echo "[6] complete"

sacct -j $SLURM_JOBID --format=JobID,JobName,Elapsed

echo $SLURM_JOBID

# RUN ON COMMAND LINE BY TYPING: 

# H3k27ac
# sbatch -J GSC1_Stem_Plus --export=ALL,f='H3K27ac',c='GSC1_Stem',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Stem_Minus --export=ALL,f='H3K27ac',c='GSC1_Stem',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Diff_Plus --export=ALL,f='H3K27ac',c='GSC1_Diff',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Diff_Minus --export=ALL,f='H3K27ac',c='GSC1_Diff',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Stem_Plus --export=ALL,f='H3K27ac',c='GSC2_Stem',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Stem_Minus --export=ALL,f='H3K27ac',c='GSC2_Stem',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Diff_Plus --export=ALL,f='H3K27ac',c='GSC2_Diff',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Diff_Minus --export=ALL,f='H3K27ac',c='GSC2_Diff',s='minus' sbatch_extract_100bp.sh

# CTCF
# sbatch -J GSC1_Stem_Plus --export=ALL,f='CTCF',c='GSC1_Stem',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Stem_Minus --export=ALL,f='CTCF',c='GSC1_Stem',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Diff_Plus --export=ALL,f='CTCF',c='GSC1_Diff',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Diff_Minus --export=ALL,f='CTCF',c='GSC1_Diff',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Stem_Plus --export=ALL,f='CTCF',c='GSC2_Stem',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Stem_Minus --export=ALL,f='CTCF',c='GSC2_Stem',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Diff_Plus --export=ALL,f='CTCF',c='GSC2_Diff',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Diff_Minus --export=ALL,f='CTCF',c='GSC2_Diff',s='minus' sbatch_extract_100bp.sh

# RNApol2
# sbatch -J GSC1_Stem_Plus_RNApol2 --export=ALL,f='RNApol2',c='GSC1_Stem',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Stem_Minus_RNApol2 --export=ALL,f='RNApol2',c='GSC1_Stem',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Diff_Plus_RNApol2 --export=ALL,f='RNApol2',c='GSC1_Diff',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Diff_Minus_RNApol2 --export=ALL,f='RNApol2',c='GSC1_Diff',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Stem_Plus_RNApol2 --export=ALL,f='RNApol2',c='GSC2_Stem',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Stem_Minus_RNApol2 --export=ALL,f='RNApol2',c='GSC2_Stem',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Diff_Plus_RNApol2 --export=ALL,f='RNApol2',c='GSC2_Diff',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Diff_Minus_RNApol2 --export=ALL,f='RNApol2',c='GSC2_Diff',s='minus' sbatch_extract_100bp.sh

# ATAC
# sbatch -J GSC1_Stem_Plus_ATAC --export=ALL,f='ATAC',c='GSC1_Stem',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Stem_Minus_ATAC --export=ALL,f='ATAC',c='GSC1_Stem',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Diff_Plus_ATAC --export=ALL,f='ATAC',c='GSC1_Diff',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Diff_Minus_ATAC --export=ALL,f='ATAC',c='GSC1_Diff',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Stem_Plus_ATAC --export=ALL,f='ATAC',c='GSC2_Stem',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Stem_Minus_ATAC --export=ALL,f='ATAC',c='GSC2_Stem',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Diff_Plus_ATAC --export=ALL,f='ATAC',c='GSC2_Diff',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Diff_Minus_ATAC --export=ALL,f='ATAC',c='GSC2_Diff',s='minus' sbatch_extract_100bp.sh

# RNAseq
# sbatch -J GSC1_Stem_Plus_RNAseq --export=ALL,f='RNAseq',c='GSC1_Stem',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Stem_Minus_RNAseq --export=ALL,f='RNAseq',c='GSC1_Stem',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Diff_Plus_RNAseq --export=ALL,f='RNAseq',c='GSC1_Diff',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC1_Diff_Minus_RNAseq --export=ALL,f='RNAseq',c='GSC1_Diff',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Stem_Plus_RNAseq --export=ALL,f='RNAseq',c='GSC2_Stem',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Stem_Minus_RNAseq --export=ALL,f='RNAseq',c='GSC2_Stem',s='minus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Diff_Plus_RNAseq --export=ALL,f='RNAseq',c='GSC2_Diff',s='plus' sbatch_extract_100bp.sh
# sbatch -J GSC2_Diff_Minus_RNAseq --export=ALL,f='RNAseq',c='GSC2_Diff',s='minus' sbatch_extract_100bp.sh

