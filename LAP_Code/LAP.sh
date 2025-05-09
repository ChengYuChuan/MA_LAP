#!/bin/bash
#
#SBATCH --job-name=LAP
#SBATCH --output=BS2_RBPNI_32_3Layers_CD_Cube32_L1_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=12:00:00
#SBATCH --mem=200gb
#SBATCH --mail-user=yu-chuan.cheng@stud.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A100:1


# JOB STEPS (example: write hostname to output file, and wait 1 minute)
#LossType = sys.argv[1] # "SSIMLoss" or "MSELoss" or "L1MSSSIMLoss"
#Cubesets = sys.argv[2] # "Cubes" or "MaskedCubes" or "MergedMaskedCubes"
#CubeSize = sys.argv[3] # "24" or "32"
#PoolType = sys.argv[4] # 'avg' or 'max'
#Learning_Rate = float(sys.argv[5]) # 0.0001
#window_size = sys.argv[6] # cube24 should be 5 or 3, cube32 should 7 or 11
#alpha = float(sys.argv[7])

#CHECK The Dir before you submit the sh file!!!!!!!!!!!!!!!!!
cd /home/hd/hd_hd/hd_uu312/MA_LAP/LAP_Code
source ~/.bashrc
conda activate MAenv

srun hostname
srun python train.py "MultiLayerHungarianLoss" "MergedCubes" 32 'max' 0.001 5 0.8