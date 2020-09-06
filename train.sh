#This is for the WSOL with the SH file

#$ -l tmem=12G
#$ -l gpu=true
#$ -l h_rt=24:00:00 
#$ -j y
#$ -N WSOL
#$ -S /bin/bash
# -pe smp 2
# -R y

# activate the virtual env tensorflow

source activate covid
# Source file for CUDA 8.0 and cuDNN 7.0.5
# 23/04/18

source /share/apps/source_files/cuda/cuda-10.0.source

nvidia-smi

cd /cluster/project0/Ophthal_AV/Morphological_AV

hostname
date

export PYTHONPATH=.:$PYTHONPATH

python train.py --e=1500 --batch-size=2 --learning-rate=2e-4 --v=10.0 --alpha=0.5 --beta=1.1 --gama=0.08 --r=13 --dataset='LES-AV' --discriminator='unet'

date


