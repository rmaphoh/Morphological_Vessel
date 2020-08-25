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

hostname
date

cd /SAN/medic/WSULOD2020/artery_vein_pytorch

export PYTHONPATH=.:$PYTHONPATH

python train.py --e=1800 --b=2 --learning-rate=2e-4 --v=10.0 --r=13 --dataset='DRIVE_AV' --discriminator='unet'

date


