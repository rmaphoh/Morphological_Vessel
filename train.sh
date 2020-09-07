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

source /share/apps/source_files/cuda/cuda-10.0.source

nvidia-smi

hostname
date

export PYTHONPATH=.:$PYTHONPATH

python train.py --e=1800 --batch-size=2 --learning-rate=8e-4 --v=10.0 --alpha=0.5 --beta=1.1 --gama=0.08 --r=13 --dataset='LES-AV-patch' --discriminator='unet'

date


