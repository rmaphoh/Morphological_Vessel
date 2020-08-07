

# cuda 10.1 cudnn 7.5
#!/bin/bash  
#source /home/yukun/source/cuda_10.1.source
#source activate amd

source activate pytorch

export PYTHONPATH=.:$PYTHONPATH

python train.py --e=1500 --b=1 --learning-rate=2e-4 --v=10.0 --r=13 --dataset='DRIVE_AV' --discriminator='unet'

