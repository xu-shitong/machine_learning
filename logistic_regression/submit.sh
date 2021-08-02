#!/bin/bash
#testslurm
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sx119
#export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
source ~/.bashrc
source /vol/cuda/10.2.89-cudnn7.6.4.38/setup.sh
TERM=vt100
echo This is a test
echo Today is $( date ) 
echo This is $( /bin/hostname )
echo running python program
echo $PATH
python3 ~/machine_learning/logistic_regression/gpu_trial.py
/usr/bin/nvidia-smi
uptime

