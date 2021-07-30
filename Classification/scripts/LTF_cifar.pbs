#!/bin/bash
#PBS -q alloc-dt
#PBS -P JxG
#PBS -l select=1:ncpus=4:ngpus=1:mem=40GB
#PBS -l walltime=200:00:00


module load python/3.6.5
source /project/JxG/tf_gpu_venv/bin/activate
module load cuda/10.0.130
module load openmpi-gcc/3.1.3


cd /project/DLT/LTF
python main.py --dataset='cifar10' --nz=128 --nc=3 --num_class=10 --c_epochs=20