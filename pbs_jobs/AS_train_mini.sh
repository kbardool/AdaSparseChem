#!/bin/bash 
#PBS -l nodes=1:ppn=9:gpus=1
#PBS -l partition=gpu
#PBS -l walltime=04:00:00
#PBS -A lp_symbiosys

# module purge
source /user/leuven/326/vsc32647/.initconda
# echo PTH is $PATH
# source $PBS_O_HOME/.bashrc
cd /data/leuven/326/vsc32647/projs/pbs
# cd $PBS_O_WORKDIR # cd to the directory from which qsub is run
# echo PBS HOMEDIR is $PBS_O_HOMEDIR
# echo PBS WORKDIR is $PBS_O_WORKDIR
which conda
echo switch to pyt-gpu 
conda activate pyt-gpu
python -V


python ./src/train.py \
       --config yamls/chembl_mini_train_VSC.yaml \
       --exp_desc    6 lyrs,dropout 0.5, weight 105 bch/ep policy 105 bch/ep \
       --warmup_epochs         350 \
       --hidden_size           400 \
       --tail_hidden_size      400  \
       --seed_idx                0  \
       --batch_size            128  \
       --task_lr             0.001  \
       --backbone_lr         0.001  \
       --policy_lr           0.001  \
       --lambda_sparsity      0.02  \
       --lambda_sharing       0.01  

echo Job finished
