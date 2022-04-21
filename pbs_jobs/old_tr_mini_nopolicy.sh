#!/bin/bash 
#PBS -l nodes=1:ppn=9:gpus=1
#PBS -l partition=gpu
#PBS -l walltime=06:00:00
#PBS -A lp_symbiosys

# module purge
source /user/leuven/326/vsc32647/.initconda
cd $PBS_O_WORKDIR # cd to the directory from which qsub is run
which conda
echo switch to pyt-gpu 
conda activate pyt-gpu
python -V


python ./src/train_nopolicy.py \
                    --config yamls/chembl_mini_train_VSC.yaml \
                     --exp_desc    6 lyrs, weight 105 bch/ep policy 105 bch/ep  \
                     --warmup_epochs       100  \
                     --hidden_size         100 100 100   \
                     --tail_hidden_size    100 \
                     --middle_dropout      0.0 \
                     --last_dropout        0.0 \
                     --seed_idx             0  \
                     --batch_size         128  \
                     --task_lr         0.0001  \
                     --backbone_lr     0.0001  \
                     --policy_lr        0.001  \
                     --lambda_sparsity   0.02  \
                     --lambda_sharing    0.01  \
                     --folder_sfx      noplcy                  

echo Job finished
