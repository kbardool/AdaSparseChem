#!/bin/bash
#PBS -l nodes=1:ppn=10
#PBS -l pmem=5gb
#PBS -l qos=debugging
#PBS -l walltime=30:00
#PBS -A lp_symbiosys
#PBS -M kevin.bardool@kuleuven.be

cd $PBS_O_WORKDIR # cd to the directory from which qsub is run
. $PBS_O_HOME/.bashrc
python -V
/data/leuven/326/vsc32647/miniconda3/bin/conda init bash
/data/leuven/326/vsc32647/miniconda3/bin/conda activate pyt-gpu
python -V
python /user/leuven/326/vsc32647/data/projs/AdaSparseChem/src/train.py \
                    --config yamls/chembl_synt_train.yaml \
                    --exp_desc    6 lyrs,dropout 0.5, weight 105 bch/ep policy 105 bch/ep \
                    --hidden_size   50 50 50 50 50 50    \
                    --tail_hidden_size    50  \
                    --seed_idx             0  \
                    --batch_size         128  \
                    --task_lr          0.001  \
                    --backbone_lr      0.001  \
                    --policy_lr         0.01  \
                    --lambda_sparsity   0.02  \
                    --lambda_sharing    0.01  
echo Job finished