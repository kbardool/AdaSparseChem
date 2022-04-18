#!/bin/bash 
#   -l nodes=1:ppn=9,walltime=01:00:00
#   -A lp_symbiosys

# module purge
export JOBID=${PBS_JOBID:0:8}
echo Job $JOBID start : $(date)
source /user/leuven/326/vsc32647/.initconda
cd $PBS_O_WORKDIR # cd to the directory from which qsub is run
echo PBS VERSION is $PBS_VERSION
echo config file is $config
echo switch to pyt-gpu 
conda activate pyt-gpu
python -V
export program="../src/Adashare_Train_mini.py"
echo program excution start: $(date)

python                     ${program} \
     --config              ${config} \
     --exp_desc            ${JOBID} - Run without residiual layers  \
     --warmup_epochs       ${epochs}  \
     --hidden_size         ${layer} ${layer}  ${layer} \
     --tail_hidden_size    ${layer} \
     --first_dropout       ${dropout}\
     --middle_dropout      ${dropout}\
     --last_dropout        ${dropout}\
     --seed_idx             0  \
     --batch_size         128  \
     --task_lr             ${task_lr}  \
     --backbone_lr         ${backbone_lr}  \
     --decay_lr_rate      0.3 \
     --decay_lr_freq       10  \
     --min_samples_class    2 \
     --no_residual

echo Job $JOBID finished : $(date)