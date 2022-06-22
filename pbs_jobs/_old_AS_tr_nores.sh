#!/bin/bash 

export JOBID=${PBS_JOBID:0:8}
echo Job $JOBID start : $(date)
source /user/leuven/326/vsc32647/.initconda
cd $PBS_O_WORKDIR # cd to the directory from which qsub is run
echo PBS VERSION is $PBS_VERSION
echo config file is $config
echo switch to pyt-gpu 
conda activate pyt-gpu
python -V
program="../src/Adashare_Train.py"
echo "option: [$option]  batch_size: $batch_size "

layers=""
echo "Number Layers: $num_layers   Layer size: $layer   Dropout: $dropout  Task LR: $lr "
for ((i=0 ; i < $num_layers ; i +=1)); do
    layers+=" $layer "
done
echo "Number Layers: $num_layers   Layer size: $layers   Dropout: $dropout  Task LR: $lr "
echo "===> program $program  -- excution started: $(date)"

python                     ${program} \
     --config              ${config} \
     --exp_desc            ${JOBID} - Run without residiual layers  \
     --warmup_epochs       ${epochs}  \
     --hidden_size         ${layers}  \
     --tail_hidden_size     ${layer}  \
     --first_dropout       ${dropout} \
     --middle_dropout      ${dropout} \
     --last_dropout        ${dropout} \
     --seed_idx                    0  \
     --batch_size       ${batch_size} \
     --task_lr                 ${lr}  \
     --backbone_lr             ${lr}  \
     --decay_lr_rate             0.3  \
     --decay_lr_freq              10  \
     --min_samples_class           2  \
     --gpu_ids                  ${dev}\
     --skip_residual

echo Job $JOBID finished : $(date)