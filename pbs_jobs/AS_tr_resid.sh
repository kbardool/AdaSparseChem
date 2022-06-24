#!/bin/bash 

source /user/leuven/326/vsc32647/.initconda
if [[ -n "$PBS_JOBID" ]]
 then
    export JOBID=${PBS_JOBID:0:8}
    echo "+++++++++++++++++++++++++++++++++++++++++++"
    echo "[$PBS_JOBID] --> Job [$JOBID] start : $(date)"
    echo " PBS VERSION is $PBS_VERSION"
    echo " Switch to $PBS_O_WORKDIR"
    echo "+++++++++++++++++++++++++++++++++++++++++++"
    cd $PBS_O_WORKDIR # cd to the directory from which qsub is run
  else
    echo "PBS Not defined"
fi
echo config file is $config
echo switch to pyt-gpu 
conda activate pyt-gpu
python -V
program="../src/Adashare_Train.py"
layers=""
echo "Res option: [$res_opt]   hidden option: [$hdn_opt] desc: [$desc]   batch_size: $batch_size "

echo  " Num Layers  :     $num_layers   Layer size: $layer   Dropout: $dropout  Task LR: $lr "
echo  " config      :     $config"
echo  " datadir     :     $datadir"
echo  " outdir      :     $outdir"

for ((i=0 ; i < $num_layers ; i +=1)); do
    layers+=" $layer "
done
echo  "Number Layers:     $num_layers   Layer size: $layers   Dropout: $dropout  Task LR: $lr "


echo "===> program $program  -- excution started: $(date)"

python                     ${program} \
     --config               ${config} \
     --exp_desc             ${JOBID} - ${desc}  \
     --warmup_epochs        ${epochs} \
     --hidden_size          ${layers} \
     --tail_hidden_size      ${layer} \
     --skip_residual       ${res_opt} \
     --skip_hidden         ${hdn_opt} \
     --first_dropout       ${dropout} \
     --middle_dropout      ${dropout} \
     --last_dropout        ${dropout} \
     --seed_idx           ${seed_idx} \
     --batch_size       ${batch_size} \
     --task_lr                  ${lr} \
     --backbone_lr              ${lr} \
     --decay_lr_rate              0.3 \
     --decay_lr_freq               10 \
     --gpu_ids                 ${dev} \
     --pytorch_threads              2 \
     --min_samples_class            2  
 
     
echo "===> program $program  -- excution ended: $(date)"
echo "===> Job $JOBID "
