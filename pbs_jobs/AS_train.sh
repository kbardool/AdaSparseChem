#!/bin/bash 
divider="========================================================================\n"
program="../src/Adashare_Train.py"
layers=""
printf "===> program $program  -- excution started: $(date) \n"
printf $divider
printf " program        : $program \n"
printf " epochs         : $epochs  \n"
printf " skip residual  : $res_opt \n"
printf " skip hidden    : $hdn_opt \n"
printf " exp_desc       : $desc    \n"
printf " batch_size     : $batch_size  \n"
printf " num_layers     : ${num_layers} \n"
printf " layer_size     : ${layer}  \n"
printf " dropouts       : ${dropout}  \n"
printf " learning rate  : ${lr}  \n"
printf " datadir        : $datadir \n"
# printf " outdir         : $outdir \n"
printf " config         : $config \n"        
printf " seed_idx       : $seed_idx \n"        
printf " cuda_device_id : $cuda_device_id \n"        
printf " py_threads     : $py_threads \n"        
printf " device         : $dev \n"
printf " job date       : $job_dt \n"        
printf $divider 

if [[ -n "$PBS_JOBID" ]]
 then
    JOBID=${PBS_JOBID:0:8}
    printf " PBS JOBID  : $PBS_JOBID  \n"
    printf " PBS VERSION: $PBS_VERSION   \n"
    printf " PBS WORKDIR: $PBS_O_WORKDIR \n"
    cd $PBS_O_WORKDIR # cd to the directory from which qsub is run
  else
    JOBID=${job_dt}
    printf " PBS Not defined   \n"
    printf " JOBID  : [$JOBID] \n"
fi
printf $divider
# printf "=================================================\n" 
for ((i=0 ; i < $num_layers ; i +=1)); do
    layers+=" $layer "
done
source /user/leuven/326/vsc32647/.initconda
# echo switch to pyt-gpu 
conda activate pyt-gpu
# python -V
# export CUDA_VISIBLE_DEVICES=${gpu_id}
# echo "gpu_id:  $gpu_id    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# printf "Number Layers:     $num_layers   Layer size: $layers   Dropout: $dropout  Task LR: $lr \n"

printf $divider

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
     --cuda_devices   ${cuda_device_id} \
     --gpu_ids                 ${dev} \
     --pytorch_threads  ${py_threads} \
     --min_samples_class            2  
 
printf $divider     
printf "===> program $program  -- excution ended: $(date)\n"
printf "===> Job $JOBID Terminated Normally\n"
