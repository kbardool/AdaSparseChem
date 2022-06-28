#!/bin/bash 
#PBS -A lp_symbiosys
#PBS -e ../pbs_output/  -o ../pbs_output/ 
#PBS -N AS-resid


# pbs_account="-A lp_symbiosys "
# -l nodes=1:ppn=9,walltime=06:00:00 
# pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
# -l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00
# pbs_gpu="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00 "
# pbs_name="-N AS-resid"
# $pbs_account
# $pbs_folders
# $pbs_gpu
# $pbs_name

program="../src/Adashare_Train_mini.py"
JOBID=${PBS_JOBID:0:8}

echo Job $JOBID start : $(date)
source /user/leuven/326/vsc32647/.initconda
cd $PBS_O_WORKDIR # cd to the directory from which qsub is run
echo PBS VERSION is $PBS_VERSION
echo switch to pyt-gpu 
conda activate pyt-gpu
python -V
echo  $pbs_account
echo  $pbs_folders
echo  $pbs_gpu


echo program excution start: $(date)
echo  datadir: $datadir outdir: $outdir confg: $config

layers=""
echo "Number Layers: $num_layers   Layer size: $layer   Dropout: $dropout  Task LR: $lr "
for ((i=0 ; i < $num_layers ; i +=1)); do
    layers+=" $layer "
done
echo "Number Layers: $num_layers   Layer size: $layers   Dropout: $dropout  Task LR: $lr "


python                     ${program} \
     --config              ${config} \
     --exp_desc            ${JOBID} - Run with residiual layers  \
     --warmup_epochs       ${epochs}  \
     --hidden_size         ${layers}  \
     --tail_hidden_size     ${layer}  \
     --first_dropout       ${dropout} \
     --middle_dropout      ${dropout} \
     --last_dropout        ${dropout} \
     --seed_idx                    0  \
     --batch_size                128  \
     --task_lr                 ${lr}  \
     --backbone_lr             ${lr}  \
     --decay_lr_rate             0.3  \
     --decay_lr_freq              10  \
     --min_samples_class           2
 
     
echo Job $JOBID finished : $(date)
