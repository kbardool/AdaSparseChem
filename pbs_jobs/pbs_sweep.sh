#!/bin/bash
# config=$1
# if [ -z "${config}" ]; then
#     exit 1
# fi
export pbs_cpu="-l nodes=1:ppn=9,walltime=06:00:00"
export pbs_gpu="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00"
# export config="../yamls/chembl_synt_train_5task.yaml"
export epochs=100
export  config="../yamls/chembl_mini_train.yaml"
export datadir="../../MLDatasets/chembl23_mini"
export  outdir="../../experiments/mini-SparseChem"

for layer in  100  ; do                   
    for lr in  0.001 ; do
        # for dropout in   0.05 ; do
        for dropout in   0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95; do
        # for dropout in  0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
        # for dropout in  0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
        # for dropout in  0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
            echo "Layer size: $layer   Task LR: $lr  Dropout: $dropout  datadir: $datadir outdir: $outdir  Config: $config"
            qsub pbs_tr_resid.sh -N AS-resid  -A lp_symbiosys \
                                  $pbs_gpu\
                                  -e ../pbs_output/  -o ../pbs_output/    \
                                  -v epochs=${epochs},dropout=${dropout},task_lr=${lr},backbone_lr=${lr},layer=${layer},config=${config} 

           qsub pbs_tr_nores.sh -N AS-noresid  -A lp_symbiosys \
                                 $pbs_gpu \
                                 -e ../pbs_output/  -o ../pbs_output/    \
                                 -v epochs=${epochs},dropout=${dropout},task_lr=${lr},backbone_lr=${lr},layer=${layer},config=${config} 
        done
    done
done