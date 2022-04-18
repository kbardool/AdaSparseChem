#!/bin/bash
# config=$1
# if [ -z "${config}" ]; then
#     exit 1
# fi
# for layer in 800 ; do                   
# for lr in 1e-05; do
# for lr in  0.001 0.0001 1e-06; do
# export config="../yamls/chembl_synt_train_1task.yaml"
# export config="../yamls/chembl_synt_train_3task.yaml"
# export config="../yamls/chembl_synt_train_5task.yaml"
export epochs=100
export  config="../yamls/chembl_mini_train.yaml"
export datadir="../../MLDatasets/chembl23_mini"
export  outdir="../../experiments/mini-SparseChem"

for layer in 600 ; do                   
    for lr in  0.001 ; do
        for dropout in 0.1 ; do
            echo "Layer size: $layer   Task LR: $lr  Dropout: $dropout  datadir: $datadir outdir: $outdir  Config: $config"
            qsub pbs_train_residual.sh -v epochs=${epochs},dropout=${dropout},task_lr=${lr},backbone_lr=${lr},layer=${layer},config=${config} 
            qsub pbs_train_nonres.sh   -v epochs=${epochs},dropout=${dropout},task_lr=${lr},backbone_lr=${lr},layer=${layer},config=${config} 
        done
    done
done