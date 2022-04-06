#!/bin/bash
# config=$1
# if [ -z "${config}" ]; then
#     exit 1
# fi

# for dropout in   0.6 0.8 0.9 0.95 1.0; do
# for layer in 800 ; do                   
# for lr in 1e-05; do
# for lr in  0.001 0.0001 1e-06; do


for layer in 100 ; do                   
    for lr in  0.01 ; do
        for dropout in  0.0 0.1 0.2 0.4 0.6 0.8 0.9 0.95 1.0; do
        # for dropout in  0.0 0.1 0.2 0.3 0.4 ; do
            echo Layer size: $layer Task LR: $lr Backbone LR: $lr Dropout: $dropout
            qsub pbs_nopolicy.sh -v dropout=${dropout},task_lr=${lr},backbone_lr=${lr},layer=${layer} 
        done
    done 
done