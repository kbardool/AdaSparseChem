#!/bin/bash


pbs_account="-A lp_symbiosys "
pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
# pbs_allocate="-l nodes=1:ppn=9,walltime=06:00:00 "
pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00 "


config="../yamls/chembl_mini_train.yaml"
datadir="../../MLDatasets/chembl23_mini"
outdir="../../experiments/mini-AdaSparseChem"
echo  datadir: $datadir outdir: $outdir confg: $config
echo  $pbs_account
echo  $pbs_folders
echo  $pbs_allocate
epochs=100

# num_layers=

# for layer in  100 200 300 400 500 600 700 800 ; do  
# for dropout in  0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95; do
for num_layers in 2; do                 
    for layer in  200; do                   
        for lr in  0.001 ; do
            for dropout in  0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95; do
            # for dropout in   0.9 ; do
            # for dropout in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 ; do
            # for dropout in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 ; do
                echo "Epochs: $epochs  Layers: $num_layers   Layer size: $layer   Dropout: $dropout  Task LR: $lr "

            #     qsub pbs_tr_resid.sh -N ASR-${layer}x${num_layers}-${dropout}   \
            #                         $pbs_account \
            #                         $pbs_allocate     \
            #                         $pbs_folders \
            #                         -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr


            #    qsub pbs_tr_nores.sh -N ASN-${layer}x${num_layers}-${dropout}  \
            #                           $pbs_account \
            #                           $pbs_allocate     \
            #                           $pbs_folders \
            #                           -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr
            done
        done
    done
done
