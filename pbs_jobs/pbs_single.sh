#!/bin/bash

pbs_account="-A lp_symbiosys "
pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
pbs_allocate="-l nodes=1:ppn=9,walltime=06:00:00 "
# pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00 "
config="../yamls/chembl_mini_train.yaml"
datadir="../../MLDatasets/chembl23_mini"
outdir="../../experiments/mini-AdaSparseChem"
echo  datadir: $datadir outdir: $outdir confg: $config
echo  $pbs_account
echo  $pbs_folders
echo  $pbs_allocate
epochs=100
lr=0.001

layer=600                   
num_layers=3
dropout=0.45

echo "Epochs: $epochs  Layers: $num_layers   Layer size: $layer   Dropout: $dropout  Task LR: $lr "

# qsub pbs_tr_resid.sh -N ASR-${layer}x${num_layers}-${dropout} $pbs_account $pbs_gpu $pbs_folders \
#     -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr

qsub pbs_tr_nores.sh -N ASN-${layer}x${num_layers}-${dropout} $pbs_account $pbs_allocate $pbs_folders \
    -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr
