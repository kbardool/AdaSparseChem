#!/bin/bash 
if [ -z $1 ]; then
    echo "ERROR:: input parameter must be provided (res | nores | nohidden)"
    return 1
fi

config="../yamls/chembl_cb29_train.yaml"
datadir="../../MLDatasets/chembl29"
outdir="../../experiments/AdaSparseChem-cb29"
program="../src/Adashare_Train_mini.py"

dev=0
epochs=100
lr=0.001

num_layers=1
layer=4000
dropout=0.90
batch_size=4096

submit_job(){ 
    job_name="$2-${layer}x${num_layers}-${dropout}"
    output_file="../pbs_output/${job_name}.out"
    
    printf "* Program: $1  Epochs: $epochs    Lyrs: $num_layers  Lyr sz: $layer   Dropout: $dropout  Task LR: $lr  device: $dev  output: $output_file \n"   
    printf "* $job_name  ($1) Epochs: $epochs Task LR: $lr  ---> \n"
    . $1 > $output_file 2>&1 &
    printf "* $job_name  ($1) Submitted in background \n"

}


echo "Epochs: $epochs  Layers: $num_layers   Layer size: $layer   Dropout: $dropout  Task LR: $lr "

if [ $1 == "nohidden" ]  
then 
    submit_job "AS_tr_nohidden.sh"  "ASR"
fi

# echo Its res! -- submit pbs_tr_resid 
if [ $1 == "res" ] || [ $1 == "both" ] 
then 
    submit_job "AS_tr_resid.sh"  "ASR"
fi
# then echo Its nores! -- submit pbs_tr_nores

if [ $1 == "nores" ] || [ $1 == "both" ] 
then 
    submit_job "AS_tr_nores.sh"  "ASN"
fi
