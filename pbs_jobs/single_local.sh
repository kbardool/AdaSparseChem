#!/bin/bash 
if [ -z $1 ]; then
    echo "ERROR:: input parameter must be provided (res | nores | nohidden)"
    return 1
fi

config="../yamls/chembl_mini_train.yaml"
datadir="../../MLDatasets/chembl23_mini"
outdir="../../experiments/mini-AdaSparseChem"
program="../src/Adashare_Train_mini.py"

dev="cuda:0"
epochs=2
lr=0.001

num_layers=1
layer=1000
dropout=0.65

submit_job(){ 
    job_name="$2-${layer}x${num_layers}-${dropout}"
    output_file="../pbs_output/${job_name}.out"
    
    echo " Program: $1  Epochs: $epochs    Lyrs: $num_layers  Lyr sz: $layer   Dropout: $dropout  Task LR: $lr  device: $dev  output: $output_file "   
    printf "$job_name   Epochs: $epochs Task LR: $lr  ---> "
    . $1 > $output_file 2>&1 &
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
