#!/bin/bash 
if [ -z $1 ]; then
    echo "ERROR:: input parameter must be provided (res | nores | nohidden)"
    return 1
fi
ADASHARE_SCRIPT="AS_tr_resid.sh" 
config="../yamls/chembl_cb29_train.yaml"
datadir="../../MLDatasets/chembl29"
outdir="../../experiments/AdaSparseChem-cb29"
program="../src/Adashare_Train_mini.py"

dev=0
epochs=100
lr=0.0001

num_layers=4
layer=4000
dropout=0.80
batch_size=4096
seed_idx=0

submit_job(){ 
    dt=`date +%m%d.%H%M%S`
    job_name="$dt-$2-${layer}x${num_layers}-${dropout}"
    output_file="../pbs_output/${job_name}.out"
    seed_idx=$seed_idx
    res_opt=$3
    hdn_opt=$4
    desc="$5"
    printf "* Program: $1  Epochs: $epochs    Lyrs: $num_layers  Lyr sz: $layer   Dropout: $dropout  Task LR: $lr  device: $dev  output: $output_file \n"   
    printf "* $job_name  ($1) Epochs: $epochs Task LR: $lr  ---> \n"
    . $1 > $output_file 2>&1 &
    printf "* $job_name  ($1) Submitted in background \n"

}


echo "Epochs: $epochs  Layers: $num_layers   Layer size: $layer   Dropout: $dropout  Task LR: $lr "

# if [ $1 == "nohidden" ]  
# then 
    # submit_job "AS_tr_nohidden.sh"  "ASR"
# fi
# echo Its res! -- submit pbs_tr_resid 
# if [ $1 == "res" ] || [ $1 == "both" ] 
# then 
    # submit_job "AS_tr_resid.sh"  "ASR"
# fi
# then echo Its nores! -- submit pbs_tr_nores

# if [ $1 == "nores" ] || [ $1 == "both" ] 
# then 
    # submit_job "AS_tr_nores.sh"  "ASN"
# fi

if [ $1 == "nohidden" ]  
then echo " Its nohidden -- submit with --skip_hidden"
    hdn_opt="True"
    desc="Run without hidden layers"
    job_prefix="NH"
    submit_job $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
fi


if [ $1 == "res" ] || [ $1 == "both" ] 
then echo " Its res! -- submit pbs_tr_resid"
    opt="False"
    hdn_opt="False"
    desc="Run with residiual layers"
    job_prefix="AS"
    echo " desc:   $desc"
    submit_job $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
fi

if [ $1 == "nores" ] || [ $1 == "both" ] 
then echo " Its nores! -- submit pbs_tr_nores"
    opt="True"
    hdn_opt="False"
    desc="Run without residiual layers"
    job_prefix="NR"
    echo " desc:   $desc"
    submit_job $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
fi
 
