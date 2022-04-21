#!/bin/bash
echo $1
# parm=$1
if [ -z $1 ]; then
    echo "ERROR:: input parameter must be provided (res | nores | both)"
    exit 1
fi
# PBS -M kevin.bardool@kuleuven.be
# PBS -l pmem=5gb,
# PBS -l qos=debugging
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
lr=0.001

layer=600                   
num_layers=3
dropout=0.45

echo "Epochs: $epochs  Layers: $num_layers   Layer size: $layer   Dropout: $dropout  Task LR: $lr "

submit_job(){ 
    pbs_name="-N $2-${layer}x${num_layers}-${dropout}"
    echo " Program: $1  $pbs_name   Epochs: $epochs  Layers: $num_layers   Layer size: $layer   Dropout: $dropout  Task LR: $lr "

    qsub $1 
         $pbs_name \
         $pbs_account \
         $pbs_allocate  \
         $pbs_folders \
         -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr
}


if [ $1 == "res" ] || [ $1 == "both" ] 
then echo Its res! -- submit pbs_tr_resid 
    submit_job "AS_tr_resid.sh"  "ASR"
fi

if [ $1 == "nores" ] || [ $1 == "both" ] 
then echo Its nores! -- submit pbs_tr_nores
    submit_job "AS_tr_nores.sh"  "ASN"
fi



# qsub pbs_tr_resid.sh -N ASR-${layer}x${num_layers}-${dropout} $pbs_account $pbs_gpu $pbs_folders \
#     -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr

# qsub pbs_tr_nores.sh -N ASN-${layer}x${num_layers}-${dropout} $pbs_account $pbs_allocate $pbs_folders \
    # -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr
