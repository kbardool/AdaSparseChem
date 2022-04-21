#!/bin/bash
echo $1
# parm=$1
if [ -z $1 ]; then
    echo "ERROR:: input parameter must be provided (res | nores | both)"
    exit 1
fi
# PBS -M kevin.bardool@kuleuven.be
# PBS -l pmem=5gb
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
lr_list=(0.001)

# num_layers_list
num_layers_list=(2)

layer_size_list=(200 )
# for layer in  100 200 300 400 500 600 700 800 ; do  

dropout_list=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)
# for dropout in   0.9 ; do
# for dropout in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 ; do
# for dropout in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 ; do
# for dropout in  0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95; do

submit_list(){ 
    for num_layers in ${num_layers_list[@]}; do                 
        for layer in ${layer_size_list[@]}; do                   
            for lr in ${lr_list[@]} ; do
                for dropout in  ${dropout_list[@]}; do
                    echo " Program: $1  Pfx: $2  Epochs: $epochs  Layers: $num_layers   Layer size: $layer   Dropout: $dropout  Task LR: $lr "

                    qsub $1 -N $2-${layer}x${num_layers}-${dropout}   \
                         $pbs_account $pbs_allocate  $pbs_folders \
                        -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr
                done
            done
        done
    done
}



if [ $1 == "res" ] || [ $1 == "both" ] 
then echo Its res! -- submit pbs_tr_resid 
    submit_list "pbs_tr_resid.sh"  "ASR"
fi

if [ $1 == "nores" ] || [ $1 == "both" ] 
then echo Its nores! -- submit pbs_tr_nores
    submit_list "pbs_tr_nores.sh"  "ASN"
fi

#-------------------------------------------------------------------------
#    qsub pbs_tr_nores.sh -N ASN-${layer}x${numresid}-${dropout}  \
#                           $pbs_account \
#                           $pbs_allocate     \
#                           $pbs_folders \
#                           -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr