#!/bin/bash
echo $1
# parm=$1
if [ -z $1 ]; then
    echo "ERROR:: input parameter must be provided (res | nores | both)"
    return 1
fi

epochs=100
lr_list=(0.001)

# num_layers_list=(6)
# layer_size_list=(50) 
# dropout_list=(0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 )

num_layers_list=(1)
layer_size_list=(400) 
# dropout_list=(0.75 0.80 0.85 0.90 0.95)
dropout_list=( 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85  0.95)
# dropout_list=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)

# layer_size_list=( 50 100 200 300 400  )
# dropout_list=( 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
# dropout_list=(0.00 0.05 0.10 0.15 0.40 0.45 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)


submit_list(){ 
    for num_layers in ${num_layers_list[@]}; do                 
        for layer in ${layer_size_list[@]}; do                   
            for lr in ${lr_list[@]} ; do
                for dropout in  ${dropout_list[@]}; do
                    job_name="$2-${layer}x${num_layers}-${dropout}"
                    printf "$job_name   Epochs: $epochs Task LR: $lr  ---> "
                    # echo "  Program: $1 acct: $pbs_account  allc: $pbs_allocate  fldrs: $pbs_folders"
                    qsub $1 -N $job_name  $pbs_account  $pbs_allocate   $pbs_folders \
                        -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr
                done
            done
        done
    done
}


# PBS -M kevin.bardool@kuleuven.be
# PBS -l pmem=5gb,
# PBS -l qos=debugging
pbs_account="-A lp_symbiosys "
pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
# pbs_allocate="-l nodes=1:ppn=4,walltime=01:00:00 "
pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00 "
config="../yamls/chembl_mini_train.yaml"
datadir="../../MLDatasets/chembl23_mini"
outdir="../../experiments/mini-AdaSparseChem"
echo  datadir: $datadir outdir: $outdir confg: $config
echo  $pbs_account
echo  $pbs_folders
echo  $pbs_allocate



# layer_size_list=(100 200 300 400 500 600 700 800)
# dropout_list=( 0.9 )
# dropout_list=( 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 )
# dropout_list=( 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 )
# dropout_list=( 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 )



if [ $1 == "res" ] || [ $1 == "both" ] 
then echo Its res! -- submit pbs_tr_resid 
    submit_list "AS_tr_resid.sh"  "ASR"
fi

if [ $1 == "nores" ] || [ $1 == "both" ] 
then echo Its nores! -- submit pbs_tr_nores
    submit_list "AS_tr_nores.sh"  "ASN"
fi
 

