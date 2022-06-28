#!/bin/bash
# parm=$1
if [ -z $1 ]; then
    echo "ERROR:: input parameter must be provided (res | nores | both | nohidden)"
    return 1
fi
submit_type="LOCAL"
# PBS Parameters =========================================================
# PBS -M kevin.bardool@kuleuven.be
# PBS -l pmem=5gb,
# PBS -l qos=debugging
pbs_account="-A lp_symbiosys "
pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=72:00:00 "
# pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=24:00:00 "
# pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=01:00:00 "
# pbs_allocate="-l nodes=1:ppn=4,walltime=01:00:00 "

# Config Parameters ======================================================
ADASHARE_SCRIPT="AS_train.sh" 
config="../yamls/chembl_cb29_train.yaml"
datadir="../../MLDatasets/chembl29"
outdir="../../experiments/AdaSparseChem-cb29"

# Training Parms =========================================================
# lr_list=(0.001)
lr_list=(0.0001)
batch_size=4096

num_layers_list=(6)
# num_layers_list=(1 2 3)
layer_size_list=(4000) 
dropout_list=(0.80 0.70 0.60)

epochs=100
seed_idx=0
py_threads=1
dev=0
# dropout_list=(0.40 0.50 0.60)
# dropout_list=( 0.70  0.80  0.90)
#=========================================================================

submit_list(){ 
    for num_layers in ${num_layers_list[@]}; do                 
    for layer in ${layer_size_list[@]}; do                   
    for lr in ${lr_list[@]} ; do
    for dropout in  ${dropout_list[@]}; do
        # printf "\n"
        # printf "* seed: $seed_idx  output: $output_file \n"   
        # printf "* Epochs: $epochs  Lyrs: $num_layers  Lyr sz: $layer  Dropout: $dropout  Task LR: $lr  device: $dev \n"
        printf "\n"
        printf " submit type  : $submit_type\n"
        printf " Program      : $1 \n"
        printf " job_prefix   : $2 \n"
        # printf " Skip Residual: $res_opt \n"
        # printf " Skip Hidden  : $hdn_opt \n"
        # printf " exp_desc     : $desc \n"
        # printf " pbs_account  : $pbs_account \n"
        # printf " pbs_allocate : $pbs_allocate\n"
        # printf " pbs_folders  : $pbs_folders \n"        
        # printf " batch_size   : $batch_size  \n"
        # printf " learning rt  : $lr  \n"
        # printf " datadir      : $datadir \n"
        # printf " outdir       : $outdir \n"
        # printf " config       : $config \n"        
        printf " JobName      : $job_name  Epochs: $epochs   LR: $lr   dev: $dev ---> "
        if [[ $submit_type == "PBS" ]]; then
            job_name="$229-${layer}x${num_layers}-${dropout}"
            . submit_pbs_job.sh $ADASHARE_SCRIPT $job_name  "$pbs_account"  "$pbs_allocate"  "$pbs_folders"\
                        $epochs $batch_size $num_layers $layer $lr $dev $dropout $datadir $outdir $config $seed_idx $res_opt $hdn_opt "$desc"  
        elif [[ $submit_type == "LOCAL" ]]; then
            job_dt=`date +%m%d.%H%M%S`
            job_name="$job_dt-$229-${layer}x${num_layers}-${dropout}"
            output_file="../pbs_output/${job_name}.out"
            . submit_local_job.sh $ADASHARE_SCRIPT $job_name  $output_file \
                        $epochs $batch_size $num_layers $layer $lr $dev $dropout $datadir $outdir $config $seed_idx $res_opt $hdn_opt "$desc"  
        fi

        # qsub $1 -N $job_name  $pbs_account  $pbs_allocate  $pbs_folders\
        # -v epochs=$epochs,batch_size=$batch_size,num_layers=$num_layers,layer=$layer,lr=$lr,dev=$dev,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,seed_idx=$seed_idx,res_opt=$3,hdn_opt=$4,desc="$5"
    done
    done
    done
    done
}

z``
echo "Epochs: $epochs  Layers: ${num_layers_list[@]}   Layer size: ${layer_size_list[@]}   Dropout: ${dropout_list[@]}  Task LR: ${lr_list[@]} "

if [ $1 == "res" ] || [ $1 == "both" ] 
then echo " Its res! -- submit pbs_tr_resid"
    job_prefix="AS"
    res_opt="False"
    hdn_opt="False"
    desc="Run with residiual layers"
    echo " desc:   $desc"
    submit_list $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
fi

if [ $1 == "nores" ] || [ $1 == "both" ] 
then echo " Its nores! -- submit pbs_tr_nores"
    job_prefix="NR"
    res_opt="True"
    hdn_opt="False"
    desc="Run without residiual layers"
    echo " desc:   $desc"
    submit_list $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
fi
 
if [ $1 == "nohidden" ]  
then echo " Its nohidden -- submit with --skip_hidden"
    job_prefix="NH"
    res_opt="False"
    hdn_opt="True"
    desc="Run without hidden layers"
    submit_list $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
fi


# num_layers_list=( 0 )
# layer_size_list=(1000) 
# layer_size_list=( 50 100 200 300 400  )
# dropout_list=(0.75 0.80 0.85 0.90 0.95)
# dropout_list=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
# dropout_list=(0.10 0.20 0.30 0.40 0.50 0.60)
# AS_NO_HIDDEN_LAYERS="AS_tr_nohidden.sh"    <-- obsoleted by AS_tr_resid
# AS_NO_RES_LAYERS="AS_tr_nores.sh"          <-- obsoleted by AS_tr_resid