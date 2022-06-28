#!/bin/bash 
# parm=$1
if [ -z $1 ]; then
    echo "ERROR:: input parameter must be provided (res | nores | nohidden)"
    return 1
fi
arch_type=$1
submit_type="LOCAL"
# PBS Parameters =========================================================
# PBS -M kevin.bardool@kuleuven.be
# PBS -l pmem=5gb,
# PBS -l qos=debugging
# pbs_account="-A lp_symbiosys "
# pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
# pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=72:00:00 "
# pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=24:00:00 "
# pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=01:00:00 "
# pbs_allocate="-l nodes=1:ppn=4,walltime=01:00:00 "

# Config Parameters ======================================================
ADASHARE_SCRIPT="AS_profiler.sh" 
config="../yamls/chembl_cb29_train.yaml"
datadir="../../MLDatasets/chembl29"
outdir="../../experiments/AdaSparseChem-cb29"

# Training Parms =========================================================
lr_list=(0.001)
# lr_list=(0.0001)
batch_size=4096

num_layers_list=(4)
# num_layers_list=(1 2 3)
layer_size_list=(4000) 
dropout_list=(0.50)
 
epochs=3
seed_idx=1
py_threads=1
dev=2
# dropout_list=(0.40 0.50 0.60)
# dropout_list=( 0.70  0.80  0.90)
#=========================================================================

. submit_training_job.sh $arch_type $submit_type

# submit_job(){ 
#     for num_layers in ${num_layers_list[@]}; do                 
#     for layer in ${layer_size_list[@]}; do           
#     for lr in ${lr_list[@]} ; do    
#     for dropout in  ${dropout_list[@]}; do
#         job_dt=`date +%m%d.%H%M%S`
#         job_name="$job_dt-$229-${layer}x${num_layers}-${dropout}"
#         output_file="../pbs_output/${job_name}.out"
#         # printf "\n"
#         # printf "* seed: $seed_idx  output: $output_file \n"   
#         # printf "* Epochs: $epochs  Lyrs: $num_layers  Lyr sz: $layer  Dropout: $dropout  Task LR: $lr  device: $dev \n"
#         # printf "\n"
#         # printf " Program      : $1 \n"
#         # printf " job_prefix   : $2 \n"
#         # printf " Skip Residual: $res_opt \n"
#         # printf " Skip Hidden  : $hdn_opt \n"
#         # printf " exp_desc     : $desc \n"
#         # printf " pbs_account  : $pbs_account \n"
#         # printf " pbs_allocate : $pbs_allocate\n"
#         # printf " pbs_folders  : $pbs_folders \n"        
#         # printf " batch_size   : $batch_size  \n"
#         # printf " learning rt  : $lr  \n"
#         # printf " datadir      : $datadir \n"
#         # printf " outdir       : $outdir \n"
#         # printf " config       : $config \n"        

#         printf " JobName      : $job_name  Epochs: $epochs   LR: $lr   dev: $dev ---> "        
#         ## submit training program 
#         # . $1 > $output_file 2>&1 &
#         printf " $job_name  ($1) Submitted in background \n"
#         sleep 1
#     done
#     done
#     done
#     done
# }


# echo "Epochs: $epochs  Layers: ${num_layers_list[@]}   Layer size: ${layer_size_list[@]}   Dropout: ${dropout_list[@]}  Task LR: ${lr_list[@]} "

# if [ $1 == "res" ] || [ $1 == "both" ] 
# then echo " Its res! -- submit pbs_tr_resid"
#     job_prefix="AS"
#     res_opt="False"
#     hdn_opt="False"
#     desc="Run with residiual layers"
#     echo " desc:   $desc"
#     submit_job $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
# fi

# if [ $1 == "nores" ] || [ $1 == "both" ] 
# then echo " Its nores! -- submit pbs_tr_nores"
#     job_prefix="NR"
#     res_opt="True"
#     hdn_opt="False"
#     desc="Run without residiual layers"
#     echo " desc:   $desc"
#     submit_job $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
# fi

# if [ $1 == "nohidden" ]  
# then echo " Its nohidden -- submit with --skip_hidden"
#     job_prefix="NH"
#     res_opt="False"
#     hdn_opt="True"
#     desc="Run without hidden layers"
#     submit_job $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
# fi
