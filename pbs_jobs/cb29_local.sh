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
TRAINING_SCRIPT="AS_train.sh" 
config="../yamls/chembl_cb29_train_1task.yaml"
datadir="../../MLDatasets/chembl29"
# outdir="../../experiments/AdaSparseChem-cb29"

# Training Parms =========================================================
# lr_list=(0.001)
lr_list=(0.001)
batch_size=4096

layer_size_list=(4000) 
num_layers_list=(2)
# num_layers_list=(1 2 3)
dropout_list=(0.80)
 
epochs=150
seed_idx=1
## Kusanagi: 0 --> 1, 1 --> 2; 2 --> 0
cuda_device_id=1
py_threads=6
dev=0
# dropout_list=(0.40 0.50 0.60)
# dropout_list=( 0.70  0.80  0.90)
#=========================================================================

. submit_training_job.sh  $arch_type  $submit_type

