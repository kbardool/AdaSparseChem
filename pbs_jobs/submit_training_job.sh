#!/bin/bash 
# parm=$1
if [ -z $1 ]; then
    echo "ERROR:: input parameter must be provided (res | nores | both | nohidden)"
    return 1
fi
printf "\n"
printf "=================================================\n" 
printf " Submit $2 Training job -  $# parameters passed\n"
printf "=================================================\n" 
for ((x=1;x<=${#*};x+=1))
    do
        printf " Parm $x      : ${!x} \n"
    done
printf "=================================================\n" 
printf " ADASHARE_SCRIPT: $ADASHARE_SCRIPT \n"
printf " submit type    : $submit_type\n"
printf " Program        : $1 \n"
printf " job_prefix     : $2 \n"
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
# printf " job date     : $job_dt \n"        
# echo "=================================================" 
printf "\n"


submit_list(){ 
    for num_layers in ${num_layers_list[@]}; do                 
    for layer in ${layer_size_list[@]}; do                   
    for lr in ${lr_list[@]} ; do
    for dropout in  ${dropout_list[@]}; do
        # printf "\n"
        # printf "* seed: $seed_idx  output: $output_file \n"   
        # printf "* Epochs: $epochs  Lyrs: $num_layers  Lyr sz: $layer  Dropout: $dropout  Task LR: $lr  device: $dev \n"
        
        if [[ $submit_type == "PBS" ]]; then
            job_name="$229-${layer}x${num_layers}-${dropout}"
            printf " JobName      : $job_name  Epochs: $epochs   LR: $lr   dev: $dev ---> "
            # . submit_pbs_job.sh $ADASHARE_SCRIPT $job_name  "$pbs_account"  "$pbs_allocate"  "$pbs_folders"\
            #             $epochs $batch_size $num_layers $layer $lr $dev $dropout $datadir $outdir $config $seed_idx $res_opt $hdn_opt "$desc"  
            # qsub $1 -N $job_name  $pbs_account  $pbs_allocate  $pbs_folders\
            # -v epochs=$epochs,batch_size=$batch_size,num_layers=$num_layers,layer=$layer,lr=$lr,dev=$dev,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,seed_idx=$seed_idx,res_opt=$3,hdn_opt=$4,desc="$5"
            printf " PBS job $job_name Submitted in background \n"     

        elif [[ $submit_type == "LOCAL" ]]; then
            job_dt=`date +%m%d.%H%M%S`
            job_name="$job_dt-$229-${layer}x${num_layers}-${dropout}"
            output_file="../pbs_output/${job_name}.out" 
            profile_file="../pbs_output/${job_name}.stats" 
            printf " JobName      : $job_name  Epochs: $epochs   LR: $lr   dev: $dev ---> "
            ## submit training program 
            . $1 > $output_file 2>&1 &
            printf " $job_name  ($1) Submitted in background \n"            
        fi

    done
    done
    done
    done
}


echo "Epochs: $epochs  Layers: [${num_layers_list[@]}]   Layer size: [${layer_size_list[@]}]   Dropout: [${dropout_list[@]}]  Task LR: ${lr_list[@]} "

if [ $1 == "res" ] || [ $1 == "both" ] 
then  
    job_prefix="AS"
    res_opt="False"
    hdn_opt="False"
    desc="Run with residiual layers"
    echo "Run type:   $desc"
    submit_list $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
fi

if [ $1 == "nores" ] || [ $1 == "both" ] 
then  
    job_prefix="NR"
    res_opt="True"
    hdn_opt="False"
    desc="Run without residiual layers"
    echo "Run type:   $desc"
    submit_list $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
fi
 
if [ $1 == "nohidden" ]  
then  
    job_prefix="NH"
    res_opt="False"
    hdn_opt="True"
    desc="Run without hidden layers"
    echo "Run type:   $desc"
    submit_list $ADASHARE_SCRIPT $job_prefix $opt $hdn_opt "$desc"
fi

