#!/bin/bash 

module load worker/1.6.12-intel-2018a
wsub  -data data.txt -batch worker_tr_resid.sh 
#  -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr