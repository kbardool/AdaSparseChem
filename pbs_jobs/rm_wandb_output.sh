#!/bin/bash
find  ./wandb/ -name "run-2022*" -mmin +$1 -printf 'rm -rf ./wandb/%P \n' > del_wandb.sh