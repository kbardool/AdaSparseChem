## python src/dev/train_dev.py --config yamls/adashare/chembl_2task.yaml --cpu --batch_size 09999

python ./src/train.py --config yamls/chembl_3task_train.yaml \
                    --exp_desc    6 lyrs,dropout 0.5, weight 105 bch/ep policy 105 bch/ep \
                    --hidden_size   50 50 50 50 50 50  \
                    --tail_hidden_size    50  \
                    --seed_idx             0  \
                    --batch_size         128  \
                    --task_lr           0.01  \
                    --backbone_lr       0.01  \
                    --policy_lr         0.01  \
                    --lambda_sparsity   0.02  \
                    --lambda_sharing    0.01  
