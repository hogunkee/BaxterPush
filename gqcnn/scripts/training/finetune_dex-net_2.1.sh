#!/bin/sh

python tools/finetune.py data/training/gq2.1_5000 GQCNN-2.1 --config_filename cfg/train_dex-net_2.1.yaml --model_dir models/GQ-Bin-Picking-Eps90 --name Finetuned_2.1
#python tools/finetune.py data/training/gq2.1_5000 GQCNN-2.1 --config_filename cfg/train_dex-net_2.1.yaml --model_dir models/GQ-Bin-Picking-Eps90 --name Finetuned_2.1
#python tools/finetune.py data/training/gq2.1_5000 GQCNN-2.1 --config_filename cfg/finetune.yaml --model_dir models/GQ-Bin-Picking-Eps90 --name Finetuned_2.1
#python tools/finetune.py data/training/dexnet_2.1_eps_10 GQCNN-2.1 --config_filename cfg/train_dex-net_2.1.yaml --model_dir models/GQ-Bin-Picking-Eps90 --name Test-2.1
