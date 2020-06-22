#!/bin/sh

python tools/finetune.py data/training/gq2.0_10000 GQCNN-2.0 --config_filename cfg/train_dex-net_2.0.yaml --model_dir models/GQ-ImageWise --name Finetuned_2.0
#python tools/finetune.py data/training/gq2.0_collected_data GQCNN-2.0 --config_filename cfg/train_dex-net_2.0.yaml --model_dir models/GQ-ImageWise --name Dexnet-2.0-2000data
