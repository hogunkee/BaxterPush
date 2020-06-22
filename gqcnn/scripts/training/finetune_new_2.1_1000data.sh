#!/bin/sh

python tools/finetune.py data/training/new2.1_1000 GQCNN-2.1 --config_filename cfg/train_dex-net_2.1.yaml --model_dir models/GQ-Bin-Picking-Eps90 --name New_2.1_1000data
