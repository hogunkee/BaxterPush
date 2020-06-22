#!/bin/sh

python tools/finetune.py data/training/gq2.0_1000 GQCNN-2.0 --config_filename cfg/train_dex-net_2.0.yaml --model_dir models/GQ-ImageWise --name Finetuned_2.0_1000data
