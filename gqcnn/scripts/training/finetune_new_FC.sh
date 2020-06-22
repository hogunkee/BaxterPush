#!/bin/sh

python tools/finetune.py data/training/newFC FC-GQCNN-4.0-PJ --config_filename cfg/train_dex-net_4.0_fc_pj.yaml --model_dir models/FC-GQCNN-4.0-PJ --name Finetuned_FC
