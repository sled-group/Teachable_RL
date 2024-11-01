#!/bin/bash
# Template Foresight
exp_name="msgr_adapt_no"
LanguageType="rhf"
data_size=5 # 5/10/20
load_ckpt="checkpoint/messenger/..." # Path of checkpoints of models pretrained with no/rh/rf/rhf/....
python train.py --data_size=$data_size --newTask 1 --exp_name "$exp_name" --LanguageType "$LanguageType" --load_ckpt "$load_ckpt"
