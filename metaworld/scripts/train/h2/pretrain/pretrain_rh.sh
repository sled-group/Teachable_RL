#!/bin/bash
# Template Foresight
exp_name="msgr_rh_h2_pretrain"
LanguageType="rh"
python train.py --newTask 0 --exp_name "$exp_name" --LanguageType "$LanguageType" --data_size 20000
