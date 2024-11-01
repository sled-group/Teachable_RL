#!/bin/bash
# Template Foresight
exp_name="msgr_rhf_h2_pretrain"
LanguageType="rhf"
python train.py --newTask 0 --exp_name "$exp_name" --LanguageType "$LanguageType" --data_size 20000
