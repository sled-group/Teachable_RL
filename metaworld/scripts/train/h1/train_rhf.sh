#!/bin/bash
# GPT_augmented hindsight + foresight
exp_name="msgr_rhf_h1"
LanguageType="rhf"
python train.py --newTask 0 --exp_name "$exp_name" --LanguageType "$LanguageType" --data_size 20000
