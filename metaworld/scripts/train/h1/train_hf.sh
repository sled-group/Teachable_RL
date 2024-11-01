#!/bin/bash
# Template Hindsight + Foresight
exp_name="msgr_hf_h1"
LanguageType="hf"
python train.py --newTask 0 --exp_name "$exp_name" --LanguageType "$LanguageType" --data_size 20000
