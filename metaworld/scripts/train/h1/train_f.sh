#!/bin/bash
# Template Foresight
exp_name="msgr_f_h1"
LanguageType="f"
python train.py --newTask 0 --exp_name "$exp_name" --LanguageType "$LanguageType" --data_size 20000
