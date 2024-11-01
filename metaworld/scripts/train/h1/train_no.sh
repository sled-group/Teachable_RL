#!/bin/bash
# No Language
exp_name="msgr_no_h1"
LanguageType="no"
python train.py --newTask 0 --exp_name "$exp_name" --LanguageType "$LanguageType" --data_size 20000
