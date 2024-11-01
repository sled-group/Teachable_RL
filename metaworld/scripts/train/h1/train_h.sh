#!/bin/bash
# Template Hindsight
exp_name="msgr_h_h1"
LanguageType="h"
python train.py --newTask 0 --exp_name "$exp_name" --LanguageType "$LanguageType" --data_size 20000
