#!/bin/bash
load_path="<path/to/checkpoint.ckpt>"
python eval.py --newTask 1 --load_path "$load_path"
