# ALFWorld
For training data and checkpoints, please refer to [Data & Model Checkpoints](https://huggingface.co/datasets/sled-umich/Teachable_RL). Please fill in OpenAI key in eval.py for online gpt access. 
## RQ 1

### Train
GPT-augmented H + F:
```
python train.py --rq 1 --load_data_path ./data/rq1/alfworld-gpt_pool-h_f-rq1.pkl --save <path/to/save/model>
```
Template H + F:
```
python train.py --rq 1 --load_data_path ./data/rq1/alfworld-template-h_f-rq1.pkl --save <path/to/save/model>
```
Template F:
```
python train.py --rq 1 --load_data_path ./data/rq1/alfworld-template-f-rq1.pkl --save <path/to/save/model>
```
Template H:
```
python train.py --rq 1 --load_data_path ./data/rq1/alfworld-template-h-rq1.pkl --save <path/to/save/model>
```
No Lang:
```
python train.py --rq 1 --load_data_path ./data/rq1/alfworld-no-lang-rq1.pkl --save <path/to/save/model>
```

### Evaluation
```
python eval.py --rq 1 --load_model_path <path/to/model> --model_iter <optional> 
```
If 'model_iter' is not provided, 'ckpt.pt' will be loaded.

## RQ 2

### Pretrain
GPT-augmented H + F:
```
python train.py --rq 1 --load_data_path ./data/rq2/pretrain/alfworld-gpt_pool-h_f-rq2-pretrain.pkl --save <path/to/save/model>
```
GPT-augmented F:
```
python train.py --rq 1 --load_data_path ./data/rq2/pretrain/alfworld-gpt_pool-f-rq2-pretrain.pkl --save <path/to/save/model>
```
GPT-augmented H:
```
python train.py --rq 1 --load_data_path ./data/rq2/pretrain/alfworld-gpt_pool-h-rq2-pretrain.pkl --save <path/to/save/model>
```
No Lang:
```
python train.py --rq 1 --load_data_path ./data/rq2/pretrain/alfworld-no-lang-rq2-pretrain.pkl --save <path/to/save/model>
```

### Adaptation
GPT-augmented H + F:
```
python train.py --rq 2 --load_model_path <path/to/pretrain-model> --model_iter <optional> --shot <#shot> --load_data_path ./data/rq2/adaptation/alfworld-gpt_pool-h_f-rq2-<#shot>.pkl --num_steps_per_iter 100 --save <path/to/save/model>
```
GPT-augmented F:
```
python train.py --rq 2 --load_model_path <path/to/pretrain-model> --model_iter <optional> --shot <#shot> --load_model_path <path/to/pretrain-model> --load_data_path ./data/rq2/adaptation/alfworld-gpt_pool-f-rq2-<#shot>.pkl --num_steps_per_iter 100 --save <path/to/save/model>
```
GPT-augmented H:
```
python train.py --rq 2 --load_model_path <path/to/pretrain-model> --model_iter <optional> --shot <#shot> --load_model_path <path/to/pretrain-model> --load_data_path ./data/rq2/adaptation/alfworld-gpt_pool-h-rq2-<#shot>.pkl --num_steps_per_iter 100 --save <path/to/save/model>
```
No Lang:
```
python train.py --rq 2 --load_model_path <path/to/pretrain-model> --model_iter <optional> --shot <#shot> --load_model_path <path/to/pretrain-model> --load_data_path ./data/rq2/adaptation/alfworld-no-lang-rq2-<#shot>.pkl --num_steps_per_iter 100 --save <path/to/save/model>
```

### Evaluation
```
python eval.py --rq 2 --load_model_path <path/to/model> --model_iter <optional> 
```
If 'model_iter' is not provided, 'ckpt.pt' will be loaded.