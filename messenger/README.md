# Messenger_Environment
For training data and checkpoints, please refer to [Data & Model Checkpoints](https://huggingface.co/datasets/sled-umich/Teachable_RL). Please fill in OpenAI key in evaluate.py for online gpt access. 

## LanguageType
<p align="center">

| LanguageType | Meaning           |
|--------------|-------------------|
| no           | No Language       |
| h            | Hindsight         |
| f            | Foresight         |
| hf           | Hindsight & Foresight |
| rh           | GPT-augmented Hindsight |
| rf           | GPT-augmented Foresight |
| rhf          | GPT-augmented Hindsight & Foresight |

</p>

## Training
### Train models for Hypothesis 1

```
source env.sh
source scripts/train/h1/train_{LanguageType}.sh
```
### Train Models for Hypothesis 2
### Pretraining
```
source env.sh
source scripts/train/h2/pretrain/pretrain_{LanguageType}.sh
```
### Adaptation
Please modify the load_ckpt path and the shot_number in the script.
```
source env.sh
source scripts/train/h2/adapt.sh
```

### Evaluation
### Evaluation for Hypothesis 1
```
source env.sh
source scripts/eval/eval_h1.sh
```
### Evaluation for Hypothesis 2
```
source env.sh
source scripts/eval/eval_h2.sh
```