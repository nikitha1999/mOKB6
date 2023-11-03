#!/bin/bash

# Exit script on error
set -e

# Uncomment while using University GPU Resouces - these commands are to activate Conda Environment
# source $HOME/miniconda/bin/activate
# export PYTHONNOUSERSITE=true

# Create a conda environment
conda create --name mokb python=3.7 -y
conda activate mokb

# Install the requirements
pip install -r requirements.txt

# Run preprocessing
sh sh_preprocess_mono_okbs.sh

# Set variables for baseline data and name
baseline_data="./mokb6/union+trans/union+trans_en2hi"
baseline_name="union+trans_en2hi"

# Convert format for mOKB
python convert_format_mokb.py \
  --train ${baseline_data}/train.txt \
  --val ${baseline_data}/valid.txt \
  --test ${baseline_data}/test.txt \
  --out_dir ./data/${baseline_name}

# Preprocess data for mopenkb
python3 preprocess.py \
  --train-path ./data/${baseline_name}/train.txt \
  --valid-path ./data/${baseline_name}/valid.txt \
  --test-path ./data/${baseline_name}/test.txt \
  --task mopenkb

# Set batch size
batch_size=256

# Main training script
python3 main.py \
  --model-dir ./checkpoint/${baseline_name} \
  --pretrained-model bert-base-multilingual-cased \
  --pooling mean \
  --lr 3e-5 \
  --train-path ./data/${baseline_name}/train.txt.json \
  --valid-path ./data/${baseline_name}/valid.txt.json \
  --task mopenkb \
  --batch-size ${batch_size} \
  --print-freq 20 \
  --additive-margin 0.02 \
  --use-amp \
  --use-self-negative \
  --finetune-t \
  --pre-batch 0 \
  --epochs 100 \
  --workers 3 \
  --max-to-keep 0 \
  --patience 10 \
  --seed 2022

# Set language variable
language="hi"

# Evaluation script
python3 evaluate.py \
  --task mopenkb \
  --pretrained-model bert-base-multilingual-cased \
  --is-test \
  --eval-model-path ./checkpoint/${baseline_name}/model_best.mdl \
  --train-path data/mono_${language}/train.txt.json \
  --valid-path data/mono_${language}/test.txt.json
