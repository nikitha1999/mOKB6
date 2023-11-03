# mOKB6: A Multilingual Open Knowledge Base Benchmark

Implementing the best scores resulted Model

## mOKB6 Dataset
The `./mokb6/mono/` folder contains the mOKB6 dataset, containing six monolingual open KBs in six languages: 
1. English Open KB inside `./mokb6/mono/mono_en`
2. Hindi Open KB inside `./mokb6/mono/mono_hi`
3. Telugu Open KB inside `./mokb6/mono/mono_te`
4. Spanish Open KB inside `./mokb6/mono/mono_es`
5. Portuguese Open KB inside `./mokb6/mono/mono_pt`
6. Chinese Open KB inside `./mokb6/mono/mono_zh`

Each monolingual Open KB's folder contains three files: `train.txt`, `valid.txt`, and `test.txt`.
These files are the train-dev-test splits of the respective language's Open KB, which contain tab-separated Open IE triples of the form (subject, relation, object).

The translated Open KB facts are already provided.
Thus, for each baseline given in Table 3 in the paper, the corresponding dataset inside `./mokb6/` folder is provided.
For e.g., The best baseline (for all languages except English) called Union+Trans is trained using data contained in `./mokb6/union+trans/` for the 5 languages (`./mokb6/union+trans/union+trans_en2hi/` for Hindi).
Whereas the best performing baseline for English called Union can be reproduced using data contained in `./mokb6/union/`.

## Model
The code of SimKGC mBERT initialization model is in the repository (adapted from [Wang et al., 2022](https://aclanthology.org/2022.acl-long.295)) as it showed the best performance when compared with the other KGE models. 

## How to Run
Here, are the commands to train and get the scores for Union+Trans.

### Requirements

* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15

### Installation

```
conda create --name mokb python=3.7
conda activate mokb
pip install -r requirements.txt
```

### Preprocessing the dataset

First, run the below command to process the data of monolingual Open KBs. This is necessary and must to do to have the correct evaluation of any model, i.e., a given model will be evaluated on each language's Open KB (which will not include any translated facts).

```
sh sh_preprocess_mono_okbs.sh
```

Then, set the environment variables `baseline_name` and `baseline_data` to the name of baseline of interest, and to its corresponding data's path, respectively.

```
baseline_data=./mokb6/union+trans/union+trans_en2hi
baseline_name=union+trans_en2hi
```

Run the following commands to preprocess the data and store it in `./data/${baseline_name}/` folder.

```
python convert_format_mokb.py --train ${baseline_data}/train.txt   --val ${baseline_data}/valid.txt --test ${baseline_data}/test.txt  --out_dir ./data/${baseline_name}

python3 preprocess.py --train-path ./data/${baseline_name}/train.txt --valid-path ./data/${baseline_name}/valid.txt --test-path ./data/${baseline_name}/test.txt --task mopenkb
```

### Training
Set the environment variable `batch_size` as per the baseline (e.g., 128 for Mono baseline for all languages except English, and 256 for the remaining baselines).
```
batch_size=256
```

Run the below command to train SimKGC (mBERT) and store its best checkpoint in `./checkpoint/${baseline_name}/` folder.

```
python3 main.py --model-dir ./checkpoint/${baseline_name} --pretrained-model bert-base-multilingual-cased --pooling mean --lr 3e-5 --train-path ./data/${baseline_name}/train.txt.json  --valid-path ./data/${baseline_name}/valid.txt.json  --task mopenkb --batch-size ${batch_size} --print-freq 20 --additive-margin 0.02 --use-amp --use-self-negative --finetune-t --pre-batch 0 --epochs 100 --workers 3 --max-to-keep 0 --patience 10 --seed 2022
```

### Evaluating
To evaluate the model on a given language's Open KB's testset, say `hi`, set the `language` variable
```
language=hi
```

Then, evaluate the model checkpoint `./checkpoint/${baseline_name}/model_best.mdl` using the below command.

```
python3 evaluate.py --task mopenkb --pretrained-model bert-base-multilingual-cased --is-test --eval-model-path ./checkpoint/${baseline_name}/model_best.mdl --train-path data/mono_${language}/train.txt.json  --valid-path data/mono_${language}/test.txt.json
```

## FAQ
When you are facing any CUDA related issues, set the below another environment Variable:

```
export CUDA_LAUNCH_BLOCKING=1

```