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

## PreRequisite
Conda environment is needed for the implementation.
```

conda create --name mokb python=3.7 -y
conda activate mokb
```

## How to Run
Here, are the commands to train and get the scores for Union+Trans.

From packages requirements to Preprocessing, Training, Testing are written in a shell script.
Just execute the run.sh file in the required GPU Resources.

```
sh run.sh

or 

run.sh
```

## To Checking Robustness:

Use the below commands to view the results after the model got trained for Test files.
Perturbed Data is available in MONO_EN Folder.

PreProcessing Perturbed File:
```
python convert_format_mokb.py --train ${baseline_data}/train.txt --val ${baseline_data}/valid.txt --test ${baseline_data}/negation.txt --out_dir ./data/${baseline_name}
```
* Replace negation.txt with required perturbed file name in the above command.

```
python3 preprocess.py --train-path ./data/${baseline_name}/train.txt --valid-path ./data/${baseline_name}/valid.txt --test-path ./data/${baseline_name}/test.txt --task mopenkb
```

Evaluating File:
```
python3 evaluate.py --task mopenkb --pretrained-model bert-base-multilingual-cased --is-test --eval-model-path ./checkpoint/${baseline_name}/model_best.mdl --train-path data/mono_${language}/train.txt.json --valid-path data/mono_${language}/test.txt.json
```

## FAQ
Facing issue, while running the run.sh due to conda environment(Applicable to the Linux Environments):

Enable conda using below commands:

```
source $HOME/miniconda/bin/activate
export PYTHONNOUSERSITE=true
```

When you are facing any CUDA related issues, set the below another environment Variable:

```
export CUDA_LAUNCH_BLOCKING=1
```

When unable to execute run.sh file due to permission issues:

For permission:
```
chmod 744 *.sh
```

To run:
```
./run.sh
```
