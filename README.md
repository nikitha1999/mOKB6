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

From packages requirements to Preprocessing, Training, Testing are written in a shell script.
Just execute the run.sh file in the required GPU Resources.

```
sh run.sh

or 

run.sh
```

## FAQ
When you are facing any CUDA related issues, set the below another environment Variable:

```
export CUDA_LAUNCH_BLOCKING=1
```

When unable to execute run.sh file:

```
Chmod 744 *.sh
./run.sh
```