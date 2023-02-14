# evidence-features
Linguistic feature extraction for German (lang: de) as 8-bit interger representations.


## Install a virtual environment for CPU

```sh
# Ensure that python packages are availabe
sudo apt install python3.9-dev python3-venv

# install virtualenv
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# install other packages
pip install --use-pep517 -e .
# pip install --use-pep517 -r requirements.txt --no-cache-dir
pip install --use-pep517 -r requirements-dev.txt --no-cache-dir
pip install --use-pep517 -r requirements-demo.txt --no-cache-dir

# reinstall TF for better Intel-CPU support
# pip install intel-tensorflow
```

And, or install python package `evidence-features` from Github.

```sh
pip install git+ssh://git@github.com/satzbeleg/evidence-features.git
```

### Install MiniConda for GPU
In to ensure compatible CUDA drivers, use Conda to install them (Nvidia does not maintain PyPi packages).

```sh
# fasttext requires `python3.9-dev`
sudo apt install python3.9-dev
```

```sh
conda install -y pip
conda create -y --name gpu-venv-evidence-features python=3.9 pip
conda activate gpu-venv-evidence-features

conda install -y cudatoolkit=11.3.1 cudnn=8.3.2 -c conda-forge
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install torch==1.12.1+cu113 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html

# install other packages
pip install -e .
# pip install -r requirements.txt --no-cache-dir
pip install -r requirements-dev.txt --no-cache-dir
pip install -r requirements-demo.txt --no-cache-dir
```

Install MiniConda if not exists
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# prevent conda autostart in shell
# conda config --set auto_activate_base false
```


## Download pretrained models and statistics


### Download from original sources
The software uses pretrained NLP models and statistics.

```sh
# Ensure Debian packages are available
sudo apt install unzip p7zip-full

# some python package are called
conda activate gpu-venv-evidence-features
# source .venv/bin/activate

# set the location for pretrained models and other lists
export MODELFOLDER="$(pwd)/models"
# download
bash download-models.sh

# run tests
pytest
```

### Download via DVC Backend
If you have access to ZDL's DVC backend, run

```sh
dvc pull -r zdl
```

### Features Overview
Currently, 1024 binary and 157 floating-point features are extracted whcih can be stored as 293 int8 elements in a database.

| ID | Language level | Used models & statistics | Metrics | Memory |
|:---:|:---:|:---|:---|:---:|
| 1 | semantics | [SBert](http://dx.doi.org/10.18653/v1/D19-1410), `paraphrase-multilingual-MiniLM-L12-v2`; Hashed random projection | Contextual sentence embeddings | 1024-bit or 128x Int8 (128 bytes) |
| 2 | morphosyntax | [Trankit](http://dx.doi.org/10.18653/v1/2021.eacl-demos.10), `german-hdt`, [CoNLL-U UPOS](https://universaldependencies.org/u/pos/index.html) | Distribution of Part-of-Speech (PoS) tags of a sentence | 16+1x Int8 (17 bytes) |
| 3 | morphosyntax | [Trankit](http://dx.doi.org/10.18653/v1/2021.eacl-demos.10), `german-hdt`, [CoNNL-U Universal Features](https://universaldependencies.org/u/feat/index.html) | Distribution of other lexical and grammatical properties in a sentence | 47+1x Int8 (48 bytes) |
| 4 | syntax | [Trankit](http://dx.doi.org/10.18653/v1/2021.eacl-demos.10), `german-hdt`, dependency parser; [node-distance](https://doi.org/10.5281/zenodo.5747823) | The distribution of the shortest paths between all nodes (word tokens) within the dependency tree of a sentence; adjusted by the visual distance between words. | 21x Int8 (21 bytes) |
| 5 | phonetics | [epitran](https://aclanthology.org/L18-1429/), `deu-Latn`; [ipasymbols](https://pypi.org/project/ipasymbols/)  | The number of IPA-based consonant clusters within a sentence | 3+1x Int16 (8 bytes) |
| 6, 7 | morphology | [DeReChar](https://www.ids-mannheim.de/fileadmin/kl/derewo) | Distribution of character and character-level bi-gram frequencies | 6+1 + 10+1 Int16 (36 bytes) |
| 8 | lexicology | [COW](https://ids-pub.bsz-bw.de/frontdoor/index/index/year/2015/docId/3836) | Distribution of lemmata frequencies | 6+1x Int8 (7 bytes) |
| 9 | morphology | [SMOR](https://aclanthology.org/L04-1275/) | Occurence of a) all possible parsed variants (syntactial ambivalence), b) all possible unique lexemes (lexeme ambivalence), c) the longest possible lexeme (working memory for composita comprehension) | 14+1x Int8 (15 bytes) |
| 12 | - | - | Other statistics, e.g., text length | 2x Int16 (4 bytes) |
| 13 | semantics | [FastText language detection](https://fasttext.cc/docs/en/language-identification.html) | Proba. of language or dialect (de, nds, als, bar) or lang. groups (franconian, north germanic, anglo-friesian, romanic, slavic) | 10x Int8 (10 bytes) |
| 14 | semantics | [Emoji Sentiment](https://www.clarin.si/repository/xmlui/handle/11356/1048) | Distribution of emoji frequencies, pos., neg., and neutral sentiment for all emojis within a sentence | 22+1 Int8 (23 bytes) |

Not included in `.to_float()`, i.e., only the function `.to_int()` will return these features.

| ID | Language level | Used models & statistics | Metrics | Dim (as int8) |
|:---:|:---:|:---|:---|:---:|
| 15 | syntax | Uses [Trankit](http://dx.doi.org/10.18653/v1/2021.eacl-demos.10), `german-hdt`, dependency parser; [datasketch.MinHash](http://ekzhu.com/datasketch/minhash.html), [mmh3](https://pypi.org/project/mmh3/), and [treesimi](https://pypi.org/project/treesimi/) | MinHash/mmh3 hashes for syntatic similarity | 32x Int32 (128 bytes) |
| 16 | - | Uses [datasketch.MinHash](http://ekzhu.com/datasketch/minhash.html), [mmh3](https://pypi.org/project/mmh3/), and [kshingle](https://pypi.org/project/kshingle/) | MinHash/mmh3 hashes for Near Duplicate Detection | 32x Int32 (128 bytes) |
| 17 | - | List of headwords (lemmata) that NOUN, VERB or ADJ |  | List |
| 18 | - | same as 16; For hashing bibliographic information as simple string | same as 16 | 32x Int32 (128 bytes) |


### Int8 vs floating-point features 
All features are encoded as Int8 features.
Most features are count data or naturally integer numbers that are transformed to ratios lateron, i.e., we will save 8-bit integers instead of 32-bit floating-points.
In case of SBert wer compress the floating-point feature with hashed random projections to bit-values that are stored as Int8 representations - The storage requirement can be reduced by factor 12 to 16.



## Demo Scripts and Notebooks

### Correlation among features

```sh
conda activate gpu-venv-evidence-features 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# source .venv/bin/activate

export MODELFOLDER="$(pwd)/models"
cd demo/corr
bash download-corpora.sh
python3 preprocess.py
jupyter lab
```

### Benchmarking
[Sentence embedding evaluation for German](https://github.com/ulf1/sentence-embedding-evaluation-german)

```sh
conda activate gpu-venv-evidence-features 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
#source .venv/bin/activate

export MODELFOLDER="$(pwd)/models"
cd demo/benchmark
bash download-datasets.sh
nohup python3 run.py > log.log &
# CUDA_LAUNCH_BLOCKING=1  python3 run.py
tail -f log.log
watch -n 0.5 nvidia-smi
```

Balanced F1 scores on the test sets. 
*EV feats.* uses hashed random projections of SBert features,
i.e., the F1 scores are expected to decrease but not too much.

| Task | SBert | EV Feats |
|---:|---:|---:|
| FCLAIM | 0.672 | 0.632 |
|   VMWE | 0.751 | 0.730 |
| OL19-C | 0.611 | 0.598 |
| ABSD-2 | 0.521 | 0.528 |
|  MIO-P | 0.820 | 0.844 |
|  ARCHI | 0.374 | 0.357 |
|   LSDC | 0.396 | 0.406 |


## Appendix

### Python commands

* Jupyter for the examples: `jupyter lab`
* Check syntax: `flake8 --ignore=F401 --exclude=demo,$(grep -v '^#' .gitignore | xargs | sed -e 's/ /,/g')`
* Run Unit Tests: `PYTHONPATH=. pytest`

### Clean up 

```sh
find . -type f -name "*.pyc" | xargs rm
find . -type d -name "__pycache__" | xargs rm -r
rm -r .pytest_cache
rm -r .venv
```

### Citation
You can cite the following paper if you want to use this repository in your research work.

```
@inproceedings{hamster-2022-everybody,
    title = "Everybody likes short sentences - A Data Analysis for the Text Complexity {DE} Challenge 2022",
    author = "Hamster, Ulf A.",
    booktitle = "Proceedings of the GermEval 2022 Workshop on Text Complexity Assessment of German Text",
    month = sep,
    year = "2022",
    address = "Potsdam, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.germeval-1.2",
    pages = "10--14",
}
```

### Support
Please [open an issue](https://github.com/satzbeleg/evidence-features/issues/new) for support.


### Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/satzbeleg/evidence-features/compare/).
