[![PyPI version](https://badge.fury.io/py/evidence-features.svg)](https://badge.fury.io/py/evidence-features)
[![PyPi downloads](https://img.shields.io/pypi/dm/evidence-features)](https://img.shields.io/pypi/dm/evidence-features)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/satzbeleg/evidence-features.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/satzbeleg/evidence-features/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/satzbeleg/evidence-features.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/satzbeleg/evidence-features/context:python)

# evidence-features
Linguistic feature extraction for German (lang: de) as 8-bit interger representations.


## Install a virtual environment
```sh
# install virtualenv
python3.7 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# install PyTorch with GPU support for Trankit
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
# install other packages
pip install -e .
# pip install -r requirements.txt --no-cache-dir
# pip install -r requirements-dev.txt --no-cache-dir
# pip install -r requirements-demo.txt --no-cache-dir
```

And, or install python package `evidence-features` from Github.

```sh
pip install git+ssh://git@github.com/satzbeleg/evidence-features.git
```


## Download pretrained models and statistics
The software uses pretrained NLP models and statistics.

```sh
source .venv/bin/activate
export MODELFOLDER="$(pwd)/models"
bash download-models.sh
```

If you have access to ZDL's DVC backend, run

```sh
dvc pull
```

| Language level | Used models & statistics | Metrics |
|:---:|:---|:---|
| semantics | [SBert](http://dx.doi.org/10.18653/v1/D19-1410), `paraphrase-multilingual-MiniLM-L12-v2`; Hashed random projection | Contextual sentence embeddings |
| syntax | [Trankit](http://dx.doi.org/10.18653/v1/2021.eacl-demos.10), `german-hdt`, dependency parser; [node-distance](https://doi.org/10.5281/zenodo.5747823) | |
| morphosyntax | [Trankit](http://dx.doi.org/10.18653/v1/2021.eacl-demos.10), `german-hdt`, [CoNLL-U UPOS](https://universaldependencies.org/u/pos/index.html) | Part-of-Speech (PoS) tags |
| morphosyntax | [Trankit](http://dx.doi.org/10.18653/v1/2021.eacl-demos.10), `german-hdt`, [CoNNL-U Universal Features](https://universaldependencies.org/u/feat/index.html) | Other lexical and grammatical properties |
| phonetics | [epitran](https://aclanthology.org/L18-1429/), `deu-Latn`; [ipasymbols](https://pypi.org/project/ipasymbols/)  | IPA-based consonant clusters |
| morphology | [DeReChar](https://www.ids-mannheim.de/fileadmin/kl/derewo) | Character and character-level bi-gram frequencies |
| lexicology | [COW](https://ids-pub.bsz-bw.de/frontdoor/index/index/year/2015/docId/3836) | Lemma frequencies |
| morphology | [SMOR](https://aclanthology.org/L04-1275/) | Number of morphemes |
| - | - | Other statistics, e.g., text length |
| - | - | [FastText language detection](https://fasttext.cc/docs/en/language-identification.html) | Proba. of language or lang. group |


## Int8 vs floating-point features 
All features are encoded as Int8 features.
Most features are count data or naturally integer numbers that are transformed to ratios lateron, i.e., we will save 8-bit integers instead of 32-bit floating-points.
In case of SBert wer compress the floating-point feature with hashed random projections to bit-values that are stored as Int8 representations - The storage requirement can be reduced by factor 12 to 16.



## Correlation among features

```sh
source .venv/bin/activate
export MODELFOLDER="$(pwd)/models"
cd demo/corr
bash download-corpora.sh
python3 preprocess.py
jupyter lab
```

### Benchmarking
[Sentence embedding evaluation for German](https://github.com/ulf1/sentence-embedding-evaluation-german)

```sh
source .venv/bin/activate
export MODELFOLDER="$(pwd)/models"
cd demo/benchmark
bash download-datasets.sh
nohup python3 run.py > log.log &
tail -f log.log
```

Balanced F1 scores on the test sets. 
*EV feats.* uses hashed random projections of SBert features,
i.e., the F1 scores are expected to decrease but not too much.
Interestingly, Lower German dialect detection (LSDC) does not work at all with *EV feats.*. 

| Task | SBert | EV Feats |
|---:|---:|---:|
| FCLAIM | 0.672 | 0.634 |
|   VMWE | 0.751 | 0.729 |
| OL19-C | 0.611 | 0.591 |
| ABSD-2 | 0.521 | 0.514 |
|  MIO-P | 0.820 | 0.833 |
|  ARCHI | 0.374 | 0.365 |
|   LSDC | 0.396 | 0.007 |


## Appendix

### Python commands

* Jupyter for the examples: `jupyter lab`
* Check syntax: `flake8 --ignore=F401 --exclude=demo,$(grep -v '^#' .gitignore | xargs | sed -e 's/ /,/g')`
* Run Unit Tests: `PYTHONPATH=. pytest`

Publish

```sh
pandoc README.md --from markdown --to rst -s -o README.rst
python setup.py sdist 
twine upload -r pypi dist/*
```

### Clean up 

```sh
find . -type f -name "*.pyc" | xargs rm
find . -type d -name "__pycache__" | xargs rm -r
rm -r .pytest_cache
rm -r .venv
```


### Support
Please [open an issue](https://github.com/satzbeleg/evidence-features/issues/new) for support.


### Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/satzbeleg/evidence-features/compare/).
