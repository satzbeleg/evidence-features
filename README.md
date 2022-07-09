[![PyPI version](https://badge.fury.io/py/evidence-features.svg)](https://badge.fury.io/py/evidence-features)
[![PyPi downloads](https://img.shields.io/pypi/dm/evidence-features)](https://img.shields.io/pypi/dm/evidence-features)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/satzbeleg/evidence-features.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/satzbeleg/evidence-features/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/satzbeleg/evidence-features.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/satzbeleg/evidence-features/context:python)

# evidence-features
Linguistic feature extraction for German (lang: de) as 8-bit interger representations.


## Install a virtual environment
```sh
python3.7 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# pip install -r requirements.txt --no-cache-dir
# pip install -r requirements-dev.txt --no-cache-dir
# pip install -r requirements-demo.txt --no-cache-dir
pip install -e .
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


## Int8 vs floating-point features 
All features are 



## Correlation among features

```sh
source .venv/bin/activate
export MODELFOLDER="$(pwd)/models"
cd demo/corr
bash download-corpora.sh
python3 preprocess.py
python3 train.py
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


## Appendix

### Python commands

* Jupyter for the examples: `jupyter lab`
* Check syntax: `flake8 --ignore=F401 --exclude=$(grep -v '^#' .gitignore | xargs | sed -e 's/ /,/g')`
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
