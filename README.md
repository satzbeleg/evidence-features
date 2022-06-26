[![PyPI version](https://badge.fury.io/py/evidence-features.svg)](https://badge.fury.io/py/evidence-features)
[![PyPi downloads](https://img.shields.io/pypi/dm/evidence-features)](https://img.shields.io/pypi/dm/evidence-features)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/satzbeleg/evidence-features.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/satzbeleg/evidence-features/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/satzbeleg/evidence-features.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/satzbeleg/evidence-features/context:python)

# evidence-features
Linguistic feature extraction.

## Usage

Table of Contents

* [Use Case 1](#use-case-1)


### Use Case 1


## Appendix

### Installation
The `evidence-features` [git repo](http://github.com/satzbeleg/evidence-features) is available as [PyPi package](https://pypi.org/project/evidence-features)

```sh
pip install evidence-features
pip install git+ssh://git@github.com/satzbeleg/evidence-features.git
```

### Install a virtual environment

```sh
python3.6 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
pip install -r requirements-dev.txt --no-cache-dir
pip install -r requirements-demo.txt --no-cache-dir
```

(If your git repo is stored in a folder with whitespaces, then don't use the subfolder `.venv`. Use an absolute path without whitespaces.)

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
