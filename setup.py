import setuptools
import os


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as fp:
        s = fp.read()
    return s


def get_version(path):
    with open(path, "r") as fp:
        lines = fp.read()
    for line in lines.split("\n"):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name='evidence-features',
    version=get_version("evidence_features/__init__.py"),
    description=(
        "Linguistic feature extraction for German (lang: de) as 8-bit or"
        " 16-bit integer representations."
    ),
    long_description=read('README.md'),  # README.rst
    url='http://github.com/satzbeleg/evidence-features',
    author='Ulf Hamster',
    author_email='554c46@gmail.com',
    license='Apache License 2.0',
    packages=['evidence_features'],
    install_requires=[
        "numpy>=1.21.6,<2",
        "dvc[ssh]>=2,<3",
        "cassandra-driver>=3.25.0,<4",
        "sentence-transformers>=2.2.0,<3",
        "keras-hrp>=0.1.0,<1",
        "torch>=1,<2",
        "trankit>=1.1.1,<2",
        "node-distance>=0.1.0,<1",
        "epitran>=1.18,<2",
        "ipasymbols>=0.0.1,<1",
        "sfst-transduce>=1.0.1,<2",
        "nltk>=3.7,<4",
        "pandas>=1.3.5,<2",
        "fasttext>=0.9.2,<1",
        "treesimi>=0.2.0,<1",
        "datasketch>=1.5.8,<2",
        "mmh3>=3.0.0,<4",
        "kshingle>=0.10.0,<1"
    ],
    # scripts=['scripts/examplescript.py'],
    python_requires='>=3.7',
    zip_safe=True
)
