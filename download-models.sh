#!/bin/bash

if [[ -z "${MODELFOLDER}" ]]; then
  MODELFOLDER2="./models"
else
  MODELFOLDER2="${MODELFOLDER}"
fi

mkdir -p "${MODELFOLDER2}"
echo "Save models and lists in: ${MODELFOLDER2}"


# SBert
if [ ! -d "${MODELFOLDER2}/sbert" ]; then
  export SENTENCE_TRANSFORMERS_HOME="${MODELFOLDER2}/sbert"
  mkdir -p "${SENTENCE_TRANSFORMERS_HOME}"
  python -c "import sentence_transformers; sentence_transformers.SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
fi

# trankit
if [ ! -d "${MODELFOLDER2}/trankit" ]; then
  mkdir -p "${MODELFOLDER2}/trankit"
  python -c "import trankit; trankit.Pipeline(lang='german-hdt', gpu=False, cache_dir='${MODELFOLDER2}/trankit')"
fi


# go to directory
cd "${MODELFOLDER2}"

# SMOR parser
if [ ! -f "${MODELFOLDER2}/smor.a" ]; then
  wget -nc -q "https://www.cis.uni-muenchen.de/~schmid/tools/SMOR/data/SMOR-linux.zip"
  unzip -n "SMOR-linux.zip"
  mv SMOR/lib/smor.a smor.a
  rm -rf SMOR/
  rm SMOR-linux.zip
fi

# COW lemma frequencies
if [ ! -f "${MODELFOLDER2}/decow.csv" ]; then
  wget -nc -q "https://nlp-data-filestorage.s3.eu-central-1.amazonaws.com/word-frequencies/decow_wordfreq_cistem.csv.7z"
  p7zip -d "decow_wordfreq_cistem.csv.7z"
  mv decow_wordfreq_cistem.csv decow.csv
  rm decow_wordfreq_cistem.csv.7z
fi

# DeReChar frequencies
if [ ! -f "${MODELFOLDER2}/derechar.txt" ]; then
  wget -nc -q "https://www.ids-mannheim.de/fileadmin/kl/derewo/DeReChar-v-bi-DRC-2021-10-31-1.0.txt" -O "derechar.txt"
fi

echo "Done"
