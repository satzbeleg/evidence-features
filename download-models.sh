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
  python -c "import sentence_transformers; import torch; sentence_transformers.SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
fi

# trankit
if [ ! -d "${MODELFOLDER2}/trankit" ]; then
  mkdir -p "${MODELFOLDER2}/trankit"
  python -c "import trankit; import torch; trankit.Pipeline(lang='german-hdt', gpu=torch.cuda.is_available(), cache_dir='${MODELFOLDER2}/trankit')"
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

# DeReChar frequencies
if [ ! -f "${MODELFOLDER2}/derechar.txt" ]; then
  wget -nc -q "https://www.ids-mannheim.de/fileadmin/kl/derewo/DeReChar-v-bi-DRC-2021-10-31-1.0.txt" -O "derechar.txt"
fi

# FastText 176 model
if [ ! -f "${MODELFOLDER2}/lid.176.ftz" ]; then
  wget -nc -q "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz" -O "lid.176.ftz"
fi

# Emoji Sentiment
if [ ! -f "${MODELFOLDER2}/emoji-sentiment.csv" ]; then
  wget -nc -q "https://www.clarin.si/repository/xmlui/handle/11356/1048/allzip" -O "emoji-sentiment.zip"
  unzip -n "emoji-sentiment.zip"
  rm "emoji-sentiment.zip"
  rm "ESR_v1.0_format.txt"
  mv "Emoji_Sentiment_Data_v1.0.csv" "emoji-sentiment.csv"
  mv "Emojitracker_20150604.csv" "emoji-frequency.csv"
fi

# try last
# COW lemma frequencies
if [ ! -f "${MODELFOLDER2}/decow.csv" ]; then
  wget -nc -q "https://nlp-data-filestorage.s3.eu-central-1.amazonaws.com/word-frequencies/decow_wordfreq_cistem.csv.7z"
  p7zip -d "decow_wordfreq_cistem.csv.7z"
  mv decow_wordfreq_cistem.csv decow.csv
  rm decow_wordfreq_cistem.csv.7z
fi

echo "Done"
