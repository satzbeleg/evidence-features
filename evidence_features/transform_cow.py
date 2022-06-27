import os
import nltk
import pandas as pd
import numpy as np
import gc
from typing import List
import json
from .utils import divide_by_1st_col

# path to the COW datasets
MODELPATH = os.getenv("MODELFOLDER", "./models")
MODELPATH1 = os.path.join(MODELPATH, "decow.csv")
MODELPATH2 = os.path.join(MODELPATH, "decow.json")


# load NLTK stemmer
stemmer = nltk.stem.Cistem()


# load COW frequency list
if os.path.isfile(MODELPATH2):
    with open(MODELPATH2, "r") as fp:
        decow = json.load(fp)
else:
    df_decow = pd.read_csv(MODELPATH1, index_col=['word'])
    df_decow = df_decow[df_decow["freq"] > 100]  # removes ~97% of rows!
    df_decow = np.log(df_decow + 1.)
    df_decow = df_decow / df_decow.max()
    decow = {row[0]: row[1].values[0] for row in df_decow.iterrows()}
    del df_decow
    gc.collect()
    with open(MODELPATH2, "w") as fp:
        json.dump(decow, fp)


brackets = np.percentile(
    [num for _, num in decow.items()],
    q=[100 / 6, 100 / 3, 50, 200 / 3, 500 / 6, 100])


def cow_to_float(sentences: List[str]):
    feats = cow_to_int8(sentences)
    return divide_by_1st_col(feats)


def cow_to_int8(sentences: List[str]):
    feats = []
    for sent in sentences:
        # whitspace tokenization
        words = sent.split(" ")
        # lookup frequency
        freqs = [decow.get(stemmer.stem(w), 0) for w in words]
        # assign to decentile
        quantiles = [np.argmax(f <= brackets) for f in freqs]
        # count decentiles
        cnt = np.zeros((len(brackets),), dtype=np.int8)
        for q in quantiles:
            cnt[q] += 1
        # save
        feats.append((
            len(freqs),  # num words
            *cnt.tolist()
        ))
    # done
    return np.vstack(feats).astype(np.int8)
