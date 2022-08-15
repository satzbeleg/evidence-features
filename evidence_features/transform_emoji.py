import os
import pandas as pd
import gc
from typing import List
from .utils import divide_by_1st_col
import numpy as np


# path to the pretrained epitran model
MODELPATH = os.getenv("MODELFOLDER", "./models")
# real files
df_senti = pd.read_csv(os.path.join(MODELPATH, "emoji-sentiment.csv"))
df_freq = pd.read_csv(os.path.join(MODELPATH, "emoji-frequency.csv"))

# get emoji types
emoji_types = sorted(list(df_senti["Unicode block"].unique()))

# compute sentiments as ratios, and emoji type
emojis = {}
for i, row in df_senti.iterrows():
    emojis[row["Emoji"]] = {
        "neg": row["Negative"] / row["Occurrences"],
        "neut": row["Neutral"] / row["Occurrences"],
        "pos": row["Positive"] / row["Occurrences"],
        "typ": emoji_types.index(row["Unicode block"])
    }
# relative emoji frequency
total = df_freq["Occurrences"].astype(float).max()
for i, row in df_freq.iterrows():
    if row["Emoji"] in emojis:
        emojis[row["Emoji"]]["freq"] = row["Occurrences"] / total

del df_senti, df_freq, emoji_types, total
gc.collect()


# brackets_neg = np.percentile(
#     [val.get("neg") for _, val in emojis.items() if val.get("neg") > 0.01],
#     q=[100 / 6, 100 / 3, 50, 200 / 3, 500 / 6, 100])
brackets_neg = np.array([
    0.05882353, 0.1, 0.14285714, 0.2173913, 0.38605479, 1.])

# brackets_pos = np.percentile(
#     [val.get("pos") for _, val in emojis.items() if val.get("pos") > 0.01],
#     q=[100 / 6, 100 / 3, 50, 200 / 3, 500 / 6, 100])
brackets_pos = np.array([0.28449328, 0.4, 0.5, 0.6, 0.71685606, 1.])

# brackets_neut = np.percentile(
#     [val.get("neut") for _, val in emojis.items() if val.get("neut") > 0.01],
#     q=[100 / 6, 100 / 3, 50, 200 / 3, 500 / 6, 100])
brackets_neut = np.array([0.23561243, 0.3, 0.4 , 0.5, 0.69397993, 1.])

# brackets_freq = np.percentile(
#     [val.get("freq", 0.0) for _, val in emojis.items() 
#      if val.get("freq", 0.0) > 1e-6],
#     q=[25, 50, 75, 100])
brackets_freq = np.array([
    6.54701449e-04, 1.98708925e-03, 8.48139347e-03, 1.0])



def emoji_to_int8(sentences: List[str]):
    feats = []
    for sent in sentences:
        # lookup sentiment scores, and frequency
        score_neg = [emojis.get(c, {}).get("neg", 0.0) for c in sent]
        score_pos = [emojis.get(c, {}).get("pos", 0.0) for c in sent]
        score_neut = [emojis.get(c, {}).get("neut", 0.0) for c in sent]
        freqs = [emojis.get(c, {}).get("freq", None) for c in sent]
        # remove all scores below 0.01
        score_neg = [s for s in score_neg if s > 0.01]
        score_pos = [s for s in score_pos if s > 0.01]
        score_neut = [s for s in score_neut if s > 0.01]
        freqs = [f for f in freqs if f is not None]
        # assign to decentile
        quantiles_neg = [np.argmax(s <= brackets_neg) for s in score_neg]
        quantiles_pos = [np.argmax(s <= brackets_pos) for s in score_pos]
        quantiles_neut = [np.argmax(s <= brackets_neut) for s in score_neut]
        quantiles_freq = [np.argmax(f <= brackets_freq) for f in freqs]
        # count decentiles neg
        cnt_neg = np.zeros((len(brackets_neg),), dtype=np.int8)
        for q in quantiles_neg:
            cnt_neg[q] += 1
        # count decentiles pos
        cnt_pos = np.zeros((len(brackets_pos),), dtype=np.int8)
        for q in quantiles_pos:
            cnt_pos[q] += 1
        # count decentiles neut
        cnt_neut = np.zeros((len(brackets_neut),), dtype=np.int8)
        for q in quantiles_neut:
            cnt_neut[q] += 1
        # count decentiles
        cnt_freq = np.zeros((len(brackets_freq),), dtype=np.int8)
        for q in quantiles_freq:
            cnt_freq[q] += 1
        # save
        feats.append((
            len(sent.split(" ")),  # num tokens
            *cnt_neg.tolist(),
            *cnt_pos.tolist(),
            *cnt_neut.tolist(),
            *cnt_freq.tolist()
        ))
    # done
    feats = np.maximum(np.iinfo(np.int8).min, feats)
    feats = np.minimum(np.iinfo(np.int8).max, feats)
    return np.vstack(feats).astype(np.int8)


def emoji_to_float(sentences: List[str]):
    feats = emoji_to_int8(sentences)
    return divide_by_1st_col(feats)


def emoji_names():
    nam1 = [f"emoji_neg_{j}" for j in [16, 33, 50, 67, 83, 100]]
    nam2 = [f"emoji_pos_{j}" for j in [16, 33, 50, 67, 83, 100]]
    nam3 = [f"emoji_neut_{j}" for j in [16, 33, 50, 67, 83, 100]]
    nam4 = [f"emoji_freq_{j}" for j in [25, 50, 75, 100]]
    return nam1 + nam2 + nam3 + nam4
