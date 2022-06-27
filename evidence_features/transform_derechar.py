import os
import gc
from typing import List
import numpy as np

# path to the pretrained epitran model
MODELPATH = os.getenv("MODELFOLDER", "./models")
MODELPATH = os.path.join(MODELPATH, "derechar.txt")


# load DeReChar frequency list
with open(MODELPATH, 'r') as fp:
    dat = fp.readlines()

dat = [s.lstrip() for s in dat]
dat = [s for s in dat if len(s) >= 2]
dat = [s for s in dat if 48 <= ord(s[0]) <= 57]
dat = [s.split(" ") for s in dat]
dat = [row for row in dat if len(row) == 2]
dat = [(int(num.replace(".", "")), bi.split("\n")[0]) for num, bi in dat]

derechar = {s: num for num, s in dat if len(s) == 1 and num > 0}
denom = max([num for _, num in derechar.items()])
derechar = {s: num / denom for s, num in derechar.items()}

derebigr = {s: num for num, s in dat if len(s) == 2 and num > 0}
denom = max([num for _, num in derebigr.items()])
derebigr = {s: num / denom for s, num in derebigr.items()}

del dat, denom
gc.collect()


brackets1 = np.percentile(
    [num for _, num in derechar.items()], 
    q=[100 / 6, 100 / 3, 50, 200 / 3, 500 / 6, 100])

brackets2 = np.percentile(
    [num for _, num in derebigr.items()], 
    q=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])



def derechar_to_float(sentences: List[str]):
    feats = derechar_to_int16(sentences)
    n_feats = feats.shape[-1] - 1
    return feats[:, 1:] / np.tile(feats[:, 0].reshape(-1, 1), n_feats)


def derechar_to_int16(sentences: List[str]):
    feats = []
    for sent in sentences:
        # lookup frequency
        freqs = [derechar.get(c, 0.0) for c in sent]
        # assign to decentile
        quantiles = [np.argmax(f <= brackets1) for f in freqs]
        # count decentiles
        cnt = np.zeros((len(brackets1),), dtype=np.int16)
        for q in quantiles:
            cnt[q] += 1
        # save
        feats.append((
            len(sent),
            *cnt.tolist()
        ))
    # done
    return np.vstack(feats).astype(np.int16)


def derebigram_to_float(sentences: List[str]):
    feats = derebigram_to_int16(sentences)
    n_feats = feats.shape[-1] - 1
    return feats[:, 1:] / np.tile(feats[:, 0].reshape(-1, 1), n_feats)


def derebigram_to_int16(sentences: List[str]):
    feats = []
    for sent in sentences:
        if len(sent) >= 3:
            # lookup frequency
            freqs = [derebigr.get(sent[i:(i + 2)], 0.0) 
                    for i in range(1, len(sent) - 1)]
            # assign to decentile
            quantiles = [np.argmax(f <= brackets2) for f in freqs]
            # count decentiles
            cnt = np.zeros((len(brackets2),), dtype=np.int16)
            for q in quantiles:
                cnt[q] += 1
            # save
            feats.append((
                len(freqs),  # len(sent) - 1
                *cnt.tolist()
            ))
        else:
            feats.append([1] + [0] * len(brackets1))
    # done
    return np.vstack(feats).astype(np.int16)
