import os
import torch
import trankit
import node_distance as nd
from typing import List
import numpy as np
from .utils import divide_by_1st_col, divide_by_sum
# syntax hashes for similarity metrics
import treesimi as ts
import datasketch
import struct
import mmh3
import json


# path to the pretrained trankit model
MODELPATH = os.getenv("MODELFOLDER", "./models")
MODELPATH = os.path.join(MODELPATH, "trankit")

# Load the pretrained model
model_trankit = trankit.Pipeline(
    lang='german-hdt',
    gpu=torch.cuda.is_available(),
    cache_dir=MODELPATH
)


# code for PoS-tag distribution
# https://universaldependencies.org/u/pos/
# -ndbt/nid- = not detected by trankit, or not in dataset
TAGSET = [
    'ADJ',
    'ADP',
    'ADV',
    'AUX',
    'CCONJ',
    'DET',
    'INTJ',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'PUNCT',
    'SCONJ',
    # 'SYM',  # -ndbt/nid-
    'VERB',
    'X'
]


def get_postag_counts(snt):
    postags = [t.get("upos") for t in snt['tokens']]
    cnt = np.zeros((len(TAGSET),), dtype=np.int8)
    for tag in postags:
        try:
            idx = TAGSET.index(tag)
        except Exception:
            idx = TAGSET.index("X")
        cnt[idx] += 1
    return len(postags), cnt


# morphtags
# Ensure that all STTS conversions are included
# https://universaldependencies.org/tagset-conversion/de-stts-uposf.html
# -ndbt/nid- = not detected by trankit, or not in dataset
MORPHTAGS = [
    # punctuation type (3/11)
    "PunctType=Brck",  # `$(`
    "PunctType=Comm",  # `$,`
    "PunctType=Peri",  # `$.`
    # adposition type (3/4)
    "AdpType=Post",  # APPO
    "AdpType=Prep",  # APPR, APPRART
    "AdpType=Circ",  # APZR
    # particle type (3/6)
    "PartType=Res",  # PTKANT
    "PartType=Vbp",  # PTKVZ
    "PartType=Inf",  # PTKZU
    # pronominal type (8/11)
    "PronType=Art",  # APPRART, ART
    "PronType=Dem",  # PAV, PDAT, PDS
    "PronType=Ind",  # PIAT, PIDAT, PIS
    # "PronType=Neg",  # PIAT, PIDAT, PIS -ndbt/nid-
    # "PronType=Tot",  # PIAT, PIDAT, PIS -ndbt/nid-
    "PronType=Prs",  # PPER, PPOSAT, PPOSS, PRF
    "PronType=Rel",  # PRELAT, PRELS
    "PronType=Int",  # PWAT, PWAV, PWS
    # other related to STTS post tags
    # "AdjType=Pdt",  # PIDAT -ndbt/nid-
    "ConjType=Comp",  # KOKOM
    "Foreign=Yes",  # FM
    "Hyph=Yes",  # TRUNC
    "NumType=Card",  # CARD
    "Polarity=Neg",  # PTKNEG
    "Poss=Yes",  # PPOSAT, PPOSS
    "Reflex=Yes",  # PRF
    "Variant=Short",  # ADJD
    # verbs
    "VerbForm=Fin",  # VAFIN, VAIMP, VMFIN, VVFIN, VVIMP
    "VerbForm=Inf",  # VAINF, VVINF, VVIZU
    "VerbForm=Part",  # VAPP, VMPP, VVPP
    "Mood=Ind",  # VAFIN, VMFIN, VVFIN
    "Mood=Imp",  # VAIMP, VVIMP
    # "Mood=Sub",  # -ndbt/nid-
    "Aspect=Perf",  # VAPP, VMPP, VVPP
    "VerbType=Mod",  # VMPP
    # other syntax
    "Gender=Fem",
    "Gender=Masc",
    "Gender=Neut",
    "Number=Sing",
    "Number=Plur",
    "Person=1",
    "Person=2",
    "Person=3",
    "Case=Nom",
    "Case=Dat",
    "Case=Gen",
    "Case=Acc",
    # "Definite=Ind",  # -ndbt/nid-
    # "Definite=Def",  # -ndbt/nid-
    "Degree=Pos",
    "Degree=Cmp",
    "Degree=Sup",
    "Tense=Pres",
    "Tense=Past",
    # "Tense=Fut",  # -ndbt/nid-
    # "Tense=Imp",  # -ndbt/nid-
    # "Tense=Pqp",  # -ndbt/nid-
    # "Polite=",  # -ndbt/nid-
]


def get_morphtag_counts(snt):
    cnt = np.zeros((len(MORPHTAGS),), dtype=np.int8)
    for t in snt['tokens']:
        mfeats = t.get("feats")
        if isinstance(mfeats, str):
            for idx, tag in enumerate(MORPHTAGS):
                if tag in mfeats:
                    cnt[idx] += 1
    return len(snt['tokens']), cnt


def get_nodedist(snt):
    # read trankit dependency tree
    edges = [(t.get("head"), t.get("id"))
             for t in snt.get("tokens")
             if isinstance(t.get("id"), int)]
    # number of node
    num_nodes = len(snt.get("tokens")) + 1
    # node and token distance
    nodedist, tokendist, _ = nd.node_token_distances(
        [edges], [num_nodes], cutoff=25)
    # count node vs token distance
    _, _, cnt = nd.tokenvsnode_distribution(
        tokendist, nodedist, xmin=-5, xmax=15)
    # done
    return cnt.astype(np.int8)


# syntax hashes for similarity metrics
def hashfunc_mmh3_int32(data: bytes) -> np.uint32:
    """ Hash function for DataSketch
    Example:
    --------
    txt = "Hello World!"
    b = hashfunc_mmh3_int32(txt.encode('utf-8'))
    """
    return struct.unpack('<I', struct.pack('<l', mmh3.hash(data)))[0]


def get_treesimi_hashes(snt) -> List[np.int32]:
    # parse trankit sentence
    adjac = [(t.get("id"), t.get("head"), t.get("deprel"))
              for t in snt.get("tokens")
              if isinstance(t.get("id"), int)]
    # adjust trankit IDs
    d = min([c for c, _, __ in adjac]) - 1
    adjac = [(c - d, max(0, p - d), m) for c, p, m in adjac]
    # Convert to Nested Set model
    nested = ts.adjac_to_nested_with_attr(adjac)
    nested = ts.remove_node_ids(nested)
    # shingle subtrees
    shingled = ts.shingleset(nested, use_trunc_leaves=True,
                             use_drop_nodes=False, use_replace_attr=False)
    stringified = [json.dumps(subtree).encode('utf-8')
                   for subtree in shingled]
    # encode with MinHash an MurmurHash3 (mmh3)
    m = datasketch.MinHash(num_perm=32, hashfunc=hashfunc_mmh3_int32)
    for s in stringified:
        m.update(s)
    # cast hashvalues from uint64 to uint32
    hv = np.uint32(m.hashvalues)
    # cast to signed int32
    hv = [ts.uint32_to_int32(i) for i in m.hashvalues]
    return hv


def get_lemmata(snt, upos_list=["NOUN", "VERB", "ADJ"]):
    return [t.get("lemma") for t in snt.get("tokens")
            if t.get("upos") in upos_list]  


def trankit_to_float(sentences: List[str]):
    feats1, feats2, feats3 = trankit_to_int(sentences, skiphash=True)
    out1 = divide_by_1st_col(feats1)
    out2 = divide_by_1st_col(feats2)
    out3 = divide_by_sum(feats3)
    return out1, out2, out3


def trankit_to_int(sentences: List[str], skiphash=True):
    feats1 = []
    feats2 = []
    feats3 = []
    hashes15 = []
    lemmata17 = []
    for sent in sentences:
        try:
            snt = model_trankit(sent)
            snt = snt.get('sentences')[0]
            num1, cnt1 = get_postag_counts(snt)
            num2, cnt2 = get_morphtag_counts(snt)
            cnt3 = get_nodedist(snt)
            if not skiphash:
                hsh15 = get_treesimi_hashes(snt)
                lem17 = get_lemmata(snt)
        except Exception as e:  # RuntimeError, AssertionError
            num1, cnt1 = 0, np.zeros((len(TAGSET),), dtype=np.int8)
            num2, cnt2 = 0, np.zeros((len(MORPHTAGS),), dtype=np.int8)
            cnt3 = np.array([0 for _ in range(21)])
            if not skiphash:
                hsh15 = [0 for _ in range(32)]
                lem17 = []
            print(e)
        feats1.append((num1, *cnt1.tolist()))
        feats2.append((num2, *cnt2.tolist()))
        feats3.append(cnt3.tolist())
        if not skiphash:
            hashes15.append(hsh15)
            lemmata17.append(lem17)
    # 1
    feats1 = np.maximum(np.iinfo(np.int8).min, feats1)
    feats1 = np.minimum(np.iinfo(np.int8).max, feats1)
    feats1 = np.vstack(feats1).astype(np.int8)
    # 2
    feats2 = np.maximum(np.iinfo(np.int8).min, feats2)
    feats2 = np.minimum(np.iinfo(np.int8).max, feats2)
    feats2 = np.vstack(feats2).astype(np.int8)
    # 3
    feats3 = np.maximum(np.iinfo(np.int8).min, feats3)
    feats3 = np.minimum(np.iinfo(np.int8).max, feats3)
    feats3 = np.vstack(feats3).astype(np.int8)
    # 15
    if not skiphash:
        hashes15 = np.vstack(hashes15).astype(np.int32)
    # done
    if skiphash:
        return feats1, feats2, feats3
    else:
        return feats1, feats2, feats3, hashes15, lemmata17

def trankit_names():
    return (
        [f"pos_{tag.lower()}" for tag in TAGSET],
        [f"mfeat_{tag.lower().replace('=', '_')}" for tag in MORPHTAGS],
        [f"nodedist_m{j}" for j in range(5, 0, -1)] + [
            f"nodedist_p{j}" for j in range(0, 15 + 1, 1)]
    )
