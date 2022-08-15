import os
import sfst_transduce
import re
import string
import numpy as np
from typing import List
from .utils import divide_by_1st_col

# path to the pretrained SMOR model
MODELPATH = os.getenv("MODELFOLDER", "./models")
MODELPATH = os.path.join(MODELPATH, "smor.a")

# Load the pretrained model
model_fst = sfst_transduce.Transducer(MODELPATH)


# Morphemes/Lexemes
def get_morphology_stats(word):
    res = model_fst.analyse(word)
    if len(res) == 0:
        return 0, 0, 0
    variants = {}
    for sinp in res:
        s = re.sub(r'<[^>]*>', '\t', sinp)
        lexemes = [t for t in s.split("\t") if len(t) > 0]
        key = "+".join(lexemes)
        variants[key] = lexemes
    # syntactial ambivalence
    num_usecases = len(res)
    # lexeme ambivalence
    num_splittings = len(variants)
    # working memory for composita comprehension
    max_lexemes = max([len(lexemes) for lexemes in variants.values()])
    return num_usecases, num_splittings, max_lexemes


def simple_tokenizer(sinp):
    chars = re.escape(string.punctuation)
    s = re.sub(r'[' + chars + ' ]', '\t', sinp)
    tokens = [re.sub('[^a-zA-Z0-9]', '', t) for t in s.split('\t')]
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


brackets1 = np.array([1, 2, 4, 8, 16, np.inf])
brackets2 = np.array([1, 2, 3, np.inf])
brackets3 = np.array([1, 2, 3, np.inf])


def get_morphology_distributions(sent):
    tokens = simple_tokenizer(sent)
    # get stats
    num_usecases = []
    num_splittings = []
    max_lexemes = []
    for word in tokens:
        tmp = get_morphology_stats(word)
        num_usecases.append(tmp[0])
        num_splittings.append(tmp[1])
        max_lexemes.append(tmp[2])
    # (A) syntactial ambivalence
    classes = [np.argmax(f <= brackets1) for f in num_usecases]
    cnt1 = np.zeros((len(brackets1),), dtype=np.int8)
    for idx in classes:
        cnt1[idx] += 1
    # (B) lexeme ambivalence
    classes = [np.argmax(f <= brackets2) for f in num_splittings]
    cnt2 = np.zeros((len(brackets2),), dtype=np.int8)
    for idx in classes:
        cnt2[idx] += 1
    # (C) working memory for composita comprehension
    classes = [np.argmax(f <= brackets3) for f in max_lexemes]
    cnt3 = np.zeros((len(brackets3),), dtype=np.int8)
    for idx in classes:
        cnt3[idx] += 1
    # done
    return len(tokens), cnt1, cnt2, cnt3


def smor_to_float(sentences: List[str]):
    feats = smor_to_int8(sentences)
    return divide_by_1st_col(feats)


def smor_to_int8(sentences: List[str]):
    feats = []
    for sent in sentences:
        n_token, cnt1, cnt2, cnt3 = get_morphology_distributions(sent)
        feats.append((
            n_token,
            *cnt1.tolist(),
            *cnt2.tolist(),
            *cnt3.tolist()
        ))
    # done
    feats = np.maximum(np.iinfo(np.int8).min, feats)
    feats = np.minimum(np.iinfo(np.int8).max, feats)
    return np.vstack(feats).astype(np.int8)


def smor_names():
    nam1 = [f"smor_syn_ambiv_{j}" for j in [1, 2, 4, 8, 16, "more"]]
    nam2 = [f"smor_lex_ambiv_{j}" for j in [1, 2, 3, "more"]]
    nam3 = [f"smor_work_memo_{j}" for j in [1, 2, 3, "more"]]
    return nam1 + nam2 + nam3
