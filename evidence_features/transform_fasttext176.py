import os
import fasttext
import numpy as np
from typing import List


# path to the pretrained sbert model
MODELPATH = os.getenv("MODELFOLDER", "./models")
MODELPATH = os.path.join(MODELPATH, "lid.176.ftz")

# load model once
# model_ft176 = fasttext.load_model(MODELPATH)
# see https://github.com/facebookresearch/fastText/issues/1056
model_ft176 = fasttext.FastText._FastText(model_path=MODELPATH)

#
# German Dialects - werden gut erkannt
# - Hochdeutsch (de): 100 Mio
# - Plattdeutsch (nds): 7 Mio
# - Allemanisch (als): 10 Mio
# - Bairisch (bar): 13 Mio (wird schlecht erkannt)
#
# Franconian in BeNeLux
# Low Franconian lang./dialects
# - Dutch (nl): 24 Mio
# - Zeelandic (zea): 200k
# - West Flemish (vls): 1.4 Mio
# - Limburgish (li): 1.3 Mio
# - Afrikaans (af)
# West Central German lang./dialects
# - Luxembourgish (lb): 600k
#
# North Germanic languages
# - Danish (da)
# - Swedish (sv)
# - Norwegian Bokmal (no)
# - Norwegian Nynorsk (nn)
# - Icelandic (is)
#
# Anglo-Frisian languages
# - English (en)
# - Scots (sco)
# - North Frisian (frr)
# - West Frisian (fy)
#
# Romanic languages within DACH or direct neighbor countries
# - Walloon (wa)
# - French (fr)
# - Romansh (rm)
# - Italian (it)
#
# Slavic languages within DACH or direct neighbor countries
# - Upper Sorbian (hsb)
# - Polish (pl)
# - Czech (cs)
# - Slovak (sk)
# - Slovenian (sl)
LANGS = [
    ["de"], ["nds"], ["als"], ["bar"],
    ["nl", "zea", "vls", "li", "af"] + ["lb"],
    ["da", "sv", "no", "nn", "is"],
    ["en", "sco", "frr", "fy"],
    ["wa", "fr", "rm", "it"],
    ["hsb", "pl", "cs", "sk", "sl"]
]


def lang_to_id(lang: str) -> int:
    # lang_to_id("wa"), lang_to_id("xa")
    for i, GRP in enumerate(LANGS):
        if lang in GRP:
            return i
    return len(LANGS)


def lookup_lang(labels: List[str], probas: List[float]):
    # extract the tags
    labels = [s.split("__")[-1] for s in labels]
    # init output
    out = [0.0 for _ in range(len(LANGS) + 1)]
    # loop over results
    for i, lang in enumerate(labels):
        grpid = lang_to_id(lang)
        out[grpid] = max(out[grpid], probas[i])
    # done
    return out


# Utility functions
def scaledfloat_to_int8(x: float) -> np.int8:
    """ Convert a scaled [0.0, 1.0] float to an 8-bit integer.
    Example:
        idx = scaledfloat_to_int8(1.0)
        idx, int8_to_scaledfloat(idx)
    """
    x = min(1.0, max(0.0, x))
    brackets = np.flip(np.arange(0, 256) / 256)
    idx = np.argmax(brackets <= x) - 128
    return np.int8(idx)


def int8_to_scaledfloat(idx: np.int8) -> float:
    """ Convert an 8-bit integer to scaled [0.0, 1.0] float.
    Example:
        idx = scaledfloat_to_int8(1.0)
        idx, int8_to_scaledfloat(idx)
    """
    idx = min(127, max(-128, idx))
    x = 1. - (float(idx) + 128.0) / 255.0
    return x


def fasttext176_to_int8(sentences: List[str]):
    # run FastText
    sents = [s.replace("\n", " ") for s in sentences]
    labels, probas = model_ft176.predict(sents, k=3)
    # assign labels/langs to language groups
    pdf = [lookup_lang(lb, pb.tolist()) for lb, pb in zip(*(labels, probas))]
    # encode to int8
    encoded = [[scaledfloat_to_int8(p) for p in tmp] for tmp in pdf]
    # stack and cast
    return np.vstack(encoded).astype(np.int8)


def fasttext176_i2f(encoded):
    pdf = [[int8_to_scaledfloat(i) for i in tmp] for tmp in encoded]
    return np.vstack(pdf).astype(float)


def fasttext176_to_float(sentences: List[str]):
    return fasttext176_i2f(fasttext176_to_int8(sentences))


def fasttext176_names():
    return [
        "lang_de",
        "lang_nds",
        "lang_als",
        "lang_bar",
        "lang_franconian",
        "lang_northgermanic",
        "lang_anglofrisian",
        "lang_romanic",
        "lang_slavic",
        "lang_other"
    ]
