import conllu
from typing import List


# words
# 45.6 µs ± 1.22 µs
def get_words(sent: conllu.TokenList) -> List[str]:
    """ Extract words from conllu.TokenList. Ignore punctuation."""
    words = []
    sid, eid = -1, -1  # ignore SubIDs of MWAs
    for t in sent:
        # ignore TokenIDs of MWAs or other splits
        tid = t.get("id")
        if isinstance(tid, int):
            if eid >= tid >= sid:
                continue
        # reset range of TokenIDs of MWAs to ignore
        if isinstance(tid, (list, tuple)):
            sid, _, eid = tid
        else:
            sid, eid = -1, -1
        # ignore certain UPOS
        if t.get("upos") in ["PUNCT"]:
            continue
        # add word
        form = t.get("form")
        if form:
            words.append(form)
    return words


# sentence, lemmata, spans
# 80.6 µs ± 6.23 µs
def get_sentence_and_lemmata(sent: conllu.TokenList,
                             upos_list=["NOUN", "VERB", "ADJ"]):
    """ Reconstruct sentence from conllu.TokenList, 
        lemmata and their forms' spans.
    """
    sid, eid = -1, -1  # ignore SubIDs of MWAs
    text = []
    pos, spans, lemmata = 0, [], []

    for t in sent:
        # ignore TokenIDs of MWAs or other splits
        tid = t.get("id")
        if isinstance(tid, int):
            if eid >= tid >= sid:
                continue
        # reset range of TokenIDs of MWAs to ignore
        if isinstance(tid, (list, tuple)):
            sid, _, eid = tid
        else:
            sid, eid = -1, -1
        
        # reconstruct text, tracke span
        form = t.get("form")
        text.append(form)
        tmp = pos + len(form); span = (pos, tmp); pos = tmp
        if t.get("misc") is not None:
            if t.get("misc").get("SpaceAfter", "").lower() != "no":
                text.append(" "); pos += 1
        else:
            text.append(" "); pos += 1

        # lemmata
        if t.get("upos") in upos_list:
            lemmata.append(t.get("lemma"))
            spans.append(span)

    # reconstruct sentence
    reconstructed = "".join(text).strip()

    # done
    return reconstructed, lemmata, spans


def get_feats12(sentence: str, words: List[str]):
    return len(sentence), len(words)


# group lemmata
# 10.7 µs ± 2.29 µs
def group_lemma_spans(lemmata, spans):
    grp = {key: [] for key in set(lemmata)}
    for key, val in zip(*(lemmata, spans)):
        grp[key].append(val)
    lemmata2, spans2 = [], []
    for key in sorted(grp):
        lemmata2.append(key)
        spans2.append(sorted(grp[key], reverse=True))
    return lemmata2, spans2


# masking
# 7.36 µs ± 88.2 ns
def get_masks(text: str, spans) -> List[str]:
    masked = []
    for ex in spans:
        mtxt = text
        for spos, epos in ex:
            mtxt = f"{mtxt[0:spos]}[MASK]{mtxt[epos:]}"
        masked.append(mtxt)
    return masked


# feats2
# code for PoS-tag distribution
# https://universaldependencies.org/u/pos/
# -ndbt/nid- = not detected, or not in dataset
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

# feats2 (get pos tag counts)
# 24.2 µs ± 1.05 µs
def get_feats2(sent: conllu.TokenList) -> List[int]:
    cnt = [0] * len(TAGSET)
    for t in sent:
        tag = t.get("upos")
        try:
            idx = TAGSET.index(tag)
        except Exception:
            idx = TAGSET.index("X")
        cnt[idx] += 1
    return [sum(cnt), *cnt]


# feats3
# morphtags
# Ensure that all STTS conversions are included
# https://universaldependencies.org/tagset-conversion/de-stts-uposf.html
# -ndbt/nid- = not detected, or not in dataset
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


# feats3 (get morph tag tag counts)
# 107 µs ± 34.1 µs
def get_feats3(sent: conllu.TokenList) -> List[int]:
    cnt = [0] * len(MORPHTAGS)
    for t in sent:
        mfeats = t.get("feats")
        if isinstance(mfeats, dict):
            for k, v in mfeats.items():
                try:
                    idx = MORPHTAGS.index(f"{k}={v}")
                    cnt[idx] += 1
                except Exception:
                    idx = None
    return [len(sent), *cnt]


# annotations
# 11 µs ± 2.66 µs
def get_annot(sent: conllu.TokenList) -> list:
    return [dict(t) for t in sent]


# hashes15 (input data for treesimi)
# 52.8 µs ± 1.6 µs
def get_adjac(sent: conllu.TokenList) -> list:
    return [
        (t.get("id"), t.get("head"), t.get("deprel"))
        for t in sent if isinstance(t.get("id"), int)]


# feats4 (inputs for nodedist)
# 40.2 µs ± 772 ns
def get_edges(sent: conllu.TokenList):
    # read trankit dependency tree
    return [
        (t.get("head"), t.get("id"))
        for t in sent if isinstance(t.get("id"), int)]



def extract_from_sentence(sent: conllu.TokenList) -> dict:
    words = get_words(sent)
    reconstructed, lemmata0, spans0 = get_sentence_and_lemmata(sent)
    feats12 = get_feats12(reconstructed, words)
    lemmata, spans = group_lemma_spans(lemmata0, spans0)
    masked = get_masks(reconstructed, spans)  # -> f1
    feats2 = get_feats2(sent)
    feats3 = get_feats3(sent)

    annot = get_annot(sent)
    adjac = get_adjac(sent) # -> h15
    edges = get_edges(sent)  # -> f4

    return (
        {
            "sentence": reconstructed,  # f5, f6, f7, f13, f14, h16
            "lemmata": lemmata,
            "spans": spans,
            "annot": annot,
            "feats2": feats2,
            "feats3": feats3,
            "feats12": feats12,
        }, 
        words,  # -> f8, f9
        masked,  # -> f1
        adjac,  # -> h15
        edges,  # -> f4
    )

def extract_from_document():
    pass
