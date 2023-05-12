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




def extract_from_sentence(sent: conllu.TokenList) -> dict:
    words = get_words(sent)
    reconstructed, lemmata0, spans0 = get_sentence_and_lemmata(sent)
    feats12 = get_feats12(reconstructed, words)
    lemmata, spans = group_lemma_spans(lemmata0, spans0)
    masked = get_masks(reconstructed, spans)

    return (
        {
            "sentence": reconstructed,  # f5, f6, f7, f13, f14, h16
            "lemmata": lemmata,
            "spans": spans,
            "feats12": feats12,
        }, 
        words,  # -> f8, f9
        masked,  # -> f1
    )

def extract_from_document():
    pass
