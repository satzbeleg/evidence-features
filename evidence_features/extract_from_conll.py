import conllu
from typing import List


# words
# 44 µs ± 993 ns
def get_words(sent: conllu.TokenList) -> List[str]:
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

