import logging
import gc
import numpy as np
from typing import List
import itertools
from .transform_all import to_int, i2f
from .transform_sbert import sbert_i2b
from .transform_kshingle import kshingle_to_int32
import uuid
import jsonlines
import json
import quaxa


# start logger
logger = logging.getLogger(__name__)


# convert `annotation` to conllu format
def format_trankit_to_conllu(batch_annot):
    batch_result = []
    for annot in batch_annot:
        result = []
        for t in json.loads(annot):
            tmp_feats = t.get("feats")
            if isinstance(tmp_feats, str):
                tmp_feats = {k: v for k, v in [f.split("=") for f in tmp_feats.split("|")]}
            result.append({
                "id": t.get("id"),
                "form": t.get("text"),
                "lemma": t.get("lemma"),
                "upos": t.get("upos"),
                "xpos": t.get("xpos"),
                "feats": tmp_feats,
                "head": t.get("head"),
                "deprel": t.get("deprel"),
                "deps": t.get("deps"),
                "misc": t.get("misc"),
                "span": t.get("span"),
                "ner": t.get("ner")
            })
        batch_result.append(result)
    return batch_result


def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False


def encode_and_save(FILEPATH: str,
                    sentences: List[str],
                    max_chars: int = 2048,
                    sent_ids: List[str] = None,
                    biblio: List[str] = None,
                    licensetext: List[str] = None,
                    exclude_lowscores: bool = False,
                    document_level=False):

    # encode features
    # if sbert_making=True then `len(f1) = product(l17.shapes)`
    (
        f1, f2, f3, f4, f5, f6, f7, f8,
        f9, f12, f13, f14, h15, h16,
        sentences_sbd, l17, spans, annot
    ) = to_int(sentences, sbert_masking=True, document_level=document_level)
    if document_level:
        sentences = sentences_sbd

    # sent ids
    if sent_ids is None:
        sent_ids = [str(uuid.uuid4()) for _ in range(len(sentences))]
    else:
        # if sent_ids are not UUID strings, then hash strings as UUID
        sent_ids = [
            x if is_valid_uuid(x) else str(uuid.UUID(x))
            for x in sent_ids]

    # encode bibliographic information if exists
    if biblio is not None:
        if isinstance(biblio, (list, tuple)):
            h18 = kshingle_to_int32(biblio)
        elif isinstance(biblio, str):
            h18 = np.array(
                kshingle_to_int32([biblio])[0].tolist() * len(sentences)
            ).astype(np.int32)
            biblio = [biblio] * len(sentences)
    else:
        h18 = np.array([[0] * 32] * len(sentences)).astype(np.int32)
        biblio = [""] * len(sentences)

    # check license
    if isinstance(licensetext, str):
        licensetext = [licensetext] * len(sentences)
    elif licensetext is None:
        licensetext = [""] * len(sentences)

    # convert trankit annotations to conllu
    conll_annot = format_trankit_to_conllu(annot)

    # loop over each sentence
    all_items = []
    j_mask = 0  # index for `f1[j_mask]`
    for i, text in enumerate(sentences):

        # chop sentence length to `max_chars`
        text = text[:max_chars]
        # skip all sentences with less than 3 tokens
        if len(text.split(" ")) < 3:
            logger.warning(f"Sentence to short: '{text}'")
            j_mask += len(l17[i])
            continue
        # skip if no headword was found
        if len(l17[i]) == 0:
            logger.warning(f"Sentence has no VERB, NOUN, ADJ: '{text}'")
            j_mask += len(l17[i])
            continue        

        # save a row for each headword
        for k, headword in enumerate(l17[i]):
            # compute QUAXA scores
            score = quaxa.total_score(
                headword=headword, txt=text, annotation=conll_annot[i])

            # ignore low scores
            if exclude_lowscores and score < 0.5:
                j_mask += 1
                continue

            # add line to list
            all_items.append({
                'headword': headword,
                'example_id': str(uuid.uuid4()),  # save as string
                'sentence': text,
                'sent_id': str(sent_ids[i]),  # save as string
                'spans': [spans[i][k]],
                'annot': annot[i],
                'biblio': biblio[i],
                'license': licensetext[i],
                'score': score,
                'feats1': f1[j_mask].tolist(),  # masked embeddings!
                'feats2': f2[i].tolist(),
                'feats3': f3[i].tolist(),
                'feats4': f4[i].tolist(),
                'feats5': f5[i].tolist(),
                'feats6': f6[i].tolist(),
                'feats7': f7[i].tolist(),
                'feats8': f8[i].tolist(),
                'feats9': f9[i].tolist(),
                'feats12': f12[i].tolist(),
                'feats13': f13[i].tolist(),
                'feats14': f14[i].tolist(),
                'hashes15': h15[i].tolist(),
                'hashes16': h16[i].tolist(),
                'hashes18': h18[i].tolist()
            })
            # masked embeddings!
            j_mask += 1

    # save to disk
    with jsonlines.open(FILEPATH, mode='a') as writer:
        writer.write_all(all_items)

    # done
    pass
