__version__ = '0.1.0'

from .extract_from_conll_sent import (
    get_words,  # -> f8, f9, (f12)
    get_sentence_and_lemmata,  # -> sentence, f5, f6, f7, f13, f14, h16, (f12)
    get_feats12,
    group_lemma_spans,  # -> headword, spans, masked -> f1
    get_masks,  # -> f1
    get_feats2,
    get_feats3,
    get_annot,  # annot
    get_adjac,  # -> h15
    get_edges,  # -> f4
    extract_from_sentence  # wrapper
)

from .extract_from_conll_doc import (
    get_ddc_biblio,
    get_ddc_license,
    read_conll_file
)
