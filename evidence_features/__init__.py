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

# [x] scripts/masked_to_feats1_sbert_hrp.py
# [x] scripts/edges_to_feats4_nodedist.py
# [x] scripts/sentence_to_feats5.py
# [x] scripts/sentence_to_feats6.py
# [x] scripts/sentence_to_feats7.py
# [x] scripts/sentence_to_feats13.py
# [x] scripts/sentence_to_feats14.py
# [ ] scripts/sentence_to_hashes16.py
# [x] scripts/words_to_feats8_cow.py
# [x] scripts/words_to_feats9_smor.py
# [ ] scripts/adjac_to_hashes15.py
