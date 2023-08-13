import evidence_features as evf

sentences = [
    "Dieser Satz ist ein Beispiel, aber eher kurz.",
    "Die Kuh macht muh, der Hund wufft aber lauter."
]

(
    feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8,
    feats9, feats12, feats13, feats14, hashes15, hashes16,
    sentences_sbd, lemmata17, spans, annotations
) = evf.to_int(sentences, measure_time=True)

(
    feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8,
    feats9, feats12, feats13, feats14, hashes15, hashes16,
    sentences_sbd, lemmata17, spans, annotations
) = evf.to_int(sentences, measure_time=True, sbert_masking=True)
