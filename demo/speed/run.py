import sys
sys.path.append('../..')

import evidence_features as evf
import glob

FILES = glob.glob("data/*.txt")

for FILE in FILES:
    with open(FILE, 'r') as fp:
        sentences = fp.readlines()
        sentences = [s.strip() for s in sentences]
        print(len(sentences))
        # compute embeddings
        (
            feats1, feats2, feats3, feats4, feats5, feats6, feats7, feats8,
            feats9, feats12, feats13, feats14, hashes15, hashes16,
            lemmata17, spans, annotations
        ) = evf.to_int(sentences, measure_time=True, sbert_masking=True)
    break
