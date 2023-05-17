
CORPUSNAME="demodata123"

rm -r $CORPUSNAME

# Step 1 - Extract from conll file
read_conll_file.py \
    --conll-file=broken_file.conll \
    --output-dir=$CORPUSNAME \
    --io-batch-size=2

masked_to_feats1_sbert_hrp.py \
    --input-file=$CORPUSNAME/masked.jsonl \
    --output-file=$CORPUSNAME/hashed.jsonl \
    --sbert-path=../models/sbert \
    --hrp-filepath=../models/hrp.pth \
    --batch-size=3

edges_to_feats4_nodedist.py \
    --input-file=$CORPUSNAME/edges.jsonl \
    --output-file=$CORPUSNAME/feats4.jsonl
