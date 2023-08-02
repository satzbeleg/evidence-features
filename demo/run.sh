
CORPUSNAME="demodata123"

# rm -r $CORPUSNAME

# # Step 1 - Extract from conll file
# read_conll_file.py \
#     --conll-file=broken_file.conll \
#     --output-dir=$CORPUSNAME \
#     --io-batch-size=2

# masked_to_feats1_sbert_hrp.py \
#     --input-file=$CORPUSNAME/masked.jsonl \
#     --output-file=$CORPUSNAME/hashed.jsonl \
#     --sbert-path=../models/sbert \
#     --hrp-filepath=../models/hrp.pth \
#     --batch-size=3

# export B2I8_PCT_CPU=0.95
# hashed_to_feats1.py \
#     --input-file=$CORPUSNAME/hashed.jsonl \
#     --output-file=$CORPUSNAME/feats1.jsonl

# export NDIST_PCT_CPU=0.95
# edges_to_feats4_nodedist.py \
#     --input-file=$CORPUSNAME/edges.jsonl \
#     --output-file=$CORPUSNAME/feats4.jsonl

# export CONS_PCT_CPU=0.95
# sentence_to_feats5_consonants.py \
#     --input-file=$CORPUSNAME/extracted.jsonl \
#     --output-file=$CORPUSNAME/feats5.jsonl

# export DERE_PCT_CPU=0.95
# sentence_to_feats67_derechar.py \
#     --input-file=$CORPUSNAME/extracted.jsonl \
#     --output-file=$CORPUSNAME/feats67.jsonl

# export DECOW_PCT_CPU=0.95
# words_to_feats8_cow.py \
#     --input-file=$CORPUSNAME/words.jsonl \
#     --output-file=$CORPUSNAME/feats8.jsonl \
#     --model-folder=../models

export SMOR_PCT_CPU=0.95
words_to_feats9_smor.py \
    --input-file=$CORPUSNAME/words.jsonl \
    --output-file=$CORPUSNAME/feats9.jsonl \
    --model-folder=../models
