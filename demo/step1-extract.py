"""
run
    pip install .

goto
    cd demo

run
    python step1-extract.py
"""
import evidence_features as evf
import os


CONLLFILE="broken_file.conll"
OUTPUTFOLDER="korpusname123"
IO_BATCH_SIZE=2

os.makedirs(OUTPUTFOLDER, exist_ok=True)


evf.read_conll_file(CONLLFILE, OUTPUTFOLDER, IO_BATCH_SIZE)

