import kshingle as ks
import datasketch
import struct
import mmh3
import json
import treesimi as ts
from typing import List
import numpy as np


def hashfunc_mmh3_int32(data: bytes) -> np.uint32:
    return struct.unpack('<I', struct.pack('<l', mmh3.hash(data)))[0]


def get_kshingle_hashes(text: str) -> List[np.int32]:
    # k-Shingling: k=6 is approx the average word length in German
    shingled = ks.shingleset_k(text, k=6)
    # build MinHash
    m = datasketch.MinHash(num_perm=32, hashfunc=hashfunc_mmh3_int32)
    for s in shingled:
        m.update(s.encode('utf8'))
    # cast hashvalues from uint64 to uint32
    hv = np.uint32(m.hashvalues)
    # cast to signed int32
    hv = [ts.uint32_to_int32(i) for i in m.hashvalues]
    return hv


def kshingle_to_int32(sentences: List[str]):
    feats16 = []
    for text in sentences:
        try:
            hash16 = get_kshingle_hashes(text)
        except Exception as e:
            hash16 = [0 for _ in range(32)]
            print(e)
        feats16.append(hash16)
    return np.vstack(feats16).astype(np.int32)