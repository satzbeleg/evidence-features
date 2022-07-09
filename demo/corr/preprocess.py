import sys
sys.path.append('../..')

import glob
import random
import evidence_features as evf
import numpy as np
import gzip


# read files
y_train, y_test = [], []
s_train, s_test = [], []

for FILE in glob.glob("./data/*.txt"):
    # read texts
    with open(FILE, "r") as fp:
        dat = fp.readlines()
        dat = [s.replace("\n", "") for s in dat]
        random.shuffle(dat)
    # read targets
    s = FILE.split("data/")[-1]
    lang, cty = s[:3], s[4:6]
    # correct lang code
    if lang == 'deu':
        lang = f"de-{cty.upper()}"
    if lang == 'nds' and cty == 'nl':
        lang = 'nds-NL'    
    # 80/20 data split
    n_train = int(len(dat) * 0.8)
    s_train.extend(dat[:n_train])
    y_train.extend([lang] * n_train)
    s_test.extend(dat[n_train:])
    y_test.extend([lang] * (len(dat) - n_train))

# labels
langs = ['de-DE', 'de-AT', 'de-CH', 'de-LI', 'de-BE', 'de-LU', 'de-NA', 'de-EU']
dialects = ['nds', 'nds-NL', 'gsw', 'bar', 'pfl', 'ltz', 'ksh', 'lim']
labels = langs + dialects

# feature names
xnames = evf.get_names()


# Feature extraction
X_train = evf.to_int(s_train)
X_test = evf.to_int(s_test)

with gzip.GzipFile('dataset.npy.gz', 'w') as fp:
    np.save(fp, y_train)
    for dat in X_train:
        np.save(fp, dat)
    np.save(fp, y_test)
    for dat in X_test:
        np.save(fp, dat)
    np.save(fp, xnames)
    np.save(fp, labels)