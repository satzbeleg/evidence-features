import sys
sys.path.append('../..')

import gzip
import evidence_features as evf
# import sklearn.preprocessing
import sklearn.ensemble
import numpy as np
import joblib
import json
import pandas as pd


# read dataset
with gzip.GzipFile('dataset.npy.gz', 'r') as fp:
    y_train = np.load(fp)
    X_train = [np.load(fp) for _ in range(12)]
    y_test = np.load(fp)
    X_test = [np.load(fp) for _ in range(12)]
    xnames = np.load(fp)
    labels = np.load(fp)

# convert to floating point
X_train = evf.i2f(*X_train)
X_test = evf.i2f(*X_test)

# convert to binary problems
# trf = sklearn.preprocessing.MultiLabelBinarizer(classes=labels)
# y_test = trf.fit_transform([[s] for s in y_test])
# y_train = trf.transform([[s] for s in y_train])

y_train = ['de' if y[:3] == "de-" else y for y in y_train]
y_test = ['de' if y[:3] == "de-" else y for y in y_test]
labels = ['de', 'nds', 'nds-NL', 'gsw', 'bar', 'pfl', 'ltz', 'ksh', 'lim']
y_test = [labels.index(y) for y in y_test]
y_train = [labels.index(y) for y in y_train]


# modeling
model = sklearn.ensemble.RandomForestClassifier(
    n_estimators=128,
    max_depth=32,
    min_samples_leaf=30,
    max_features="sqrt",
    bootstrap=True, oob_score=True, max_samples=0.5,
    random_state=42,
    class_weight="balanced"
)

# training
model.fit(
    X=np.hstack([X_train, np.random.random((X_train.shape[0], 1))]),
    y=y_train
)

# save model
joblib.dump(model, './rf/model.joblib')


# metrics
r2_train = model.score(
    X=np.hstack([X_train, np.random.random((X_train.shape[0], 1))]),
    y=y_train)

r2_test = model.score(
    X=np.hstack([X_test, np.random.random((X_test.shape[0], 1))]),
    y=y_test)

with open('./rf/metrics.json', 'w') as fp:
    json.dump({"acc-train": r2_train, "acc-test": r2_test}, fp)


# feature importance
df_fi = pd.DataFrame(
    index=xnames.tolist() + ["RANDOM"], data=model.feature_importances_, columns=["fi"])
df_fi = df_fi.sort_values(by="fi", ascending=False)
# cutoff = df_fi.loc["RANDOM"].values[0]
df_fi.to_csv("./rf/fi.csv", index=True)
