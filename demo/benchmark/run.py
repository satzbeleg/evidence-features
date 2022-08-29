import sys
sys.path.append('../..')

from typing import List
import sentence_embedding_evaluation_german as seeg
import evidence_features as evf
import json
import numpy as np
import tensorflow as tf

# prevent TF from grabbing the whole GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(
        physical_devices[0], True)
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
except Exception as e:
    # Invalid device or cannot modify virtual devices once initialized.
    print(e)
    pass


# (2) Specify the preprocessing
def preprocesser(batch: List[str], params: dict=None) -> List[List[float]]:
    features = evf.to_float(batch)
    return features.astype(np.float32)


# (3) Training settings
params = {
    'datafolder': './datasets',
    'bias': True,
    'balanced': True,
    'batch_size': 128, 
    'num_epochs': 500,
    # 'early_stopping': True,
    # 'split_ratio': 0.2,  # if early_stopping=True
    # 'patience': 5,  # if early_stopping=True
}


# (4) Specify downstream tasks
downstream_tasks = [
    'FCLAIM', 'VMWE', 'OL19-C', 'ABSD-2', 'MIO-P', 'ARCHI', 'LSDC']


# (5) Run experiments
results = seeg.evaluate(
    downstream_tasks, preprocesser, **params)


# store results
with open("results.json", 'w') as fp:
    json.dump(results, fp, indent=2)
