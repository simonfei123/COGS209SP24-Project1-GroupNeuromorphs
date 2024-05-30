import numpy as np
import scipy
from scipy.spatial.distance import correlation
import random
import sklearn.linear_model as skl
import os
import pickle

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-from", "--transfer_from",help="Transfer From",default=2)
parser.add_argument("-size", "--size",help="Size",default=3000)
args = parser.parse_args()
sub=int(args.sub)
transfer_from=int(args.transfer_from)
common_size=int(args.size)

eeg_common = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/train_thingseeg2_avg.npy')
original_shape = eeg_common.shape
eeg_common = eeg_common[:common_size]
eeg_common = eeg_common.reshape(eeg_common.shape[0],-1)
eeg_common_transfer_from = np.load(f'data/thingseeg2_preproc/sub-{transfer_from:02d}/train_thingseeg2_avg.npy')[:common_size]
eeg_common_transfer_from = eeg_common_transfer_from.reshape(eeg_common_transfer_from.shape[0],-1)
eeg_new_transfer_from = np.load(f'data/thingseeg2_preproc/sub-{transfer_from:02d}/train_thingseeg2_avg.npy')[common_size:]
eeg_new_transfer_from = eeg_new_transfer_from.reshape(eeg_new_transfer_from.shape[0],-1)

print(eeg_common.shape, eeg_new_transfer_from.shape)

print("Training Transfer Regression")
reg = skl.Ridge(alpha=1000, max_iter=50000, fit_intercept=True)
reg.fit(eeg_common_transfer_from, eeg_common)
print('Transfer training complete')

eeg_new_pred = reg.predict(eeg_new_transfer_from)
eeg_transfered = np.concatenate((eeg_common, eeg_new_pred), axis=0)
eeg_transfered = eeg_transfered.reshape(original_shape)
eeg_nontransfered = np.concatenate((eeg_common, eeg_new_transfer_from), axis=0)
eeg_nontransfered = eeg_nontransfered.reshape(original_shape)

if not os.path.exists(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}'):
    os.makedirs(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}')
np.save(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}/train_thingseeg2_3000avg_transfered_from2.npy', eeg_transfered)
np.save(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}/train_thingseeg2_3000avg_nontransfered_from2.npy', eeg_nontransfered)