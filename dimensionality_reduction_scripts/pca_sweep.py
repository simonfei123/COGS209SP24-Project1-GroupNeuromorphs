import numpy as np
from UMAP_EEG import *
from tqdm import tqdm
import pdb

print("Loading Data")
train_sub1 = np.load("./data/thingseeg2_preproc/sub-01/preprocessed_eeg_training.npy",allow_pickle=True).item()
train_sub1_data = train_sub1['preprocessed_eeg_data']

train_sub1_flatten = train_sub1_data.reshape(train_sub1_data.shape[0]*train_sub1_data.shape[1],
                                             train_sub1_data.shape[2]*train_sub1_data.shape[3])

test_sub2 = np.load("./data/thingseeg2_preproc/sub-02/preprocessed_eeg_training.npy",allow_pickle=True).item()
test_sub2_data = test_sub2['preprocessed_eeg_data']

test_sub2_flatten = test_sub2_data.reshape(test_sub2_data.shape[0]*test_sub2_data.shape[1],
                                             test_sub2_data.shape[2]*test_sub2_data.shape[3])

combined = np.concatenate([train_sub1_flatten,test_sub2_flatten], axis=0)

# maSrc
target = np.full((train_sub1_flatten.shape[0],),False)
source = np.full((test_sub2_flatten.shape[0],),True)
maSrc = np.concatenate((target,source))

# first sweep pca
print("Running PCA Sweep")
pcaCoefs = [1, 10, 100, 1000, 1700]
pca_distances = []
failed = False
for coef in tqdm(pcaCoefs):
    ftAllNew, transMdl, ftSrcNew, ftTarNew = ftTrans_sa_pca(combined,maSrc,pcaCoef=coef)
    ftSrcAligned = ftAllNew[maSrc,:]
    try:
        if failed:
            raise RuntimeError("Failed previously, so skipping.")
        else:
            distPCA = calculateDistanceMetric(ftSrcNew, ftTarNew)
            distPCA_Aligned = calculateDistanceMetric(ftSrcAligned, ftTarNew)
            pca_distances.append((coef,distPCA,distPCA_Aligned))
    except: # will be sleeping to try to preserve if something breaks
        failed = True
        print("Metric failed, so saving state and continuing.")
        saveResults(np.array(ftSrcNew), "./errstates/ftSrcNew-PCA-"+str(coef)+"comp.npy")
        saveResults(np.array(ftSrcAligned), "./errstates/ftSrcAligned-PCA-"+str(coef)+"comp.npy")
        saveResults(np.array(ftTarNew), "./errstates/ftTarNew-PCA-"+str(coef)+"comp.npy")

# save results to disk as numpy array
if not failed:
    try:
        print("Saving Results.")
        saveResults(np.array(pca_distances), "./results/pca_sweep_distances_full.npy")
    except:
        print("Something went wrong with saving results. Pausing.")
        pdb.set_trace()
