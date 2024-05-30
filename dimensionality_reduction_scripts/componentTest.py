import numpy as np
from UMAP_EEG import *
from tqdm import tqdm

# Load the data
combined, maSrc = loadData()

# Run test
# run one component for testing the amount that works well for PCA (95% variance explained)
print("Running UMAP for 461 components")
ftAllNew, transMdl, ftSrcNew, ftTarNew = ftTrans_sa_umap(combined, maSrc, umapCoef=461)
ftSrcAligned = ftAllNew[maSrc,:]

print("Running PCA for 461 components")
ftAllNewP, transMdlP, ftSrcNewP, ftTarNewP = ftTrans_sa_pca(combined, maSrc, pcaCoef=461)
ftSrcAlignedP = ftAllNew[maSrc,:]

print("Running PCA for 571 components")
ftAllNewPMax, transMdlPMax, ftSrcNewPMax, ftTarNewPMax = ftTrans_sa_pca(combined, maSrc, pcaCoef=571)
ftSrcAlignedPMax = ftAllNew[maSrc,:]

print("Running UMAP for 571 components")
ftAllNewMax, transMdlMax, ftSrcNewMax, ftTarNewMax = ftTrans_sa_umap(combined, maSrc, umapCoef=571)
ftSrcAlignedMax = ftAllNew[maSrc,:]

# Save the results
saveResults(np.array(ftSrcNew), "./results/umap_461_src.npy")
saveResults(np.array(ftTarNew), "./results/umap_461_tar.npy")
saveResults(np.array(ftSrcAligned), "./results/umap_461_align.npy")

saveResults(np.array(ftSrcNewP), "./results/pca_461_src.npy")
saveResults(np.array(ftTarNewP), "./results/pca_461_tar.npy")
saveResults(np.array(ftSrcAlignedP), "./results/pca_461_align.npy")

saveResults(np.array(ftSrcNewPMax), "./results/pca_571_src.npy")
saveResults(np.array(ftTarNewPMax), "./results/pca_571_tar.npy")
saveResults(np.array(ftSrcAlignedPMax), "./results/pca_571_align.npy")

saveResults(np.array(ftSrcNewMax), "./results/umap_571_src.npy")
saveResults(np.array(ftTarNewMax), "./results/umap_571_tar.npy")
saveResults(np.array(ftSrcAlignedMax), "./results/umap_571_align.npy")
