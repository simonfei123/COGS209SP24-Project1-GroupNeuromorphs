# let's explore the data we got from the umap sweep
# first load the data from results
import numpy as np
from UMAP_EEG import *

# collect data from umap 461 and pca 461
umap_tar = np.load("./results/umap_461_tar.npy", allow_pickle=True)
umap_align = np.load("./results/umap_461_align.npy", allow_pickle=True)
umap_src = np.load("./results/umap_461_src.npy", allow_pickle=True)

pca_tar = np.load("./results/pca_461_tar.npy", allow_pickle=True)
pca_align = np.load("./results/pca_461_align.npy", allow_pickle=True)
pca_src = np.load("./results/pca_461_src.npy", allow_pickle=True)

# calculate distances
distUMAP = calculateDistanceMetric(umap_src, umap_tar)
distUMAP_Aligned = calculateDistanceMetric(umap_align, umap_tar)
distPCA = calculateDistanceMetric(pca_src, pca_tar)
distPCA_Aligned = calculateDistanceMetric(pca_align, pca_tar)
models = ("UMAP","PCA")
distances = {
	'Pre-Alignment': (distUMAP,distPCA),
	'Post-Alignment': (distUMAP_Aligned,distPCA_Aligned)
}

# plot two lines: one for dist_prealign one for dist_align
import matplotlib.pyplot as plt
x = np.arange(len(models))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in distances.items():
	offset = width * multiplier
	rects = ax.bar(x+offset, measurement, width, label=attribute)
	ax.bar_label(rects,padding=3)
	multiplier += 1

ax.set_ylabel("Distance Between Source and Target (Normalized)")
ax.set_title("Model Comparison: Distances Pre- and Post-Alignment")
ax.set_xticks(x+width, models)
ax.legend(loc='upper left', ncols=2)
ax.set_ylim(0,1)
plt.savefig('n=461_comparison.png')
