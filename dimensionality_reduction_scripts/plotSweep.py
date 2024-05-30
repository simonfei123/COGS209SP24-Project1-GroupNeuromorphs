# let's explore the data we got from the umap sweep
# first load the data from results
import numpy as np
data = np.load("./results/umap_sweep_distances_full.npy", allow_pickle=True)
data_pca = np.load("./results/pca_sweep_distances_full.npy", allow_pickle=True)
coefs = data[:,0]
coefs_pca = data_pca[:,0]
dist_prealign = data[:,1]
dist_prealign_pca = data_pca[:,1]
dist_align = data[:,2]
dist_align_pca = data_pca[:,2]
import matplotlib.pyplot as plt
# plot two lines: one for dist_prealign one for dist_align
plt.plot(coefs, dist_prealign, label='UMAP')
plt.plot(coefs_pca, dist_prealign_pca, label='PCA')
#plt.plot(coefs, dist_align, label='UMAP')
#plt.plot(coefs, dist_align_pca, label='PCA')
plt.legend()
plt.xlabel('N_Components')
plt.ylabel('Normalized Distance')
plt.title('Pre-Alignment Distance Between Source and Target Embeddings')

#import pdb; pdb.set_trace()

"""
print(f"Min prealign {np.min(dist_prealign)}")
print(f"Max prealign {np.max(dist_prealign)}")

print(f"Min align {np.min(dist_align)}")
print(f"Max align {np.max(dist_align)}")
"""

plt.savefig('comparison_distances_full_pre-alignment.png')
#plt.show()
