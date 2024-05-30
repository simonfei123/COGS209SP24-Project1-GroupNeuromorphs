import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_distances
from scipy.linalg import orthogonal_procrustes

import umap

def loadData(subTar='01',subSrc='02',dataPath='./data/thingseeg2_preproc'):
    print("Loading Data")
    train_sub1 = np.load(f"{dataPath}/sub-{subTar}/preprocessed_eeg_training.npy",allow_pickle=True).item()
    train_sub1_data = train_sub1['preprocessed_eeg_data']

    train_sub1_flatten = train_sub1_data.reshape(train_sub1_data.shape[0]*train_sub1_data.shape[1],
                                                train_sub1_data.shape[2]*train_sub1_data.shape[3])

    test_sub2 = np.load(f"{dataPath}/sub-{subSrc}/preprocessed_eeg_training.npy",allow_pickle=True).item()
    test_sub2_data = test_sub2['preprocessed_eeg_data']

    test_sub2_flatten = test_sub2_data.reshape(test_sub2_data.shape[0]*test_sub2_data.shape[1],
                                                test_sub2_data.shape[2]*test_sub2_data.shape[3])

    combined = np.concatenate([train_sub1_flatten,test_sub2_flatten], axis=0)

    # maSrc
    target = np.full((train_sub1_flatten.shape[0],),False)
    source = np.full((test_sub2_flatten.shape[0],),True)
    maSrc = np.concatenate((target,source))

    return combined, maSrc

def mean_center(X):
    # mean center data
    mu = np.mean(X, axis=0)
    return X - mu

def ftProc_umap_tr(X, Y, umapCoef=0):
    # TODO: sweep n_neighbors
    n_samples, n_features = X.shape

    # Try UMAP, a nonlinear dimensionality reduction technique
    reducer = umap.UMAP(n_components=umapCoef)

    # Mean center
    X_centered = mean_center(X)
  
    # run an initial UMAP reduction to estimate params
    X_new = reducer.fit_transform(X_centered)

    model = {
        'reducer': reducer,
        'param': umapCoef
    }

    return X_new, model

def trainRegression(X, Y):
    # train regression between umap of X and Y
    # used for aligning X and Y
    est = LinearRegression()
    est.fit(X, Y)
    return est

def calculateDistanceMetric(X, Y, metric='euclidean'):
    # calculate distance between X and Y
    # metric for quality of alignment

    distances = pairwise_distances(X, Y, metric=metric)
    maximum = np.max(distances)
    minimum = np.min(distances)
    distances_normalized = (distances - minimum) / (maximum-minimum) # normalize

    # take rms to summarize
    #np.mean(distances_normalized)
    rms = np.sqrt(np.mean(distances_normalized**2))

    return rms

def saveResults(X, name):
    # save results to a file
    np.save(name, X)
    return

def calculateCosineSimilarity(X, Y):
    # calculate cosine similarity between X and Y
    # metric for quality of alignment
    pass

def getScale(X):
    # get scale of a matrix
    norm = np.linalg.norm(X)
    scale = np.sqrt(np.sum((X - np.mean(X, axis=0))**2, axis=1))
    print(f"Got scale {scale} and norm {norm}.")
    return scale

def normalizeByScale(X, scale):
    # get scale of a matrix
    return X / scale[:, None]

def ftTrans_sa_umap(ft, maSrc, umapCoef=0):

    # Splitting source and target data
    ftSrc = ft[maSrc, :]
    ftTar = ft[~maSrc, :]

    # UMAP on source and target domains
    ftTarNew, umapModelT = ftProc_umap_tr(ftTar, None, umapCoef)
    ftSrcNew, umapModelS = ftProc_umap_tr(ftSrc, None, umapCoef)
    print(f"UMAP Transformed Target of Shape: {ftTarNew.shape}")
    print(f"UMAP Transformed Source of Shape: {ftSrcNew.shape}")

    # Get subspace slice
    d = min(ftTarNew.shape[1], ftSrcNew.shape[1])
    ftSrcNew = ftSrcNew[:, :d]
    ftTarNew = ftTarNew[:, :d]

    # Align using Procrustes analysis
    alignmentMat, scale = orthogonal_procrustes(ftSrcNew, ftTarNew)

    # Apply alignment to Target
    ftAllNew = np.zeros((ft.shape[0],d))
    ftAllNew[maSrc,:] = ftSrcNew @ alignmentMat
    ftAllNew[~maSrc,:] = ftTarNew

    transMdl = {
        'umapModelT': umapModelT['reducer'],
        'umapModelS': umapModelS['reducer'],
        'alignmentMat': alignmentMat,
        'scale': scale
    }

    return ftAllNew, transMdl, ftSrcNew, ftTarNew

def ftProc_pca_tr(X, Y, pcaCoef=0):
    # Default PCA coefficient
    n_samples, n_features = X.shape

    # Mean centering
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # Performing PCA
    pca = PCA(n_components=min(n_samples, n_features))

    pca.fit(X_centered)
    evs = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(evr)

    # Select the number of components to use
    if pcaCoef == 0:
        postPcaDim = min(n_samples, n_features)
    elif 0 < pcaCoef < 1:
        postPcaDim = np.argmax(cumulative_variance_ratio >= pcaCoef) + 1
    elif pcaCoef >= 1 and pcaCoef == int(pcaCoef):
        postPcaDim = int(pcaCoef)
    
    pca = PCA(n_components=postPcaDim)
    X_new = pca.fit_transform(X_centered)

    model = {
        'mu': mu,
        'W_prj': pca.components_.T,
        'eigVals': evs,
        'postPcaDim': postPcaDim,
        'param': pcaCoef
    }

    return X_new, model

def ftTrans_sa_pca(ft, maSrc, pcaCoef=0):

    # Splitting source and target data
    ftSrc = ft[maSrc, :]
    ftTar = ft[~maSrc, :]

    # PCA on source and target domains
    ftTarNew, pcaModelT = ftProc_pca_tr(ftTar, None, pcaCoef)
    ftSrcNew, pcaModelS = ftProc_pca_tr(ftSrc, None, pcaCoef)
    

    # Alignment
    d = min(pcaModelS['W_prj'].shape[1], pcaModelT['W_prj'].shape[1])
    W_prjS = pcaModelS['W_prj'][:, :d]
    W_prjT = pcaModelT['W_prj'][:, :d]
    ftSrcNew = ftSrcNew[:, :d]
    ftTarNew = ftTarNew[:, :d] # had to edit this to make shapes agree
    # matmul error if different sizes

    ftAllNew = np.zeros((ft.shape[0], d))
    ftAllNew[maSrc, :] = ftSrcNew @ W_prjS.T @ W_prjT
    ftAllNew[~maSrc, :] = ftTarNew 

    transMdl = {
        'WS': W_prjS @ W_prjS.T @ W_prjT,
        'WT': W_prjT,
        'muS': pcaModelS['mu'],
        'muT': pcaModelT['mu']
    }

    return ftAllNew, transMdl, ftSrcNew, ftTarNew