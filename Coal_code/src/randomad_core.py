import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from joblib import Parallel, delayed
from minirocket import fit, transform
import warnings
warnings.filterwarnings("ignore")

def normalize_subsequences(subsequences):
    """Normalize each subsequence individually."""
    means = np.mean(subsequences, axis=1, keepdims=True)
    stds = np.std(subsequences, axis=1, keepdims=True)
    stds[stds == 0] = 1
    return (subsequences - means) / stds

def KNN_norm(k_neighbors, train_windows, test_windows):
    train_windows_norm = normalize_subsequences(train_windows)
    test_windows_norm = normalize_subsequences(test_windows)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors)
    nbrs.fit(train_windows_norm)
    distances, _ = nbrs.kneighbors(test_windows_norm)
    return distances.mean(axis=1)

def KNN(k_neighbors, train_windows, test_windows):
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(train_windows)
    distances, _ = nbrs.kneighbors(test_windows)
    return distances.mean(axis=1)

def MiniRocket_fit(train_windows, num_features=100):
    train_windows = train_windows.astype(np.float32)
    return fit(train_windows, num_features=num_features)

def mutual_information(X, Y):
    return mutual_info_score(X, Y)

def calculate_entropy(X):
    return entropy(X)

def calculate_kss_parallel(
    train_windows,
    keep_features_ratio=0.5,
    num_samples=100,
    alpha=0.5,
    beta=0.5,
    n_jobs=-1,
):
    num_features = train_windows.shape[1]
    num_pairs = int(num_features * (num_features - 1) / 2)

    sampled_pairs = np.random.choice(num_pairs, size=min(num_samples, num_pairs), replace=False)

    entropies = Parallel(n_jobs=n_jobs)(
        delayed(calculate_entropy)(train_windows[:, i]) for i in range(num_features)
    )
    entropies = np.array(entropies)

    mi_sum = np.zeros(num_features)
    mi_count = np.zeros(num_features, dtype=int)

    def pair_index(i, j):
        return i * num_features + j - (i + 1) * (i + 2) // 2

    def compute_mi(i, j):
        mi = mutual_information(train_windows[:, i], train_windows[:, j])
        return (i, j, mi)

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_mi)(i, j)
        for i in range(num_features)
        for j in range(i + 1, num_features)
        if pair_index(i, j) in sampled_pairs
    )

    for i, j, mi in results:
        mi_sum[i] += mi
        mi_sum[j] += mi
        mi_count[i] += 1
        mi_count[j] += 1

    avg_mi = np.array([mi_sum[i] for i in range(num_features)])

    kss = alpha * entropies - beta * avg_mi
    num_keep = int(keep_features_ratio * num_features)
    keep_indices = np.argsort(kss)[:num_keep]
    return keep_indices

def RandomAD_mask(
    para,
    train_windows,
    keep_features_ratio=0.5,
    num_samples=100,
    alpha=0.5,
    beta=0.5,
    n_jobs=-1,
):
    train_windows = train_windows.astype(np.float32)
    train_windows = transform(train_windows, para)
    filter_type = 'kss'
    if keep_features_ratio == 1.0:
        return np.ones(train_windows.shape[1])

    if filter_type == 'kss':
        keep_indices = calculate_kss_parallel(
            train_windows,
            keep_features_ratio,
            num_samples,
            alpha,
            beta,
            n_jobs=n_jobs,
        )
        mask = np.zeros(train_windows.shape[1])

    elif filter_type == 'std':
        std_devs = np.std(train_windows, axis=0)
        keep_features = int(keep_features_ratio * train_windows.shape[1])
        keep_indices = np.argpartition(std_devs, -keep_features)[-keep_features:]
        mask = np.zeros_like(std_devs)

    mask[keep_indices] = 1.0
    return mask

def remove_masked_columns(data, mask):
    keep_indices = np.where(mask != 0)[0]
    return data[:, keep_indices]

def RandomAD_run(para, mask, k_neighbors, train_windows, test_windows=None, use_pca=False, pca_variance_ratio=0.8):
    train_windows = train_windows.astype(np.float32)
    if test_windows is not None:
        test_windows = test_windows.astype(np.float32)
    train_windows = transform(train_windows, para)

    if test_windows is not None:
        test_windows = transform(test_windows, para)

    train_windows = remove_masked_columns(train_windows, mask)
    if test_windows is not None:
        test_windows = remove_masked_columns(test_windows, mask)

    pca = None
    if use_pca:
        pca = PCA(n_components=pca_variance_ratio)
        train_windows = pca.fit_transform(train_windows)
        if test_windows is not None:
            test_windows = pca.transform(test_windows)

    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(train_windows)
    if test_windows is None:
        distances, _ = nbrs.kneighbors(train_windows)
    else:
        distances, _ = nbrs.kneighbors(test_windows)
    return distances.mean(axis=1)

def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return (a / 2 + 0.5)

def normalize2(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a
