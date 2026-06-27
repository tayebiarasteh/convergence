"""
alignment/metrics.py
Created on May 27, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import warnings
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


def load_embeddings(npz_path: str):
    """Load (N, D) embeddings and case_ids from a cached .npz file."""
    d = np.load(npz_path, allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["case_ids"].astype(str)


def align_by_case_ids(
    emb_a: np.ndarray, ids_a: np.ndarray,
    emb_b: np.ndarray, ids_b: np.ndarray,
):
    """Return aligned (emb_a[shared], emb_b[shared]) in matching row order."""
    set_a = {str(c): i for i, c in enumerate(ids_a)}
    set_b = {str(c): i for i, c in enumerate(ids_b)}
    shared = sorted(set_a.keys() & set_b.keys())
    if not shared:
        raise ValueError("No shared case_ids between the two embedding files.")
    idx_a = [set_a[c] for c in shared]
    idx_b = [set_b[c] for c in shared]
    return emb_a[idx_a], emb_b[idx_b], shared



def _knn_graph(X: np.ndarray, k: int) -> np.ndarray:
    """Return (N, k) int array of k-nearest neighbor indices (excluding self)
    using cosine distance (= 1 - dot product for L2-normalized vectors).

    Tries faiss first; on ANY faiss failure (not just missing package, e.g. the
    NumPy 2.x ABI break that raises AttributeError) falls back to sklearn's
    NearestNeighbors, and only as a last resort to a chunked numpy scan. The full
    N x N similarity matrix is avoided because at tens of thousands of cases it is
    many GB and OOMs."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    N = X.shape[0]

    try:
        import faiss
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        _, I = index.search(X, k + 1)
        return I[:, 1:]     # exclude self (first column)
    except Exception as e:
        print(f"[_knn_graph] faiss unavailable ({type(e).__name__}: {e}); "
              f"using sklearn fallback.")

    try:
        from sklearn.neighbors import NearestNeighbors
        # Cosine distance; request k+1 to drop self.
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
        nn.fit(X)
        _, idx = nn.kneighbors(X)
        # Drop self (usually column 0, but guard in case it isn't first).
        out = np.empty((N, k), dtype=np.int64)
        for i in range(N):
            row = idx[i]
            row = row[row != i][:k]
            if len(row) < k:   # pad if self wasn't present
                row = np.concatenate([row, idx[i][:k - len(row)]])
            out[i] = row[:k]
        return out
    except Exception as e:
        print(f"[_knn_graph] sklearn fallback failed ({type(e).__name__}: {e}); "
              f"using chunked numpy.")

    out = np.empty((N, k), dtype=np.int64)
    chunk = max(1, min(2048, N))
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        sims = X[start:stop] @ X.T              # (chunk, N)
        for r, gi in enumerate(range(start, stop)):
            sims[r, gi] = -np.inf               # exclude self
        # top-k per row via argpartition (cheaper than full argsort)
        part = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        # order the k by actual similarity
        for r in range(stop - start):
            order = np.argsort(-sims[r, part[r]])
            out[start + r] = part[r][order]
    return out


def mknn(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Mutual k-nearest neighbor overlap.

    For each point i, count how many of its k-NNs in X are also k-NNs in Y,
    averaged over all points and normalized by k. Range [0, 1].
    """
    nn_x = _knn_graph(X, k)   # (N, k)
    nn_y = _knn_graph(Y, k)   # (N, k)
    n = X.shape[0]
    total = 0.0
    for i in range(n):
        total += len(set(nn_x[i]) & set(nn_y[i]))
    return total / (n * k)


def _cknna_from_graphs(nn_x: np.ndarray, nn_y: np.ndarray, N: int) -> float:
    from scipy import sparse
    k = nn_x.shape[1]
    rows = np.repeat(np.arange(N), k)
    K = sparse.csr_matrix((np.ones(N * k, np.float32), (rows, nn_x.ravel())),
                          shape=(N, N))
    L = sparse.csr_matrix((np.ones(N * k, np.float32), (rows, nn_y.ravel())),
                          shape=(N, N))
    one = np.ones(N, dtype=np.float64)

    def _centered_trace(A, B):
        # tr(A_c B_c) with A_c=H A H, B_c=H B H, via the closed form above.
        AB_diag = A.multiply(B.T).sum()          # tr(A B) = sum_ij A_ij B_ji
        Bone = B.dot(one); Aone = A.dot(one)
        ATone = A.T.dot(one); BTone = B.T.dot(one)
        aKLb = float(one @ (A.dot(Bone)))         # 1^T A B 1
        aLKb = float(one @ (B.dot(Aone)))         # 1^T B A 1
        sA = float(A.sum()); sB = float(B.sum())
        return float(AB_diag) - aKLb / N - aLKb / N + sA * sB / (N * N)

    num   = _centered_trace(K, L)
    denom = np.sqrt(_centered_trace(K, K) * _centered_trace(L, L))
    return num / denom if denom > 0 else 0.0


def cknna(X: np.ndarray, Y: np.ndarray, k: int = 10,
          max_n: int = 20000, seed: int = 0) -> float:
    N = X.shape[0]
    if N > max_n:
        rng = np.random.RandomState(seed)
        sel = rng.choice(N, size=max_n, replace=False)
        X, Y = X[sel], Y[sel]
        N = max_n

    nn_x = _knn_graph(X, k)
    nn_y = _knn_graph(Y, k)
    return _cknna_from_graphs(nn_x, nn_y, N)



def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    N = K.shape[0]
    H = np.eye(N) - 1.0 / N
    return float(np.trace(K @ H @ L @ H))


def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cross = Yc.T @ Xc                      # (D_y, D_x)
    num   = float(np.sum(cross * cross))
    xx    = Xc.T @ Xc                      # (D_x, D_x)
    yy    = Yc.T @ Yc                      # (D_y, D_y)
    denom = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    return num / denom if denom > 0 else 0.0


def cka_rbf(
    X: np.ndarray, Y: np.ndarray,
    sigma: Optional[float] = None,
    n_subsample: int = 2000,
) -> float:
    """RBF CKA with median-heuristic bandwidth.

    Subsamples to n_subsample rows to keep memory tractable (N^2 kernel matrix).
    """
    N = X.shape[0]
    if N > n_subsample:
        rng = np.random.RandomState(0)
        idx = rng.choice(N, size=n_subsample, replace=False)
        X, Y = X[idx], Y[idx]
    N = X.shape[0]

    if sigma is None:
        # Median heuristic on X
        pairwise_sq = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] \
                      - 2.0 * X @ X.T
        sigma = float(np.sqrt(np.median(pairwise_sq[pairwise_sq > 0])))

    def _rbf_kernel(Z, s):
        d = np.sum(Z**2, axis=1)[:, None] + np.sum(Z**2, axis=1)[None, :] - 2.0 * Z @ Z.T
        return np.exp(-d / (2.0 * s**2))

    K = _rbf_kernel(X, sigma)
    L = _rbf_kernel(Y, sigma)
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    num   = _hsic(K, L)
    denom = np.sqrt(_hsic(K, K) * _hsic(L, L))
    return float(num / denom) if denom > 0 else 0.0



def procrustes(X: np.ndarray, Y: np.ndarray) -> float:
    from scipy.linalg import orthogonal_procrustes
    if X.shape[1] != Y.shape[1]:
        d = max(X.shape[1], Y.shape[1])
        if X.shape[1] < d:
            X = np.pad(X, ((0, 0), (0, d - X.shape[1])), mode="constant")
        if Y.shape[1] < d:
            Y = np.pad(Y, ((0, 0), (0, d - Y.shape[1])), mode="constant")
    R, scale = orthogonal_procrustes(Y, X)
    aligned  = Y @ R
    diff     = X - aligned
    dist     = float(np.linalg.norm(diff, "fro")) / np.sqrt(X.shape[0])
    return dist


def rsa_spearman(X: np.ndarray, Y: np.ndarray, n_subsample: int = 2000) -> float:
    """Representational Similarity Analysis via Spearman rho of upper-triangle
    pairwise distance matrices. Subsamples for tractability."""
    N = X.shape[0]
    if N > n_subsample:
        rng = np.random.RandomState(0)
        idx = rng.choice(N, size=n_subsample, replace=False)
        X, Y = X[idx], Y[idx]
    # Cosine distance = 1 - cos_sim; for L2-normalized vectors this is simply
    # 1 - X @ X.T but we want upper triangle only
    sim_x = X @ X.T    # (N, N) cosine similarities
    sim_y = Y @ Y.T
    tri   = np.triu_indices(sim_x.shape[0], k=1)
    rho, _ = spearmanr(sim_x[tri], sim_y[tri])
    return float(rho)



def _mknn_from_graphs(nn_x: np.ndarray, nn_y: np.ndarray, k: int) -> float:
    n = nn_x.shape[0]
    ax = np.ascontiguousarray(nn_x[:, :k])
    ay = np.ascontiguousarray(nn_y[:, :k])
    # Sort Y's neighbors per row, then for each of X's neighbors test membership via
    # searchsorted along the row. All operations are vectorized over (n, k).
    ay_sorted = np.sort(ay, axis=1)
    # searchsorted per row: positions where ax would insert into ay_sorted
    pos = np.empty_like(ax)
    rows = np.arange(n)[:, None]
    # vectorized row-wise searchsorted via take_along_axis trick
    for j in range(k):                       # k is small (<=50), loop over columns not rows
        pos[:, j] = (ay_sorted < ax[:, j:j+1]).sum(axis=1)
    pos = np.clip(pos, 0, k - 1)
    matched = np.take_along_axis(ay_sorted, pos, axis=1) == ax
    total = int(matched.sum())
    return total / (n * k)


def compute_all_metrics(
    X: np.ndarray, Y: np.ndarray,
    k_values: list = (5, 10, 20, 50),
    headline_k: int = 10,
    n_subsample_cka: int = 2000,
) -> dict:
    result: dict = {}
    k_values = list(k_values)
    k_max = max(k_values)

    # Cap N once (CKNNA's Gram matrices are O(N^2)); also keeps the kNN build bounded.
    N = X.shape[0]
    nn_x = _knn_graph(X, k_max)
    nn_y = _knn_graph(Y, k_max)

    for k in k_values:
        result[f"mknn_k{k}"]  = _mknn_from_graphs(nn_x, nn_y, k)
        result[f"cknna_k{k}"] = _cknna_from_graphs(nn_x[:, :k], nn_y[:, :k], N)

    result["cka_linear"]   = cka_linear(X, Y)
    result["cka_rbf"]      = cka_rbf(X, Y, n_subsample=n_subsample_cka)
    result["procrustes"]   = procrustes(X, Y)
    result["rsa_spearman"] = rsa_spearman(X, Y)
    result["headline_mknn"] = result[f"mknn_k{headline_k}"]
    result["headline_cknna"] = result[f"cknna_k{headline_k}"]
    return result