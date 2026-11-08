"""Small helpers (logging, json, cluster diagnostics)."""
import json
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:8.1f} s] {msg}", flush=True)


def save_json(obj, path):
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, default=conv)


def components_per_cluster(net, labels):
    """Number of clusters whose nodes are NOT connected by the arcs inside the cluster."""
    labels = np.asarray(labels)
    inner = labels[net.u] == labels[net.v]
    A = csr_matrix((np.ones(inner.sum()), (net.u[inner], net.v[inner])), shape=(net.N, net.N))
    _, comp = connected_components(A, directed=False)
    pairs = np.unique(labels.astype(np.int64) * (net.N + 1) + comp)
    per_cluster = np.bincount((pairs // (net.N + 1)).astype(int))
    return int((per_cluster > 1).sum())
