"""
Machine-learning surrogate of the car user equilibrium with feasibility projection
(Section II-G of the paper).

    M : (theta(w), q_c(w))  ->  phi_hat          learned on exact UE samples (step 2)
    phi* = argmin_{phi >= 0} || phi_hat - phi ||^2   s.t.  B phi = d                 Eq. (27)

Inputs.  A bike configuration w changes the car network in two ways only: the capacities
of the roads along upgraded bike arcs (coupling.py) and a uniform scaling s of the car demand
(mode shift).  The features are therefore the capacity reduction of each candidate road
(1 - theta / theta0, 0 if unchanged) and s - 1.

Model (change w.r.t. the paper: an MLP on principal components instead of a GCN; the paper's
code is not available and the environment has no deep-learning library):
    target   = phi - s * phi_0          (deviation from the scaled pre-upgrade UE)
    PCA      -> K components (captures > 99 % of the variance of the deviations)
    MLP      features -> K scores  (scikit-learn), compared with a ridge regression;
             the better one on the validation samples is kept.

Projection.  B is the node-arc incidence matrix and d the node balance of the OD demand.
Without the sign constraint the projection has a closed form
    phi = phi_hat - B^T (B B^T)^-1 (B phi_hat - d),
where B B^T is the graph Laplacian (factorised once, one node grounded).  Negative flows are
then clipped and the projection repeated (alternating projections, a few iterations).
"""
import pickle

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class FlowProjector:
    """Projection of a flow vector onto {phi >= 0, B phi = d}."""

    def __init__(self, car):
        self.car = car
        n_arcs = 2 * car.E
        rows = np.r_[car.tail, car.head]
        cols = np.r_[np.arange(n_arcs), np.arange(n_arcs)]
        vals = np.r_[np.ones(n_arcs), -np.ones(n_arcs)]
        self.B = csc_matrix((vals, (rows, cols)), shape=(car.N, n_arcs))     # out - in
        Lap = (self.B @ self.B.T).tocsc()
        self.keep = np.arange(1, car.N)                                     # ground node 0
        self.lu = splu(Lap[self.keep][:, self.keep].tocsc())
        self.BT = self.B.T.tocsr()

    def balance(self, scale=1.0):
        """d: net outflow of every node for the current car demand (veh/h)."""
        car = self.car
        d = np.zeros(car.N)
        np.add.at(d, car.sources, (car.Q.sum(1) - car.Q.sum(0)) * scale)
        return d

    def project(self, phi_hat, d, iters=10):
        phi = np.asarray(phi_hat, float).copy()
        for _ in range(iters):
            r = self.B @ phi - d
            y = np.zeros(self.car.N)
            y[self.keep] = self.lu.solve(r[self.keep])
            phi = phi - self.BT @ y
            if phi.min() >= -1e-6:
                break
            phi = np.maximum(phi, 0.0)
        return np.maximum(phi, 0.0)


class CarSurrogate:
    """PCA + regression surrogate of the UE flows (see module docstring)."""

    def __init__(self, phi0, cand_roads, n_components=60, hidden=(256, 256), seed=0):
        self.phi0 = np.asarray(phi0, float)
        self.cand_roads = np.asarray(cand_roads)
        self.n_components, self.hidden, self.seed = n_components, hidden, seed

    # ---- features / targets -------------------------------------------------------------
    def features(self, road_reduction, scale):
        """road_reduction: (n, n_cand) 1 - theta/theta0 on the candidate roads; scale: (n,)."""
        return np.c_[np.asarray(road_reduction, float), np.asarray(scale, float) - 1.0]

    def fit(self, road_reduction, scale, flows, val_fraction=0.15, verbose=True):
        X = self.features(road_reduction, scale)
        Y = flows - np.outer(scale, self.phi0)
        n = len(X)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        n_val = max(1, int(val_fraction * n))
        va, tr = idx[:n_val], idx[n_val:]
        self.xs = StandardScaler().fit(X[tr])
        self.pca = PCA(min(self.n_components, len(tr) - 1), random_state=self.seed).fit(Y[tr])
        Z = self.pca.transform(Y[tr])
        self.zs = StandardScaler().fit(Z)
        Xt, Zt = self.xs.transform(X[tr]), self.zs.transform(Z)
        cands = {
            **{f"ridge_{a:g}": Ridge(alpha=a) for a in (0.1, 1.0, 10.0)},
            "mlp": MLPRegressor(hidden_layer_sizes=self.hidden, alpha=1e-3, learning_rate_init=1e-3, max_iter=800,
                                early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
                                random_state=self.seed),
        }
        self.scores = {}
        for name, mdl in cands.items():
            mdl.fit(Xt, Zt)
            pred = self._decode(mdl.predict(self.xs.transform(X[va])), scale[va])
            err = pred - flows[va]
            self.scores[name] = float(1 - np.sum(err ** 2) / np.sum((flows[va] - flows[va].mean(0)) ** 2))
            if verbose:
                print(f"    {name:9s}: validation R2 (flows) = {self.scores[name]:.4f}")
        self.kind = max(self.scores, key=self.scores.get)
        self.model = cands[self.kind]
        self.explained = float(self.pca.explained_variance_ratio_.sum())
        self.loss_curve = list(getattr(cands["mlp"], "loss_curve_", []))
        self.val_loss_curve = list(getattr(cands["mlp"], "validation_scores_", []))
        return dict(train=tr, val=va, scores=self.scores, kind=self.kind, explained=self.explained)

    def _decode(self, Zs, scale):
        Z = self.zs.inverse_transform(np.atleast_2d(Zs))
        return self.pca.inverse_transform(Z) + np.outer(np.atleast_1d(scale), self.phi0)

    def predict_raw(self, road_reduction, scale):
        X = self.xs.transform(self.features(np.atleast_2d(road_reduction), np.atleast_1d(scale)))
        return self._decode(self.model.predict(X), np.atleast_1d(scale))

    # ---- persistence ----------------------------------------------------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
