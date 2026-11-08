"""
Graph neural network surrogate of the car user equilibrium (Section II-G of the paper).

Added next to the PCA + ridge surrogate of ``surrogate.py`` (which is kept unchanged); both have
the same interface (``cand_roads``, ``predict_raw``), so ``problems.MultiModal`` can use either
(``config.SURROGATE_MODEL``).  The feasibility projection of ``surrogate.FlowProjector`` is
applied to the output of both models.

Graph.  The car network itself: nodes = intersections (4,720), edges = directed arcs (12,350).
Every arc carries the state h_a; one message-passing layer
    m_v^out = mean_{a: tail(a) = v} h_a,   m_v^in = mean_{a: head(a) = v} h_a
    h_v     = MLP_v([m_v^out, m_v^in, g])                   g = global mean of the arc states
    h_a    <- LayerNorm(h_a + MLP_a([h_a, h_tail(a), h_head(a)]))
moves information one intersection further, so L layers see the detours of up to L hops; the
global term g carries network-wide information (total capacity lost, demand level) in one step.

Arc features (standardised)
    static : log t0, log capacity, lanes, today's v/c, log(1 + phi_0), BPR marginal cost
             0.6 (phi_0/theta_0)^4, candidate road (0/1)
    design : capacity reduction r_a = 1 - theta/theta_0, demand scale s - 1,
             new v/c  s phi_0 / theta, and the "time shock" log t(s phi_0; theta) / t(s phi_0; theta_0)
             (the delay the arc would have if nobody re-routed - the reason for re-routing)
Target: the deviation phi - s phi_0 divided by its standard deviation (as for the ridge model).
Loss: mean squared error; early stopping on the validation R2 of the deviation.

Written with plain PyTorch sparse products (no PyTorch-Geometric needed); CPU only.
"""
import copy
import time

import numpy as np
import torch
from torch import nn

from . import config as C


def _mlp(n_in, n_out, hidden):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, n_out))


class _Net(nn.Module):
    def __init__(self, n_feat, hidden, layers, tail, head, n_nodes):
        super().__init__()
        self.register_buffer("tail", torch.as_tensor(tail, dtype=torch.long))
        self.register_buffer("head", torch.as_tensor(head, dtype=torch.long))
        self.n_nodes, self.hidden = n_nodes, hidden
        self.register_buffer("w_out", self._mean_weights(tail, n_nodes))
        self.register_buffer("w_in", self._mean_weights(head, n_nodes))
        self.enc = _mlp(n_feat, hidden, hidden)
        self.node = nn.ModuleList([_mlp(3 * hidden, 2 * hidden, hidden) for _ in range(layers)])
        # edge update  h_a + W2 SiLU(W_e h_a + (W_t h_v)[tail] + (W_h h_v)[head]); the node maps
        # W_t, W_h are folded into the node MLP output (2H = [W_t h_v, W_h h_v]), so only
        # H-wide tensors are gathered on the 12,350 arcs (memory traffic is the CPU bottleneck)
        self.edge_in = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.edge_out = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dec = _mlp(hidden, 1, hidden)

    @staticmethod
    def _mean_weights(node_of_arc, n_nodes):
        """1 / degree of the node of every arc (mean aggregation)."""
        deg = np.bincount(node_of_arc, minlength=n_nodes).astype(np.float32)
        return torch.as_tensor(1.0 / np.maximum(deg[node_of_arc], 1.0))[None, :, None]

    def _aggregate(self, node_of_arc, w, h):
        """(B, A, H) arc states -> (B, N, H) mean over the arcs of every node."""
        out = h.new_zeros(h.shape[0], self.n_nodes, h.shape[2])
        return out.index_add(1, node_of_arc, h * w)

    def forward(self, x):
        h = self.enc(x)                                            # (B, A, H)
        H = self.hidden
        for node, e_in, e_out, norm in zip(self.node, self.edge_in, self.edge_out, self.norm):
            g = h.mean(1, keepdim=True).expand(-1, self.n_nodes, -1)
            hv = node(torch.cat([self._aggregate(self.tail, self.w_out, h),
                                 self._aggregate(self.head, self.w_in, h), g], -1))
            z = e_in(h) + hv[:, self.tail, :H] + hv[:, self.head, H:]
            h = norm(h + e_out(nn.functional.silu(z)))
        return self.dec(h).squeeze(-1)                              # (B, A)


class EnsembleSurrogate:
    """Average of several surrogates with the same interface (e.g. ridge + GNN)."""

    kind = "ensemble"

    def __init__(self, *models):
        self.models = models
        self.cand_roads = models[0].cand_roads

    def predict_raw(self, road_reduction, scale):
        return np.mean([m.predict_raw(road_reduction, scale) for m in self.models], axis=0)


def load_surrogate(model=None):
    """(surrogate, correction iterations) according to config.SURROGATE_MODEL (see config)."""
    from . import utils
    from .surrogate import CarSurrogate
    model = model or C.SURROGATE_MODEL
    iters = C.SURROGATE_UE_ITERS
    choice_file = C.RESULTS / "03b_surrogate_choice.json"
    if model == "auto":
        if choice_file.exists():
            ch = utils.load_json(choice_file)
            model, iters = ch["model"], ch["iterations"]
        else:
            model = "ridge"
    ridge = lambda: CarSurrogate.load(C.RESULTS / "03_surrogate.pkl")      # noqa: E731
    gnn = lambda: GNNSurrogate.load(C.RESULTS / "03b_surrogate_gnn.pt")   # noqa: E731
    if model == "ridge":
        return ridge(), iters
    if model == "gnn":
        return gnn(), iters
    if model == "ensemble":
        return EnsembleSurrogate(ridge(), gnn()), iters
    raise ValueError(f"unknown surrogate model {model!r}")


class GNNSurrogate:
    """GNN surrogate of the UE flows (see module docstring); same interface as CarSurrogate."""

    kind = "gnn"

    def __init__(self, car, phi0, cand_roads, hidden=48, layers=8, seed=0):
        self.phi0 = np.asarray(phi0, float)
        self.cand_roads = np.asarray(cand_roads)
        self.hidden, self.layers, self.seed = hidden, layers, seed
        self.E, self.N = car.E, car.N
        self.tail, self.head = car.tail.copy(), car.head.copy()
        self.t0, self.cap0 = car.t0.copy(), car.capacity0.copy()
        vc = self.phi0 / self.cap0
        cand = np.zeros(2 * self.E)
        cand[self.cand_roads] = cand[self.cand_roads + self.E] = 1.0
        self.static = np.c_[np.log(self.t0), np.log(self.cap0), car.lanes, vc, np.log1p(self.phi0),
                            C.BPR_MU * vc ** C.BPR_SIGMA, cand].astype(np.float32)

    # ---- features ---------------------------------------------------------------------------
    def _bpr(self, x, cap):
        return 1.0 + C.BPR_MU * (x / cap) ** C.BPR_SIGMA

    def features(self, road_reduction, scale):
        """(n, A, F) arc features of n designs (road_reduction: (n, n_cand), scale: (n,))."""
        red = np.atleast_2d(np.asarray(road_reduction, float))
        scale = np.atleast_1d(np.asarray(scale, float))
        n, A = len(red), 2 * self.E
        r = np.zeros((n, A))
        r[:, self.cand_roads] = red
        r[:, self.cand_roads + self.E] = red
        cap = self.cap0 * (1 - r)
        x = scale[:, None] * self.phi0
        shock = np.log(self._bpr(x, cap) / self._bpr(x, self.cap0))
        dyn = np.stack([r, np.broadcast_to(10 * (scale[:, None] - 1), (n, A)), x / cap, shock], -1)
        F = np.concatenate([np.broadcast_to(self.static, (n, A, self.static.shape[1])), dyn], -1)
        return F[:, self.active].astype(np.float32)

    def _standardise(self, F):
        return (F - self.f_mu) / self.f_sd

    # ---- graph pruning ---------------------------------------------------------------------
    def _set_active(self, train_flows, min_flow=1.0):
        """Arcs kept in the GNN graph: loaded today, loaded in some training sample, or on a
        candidate road.  About half of the arcs (residential streets, the "wrong" direction of
        some links) never carry 1 veh/h in any equilibrium; they are dropped from the graph
        (their flow stays s * phi_0 ~ 0), which halves the training time on CPU."""
        cand = self.static[:, -1] > 0
        self.active = (train_flows.max(0) >= min_flow) | (self.phi0 >= min_flow) | cand
        nodes = np.unique(np.r_[self.tail[self.active], self.head[self.active]])
        renum = np.full(self.N, -1)
        renum[nodes] = np.arange(len(nodes))
        self.g_tail, self.g_head, self.g_nodes = renum[self.tail[self.active]], renum[self.head[self.active]], len(nodes)

    def _build_net(self, n_feat):
        return _Net(n_feat, self.hidden, self.layers, self.g_tail, self.g_head, self.g_nodes)

    # ---- training ---------------------------------------------------------------------------
    def fit(self, road_reduction, scale, flows, val_fraction=0.15, epochs=40, batch=4, lr=2e-3,
            patience=8, threads=8, verbose=True):
        torch.manual_seed(self.seed)
        torch.set_num_threads(threads)
        scale = np.asarray(scale, float)
        Y = flows - np.outer(scale, self.phi0)
        n = len(Y)
        idx = np.random.default_rng(self.seed).permutation(n)
        n_val = max(2, int(val_fraction * n))         # >= 2 so the R2 is defined
        va, tr = idx[:n_val], idx[n_val:]
        self._set_active(Y[tr] + np.outer(scale[tr], self.phi0))
        F = self.features(road_reduction, scale)
        Y = Y[:, self.active]
        self.f_mu = F[tr].reshape(-1, F.shape[-1]).mean(0)
        self.f_sd = F[tr].reshape(-1, F.shape[-1]).std(0) + 1e-6
        self.y_sd = float(Y[tr].std())
        X = torch.as_tensor(self._standardise(F))
        T = torch.as_tensor((Y / self.y_sd).astype(np.float32))
        self.net = self._build_net(F.shape[-1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Yva = Y[va]
        ss_va = max(float(np.sum((Yva - Yva.mean(0)) ** 2)), 1e-12)
        best, best_state, wait = -np.inf, None, 0
        self.history = []
        rng = np.random.default_rng(self.seed + 1)
        for ep in range(epochs):
            t0 = time.time()
            self.net.train()
            perm = rng.permutation(tr)
            loss_sum = 0.0
            for k in range(0, len(perm), batch):
                b = perm[k:k + batch]
                opt.zero_grad()
                loss = torch.mean((self.net(X[b]) - T[b]) ** 2)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item() * len(b)
            sched.step()
            pred = self._predict_scaled(X[va]) * self.y_sd
            r2 = 1 - float(np.sum((pred - Yva) ** 2)) / ss_va
            self.history.append(dict(epoch=ep, loss=loss_sum / len(tr), val_R2_deviation=r2, sec=time.time() - t0))
            if verbose:
                print(f"    epoch {ep:3d}: loss {loss_sum / len(tr):.4f}, validation R2 (deviation) {r2:.4f} "
                      f"({time.time() - t0:.0f} s)", flush=True)
            if r2 > best:
                best, best_state, wait = r2, copy.deepcopy(self.net.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break
        self.net.load_state_dict(best_state)
        self.val_R2_deviation = best
        return dict(train=tr, val=va, val_R2_deviation=best, epochs=len(self.history))

    # ---- prediction -------------------------------------------------------------------------
    @torch.no_grad()
    def _predict_scaled(self, X, batch=8):
        self.net.eval()
        return np.concatenate([self.net(X[k:k + batch]).numpy() for k in range(0, len(X), batch)]).astype(float)

    def predict_raw(self, road_reduction, scale):
        scale = np.atleast_1d(np.asarray(scale, float))
        X = torch.as_tensor(self._standardise(self.features(road_reduction, scale)))
        out = np.outer(scale, self.phi0)
        out[:, self.active] += self._predict_scaled(X) * self.y_sd
        return out

    # ---- persistence ------------------------------------------------------------------------
    def save(self, path):
        state = {k: v for k, v in self.__dict__.items() if k != "net"}
        torch.save(dict(meta=state, net=self.net.state_dict(), n_feat=int(self.f_mu.shape[0])), path)

    @staticmethod
    def load(path):
        d = torch.load(path, weights_only=False)
        obj = GNNSurrogate.__new__(GNNSurrogate)
        obj.__dict__.update(d["meta"])
        obj.net = obj._build_net(d["n_feat"])
        obj.net.load_state_dict(d["net"])
        obj.net.eval()
        return obj
