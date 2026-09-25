"""Toy CLIP in NumPy: two encoders + projection heads, symmetric InfoNCE, manual backprop.

image encoder:  x_img (192) -> Linear -> ReLU -> Linear (proj) -> L2 normalize -> e_img (d)
text encoder:   x_bow (V)   -> Linear -> ReLU -> Linear (proj) -> L2 normalize -> e_txt (d)
logits = e_img @ e_txt.T / tau
loss   = 0.5 * [CE(logits, diag) + CE(logits.T, diag)]      (InfoNCE, both directions)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def init_encoder(d_in: int, d_hid: int, d_out: int, rng: np.random.Generator, prefix: str) -> Dict[str, np.ndarray]:
    return {
        f"{prefix}W1": rng.normal(0, np.sqrt(2 / d_in), (d_in, d_hid)), f"{prefix}b1": np.zeros(d_hid),
        f"{prefix}W2": rng.normal(0, np.sqrt(1 / d_hid), (d_hid, d_out)), f"{prefix}b2": np.zeros(d_out),
    }


def encode(p: Dict[str, np.ndarray], X: np.ndarray, prefix: str):
    h_pre = X @ p[f"{prefix}W1"] + p[f"{prefix}b1"]
    h = np.maximum(h_pre, 0)
    z = h @ p[f"{prefix}W2"] + p[f"{prefix}b2"]
    nrm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    return z / nrm, (X, h_pre, h, z, nrm)


def encoder_backward(p, cache, de, prefix):
    X, h_pre, h, z, nrm = cache
    e = z / nrm
    dz = (de - e * np.sum(de * e, axis=1, keepdims=True)) / nrm  # grad through L2 normalisation
    g = {f"{prefix}W2": h.T @ dz, f"{prefix}b2": dz.sum(0)}
    dh = dz @ p[f"{prefix}W2"].T * (h_pre > 0)
    g[f"{prefix}W1"] = X.T @ dh
    g[f"{prefix}b1"] = dh.sum(0)
    return g


def _xent_rows(logits: np.ndarray) -> Tuple[float, np.ndarray]:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(p[np.arange(n), np.arange(n)] + 1e-12).mean()
    g = p.copy()
    g[np.arange(n), np.arange(n)] -= 1
    return float(loss), g / n


def info_nce(e_img: np.ndarray, e_txt: np.ndarray, tau: float):
    logits = e_img @ e_txt.T / tau
    l_i2t, g_i2t = _xent_rows(logits)
    l_t2i, g_t2i = _xent_rows(logits.T)
    dlogits = 0.5 * (g_i2t + g_t2i.T)
    de_img = dlogits @ e_txt / tau
    de_txt = dlogits.T @ e_img / tau
    return 0.5 * (l_i2t + l_t2i), de_img, de_txt


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, g):
        self.t += 1
        for k in g:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh, vh = self.m[k] / (1 - self.b1 ** self.t), self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train_clip(p, X_img, X_txt, *, epochs, batch, lr, tau, rng) -> List[float]:
    opt = Adam(p, lr=lr)
    hist = []
    n = len(X_img)
    for _ in range(epochs):
        idx = rng.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n - batch + 1, batch):
            b = idx[s:s + batch]
            ei, ci = encode(p, X_img[b], "img_")
            et, ct = encode(p, X_txt[b], "txt_")
            loss, dei, det = info_nce(ei, et, tau)
            g = encoder_backward(p, ci, dei, "img_")
            g.update(encoder_backward(p, ct, det, "txt_"))
            opt.step(g)
            tot += loss
            nb += 1
        hist.append(round(tot / nb, 5))
    return hist


def recall_at_k(sim: np.ndarray, k: int, match: np.ndarray) -> float:
    """sim: (Nq, Ng) scores; match[i, j] True if gallery j is a correct answer for query i."""
    topk = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    return float(np.mean([match[i, topk[i]].any() for i in range(sim.shape[0])]))
