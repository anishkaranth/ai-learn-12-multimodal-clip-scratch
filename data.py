"""Synthetic 8x8 RGB 'images' of shapes + bag-of-words captions.

Concept = (size, color, shape): 2 sizes x 4 colors x 4 shapes = 32 concepts.
Images are drawn procedurally with random position jitter, colour jitter and pixel noise.
Captions come from templates with synonyms ("red"/"crimson", "square"/"box", ...), so the
text encoder has to learn that different words mean the same thing.
A few (color, shape) combos are HELD OUT of training to test compositional zero-shot.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

H = W = 8
SHAPES = ["square", "circle", "triangle", "cross"]
COLORS = ["red", "green", "blue", "yellow"]
SIZES = ["small", "large"]
_RGB = {"red": (0.9, 0.1, 0.1), "green": (0.1, 0.8, 0.2), "blue": (0.15, 0.25, 0.95), "yellow": (0.95, 0.9, 0.1)}
SYN = {
    "square": ["square", "box", "block"], "circle": ["circle", "ring", "disc"],
    "triangle": ["triangle", "wedge"], "cross": ["cross", "plus"],
    "red": ["red", "crimson"], "green": ["green", "emerald"], "blue": ["blue", "navy"], "yellow": ["yellow", "golden"],
    "small": ["small", "tiny", "little"], "large": ["large", "big", "huge"],
}
TEMPLATES = ["a {size} {color} {shape}", "{color} {shape} that is {size}", "picture of a {size} {color} {shape}",
             "the {shape} is {color} and {size}"]
HELD_OUT = [("yellow", "triangle"), ("blue", "cross"), ("red", "circle"), ("green", "square")]


def _mask(shape: str, size: str, cy: int, cx: int) -> np.ndarray:
    r = 1 if size == "small" else 2
    yy, xx = np.mgrid[0:H, 0:W]
    dy, dx = yy - cy, xx - cx
    if shape == "square":
        m = (np.abs(dy) <= r) & (np.abs(dx) <= r)
    elif shape == "circle":
        d2 = dy ** 2 + dx ** 2
        m = (d2 <= (r + 0.5) ** 2) & (d2 >= (r - 0.5) ** 2 if r > 1 else d2 >= 1)
    elif shape == "triangle":
        m = (dy >= -r) & (dy <= r) & (np.abs(dx) <= (dy + r) / 2 + 0.01)
    else:  # cross
        m = ((np.abs(dy) <= r) & (dx == 0)) | ((np.abs(dx) <= r) & (dy == 0))
    return m.astype(float)


def draw(size: str, color: str, shape: str, rng: np.random.Generator) -> np.ndarray:
    r = 1 if size == "small" else 2
    cy, cx = rng.integers(r, H - r), rng.integers(r, W - r)
    m = _mask(shape, size, cy, cx)
    rgb = np.clip(np.array(_RGB[color]) + rng.normal(0, 0.05, 3), 0, 1)
    img = m[..., None] * rgb[None, None, :]
    img += rng.normal(0, 0.08, img.shape)
    return np.clip(img, 0, 1)


def caption(size: str, color: str, shape: str, rng: np.random.Generator) -> str:
    t = TEMPLATES[rng.integers(len(TEMPLATES))]
    pick = lambda k: SYN[k][rng.integers(len(SYN[k]))]  # noqa: E731
    return t.format(size=pick(size), color=pick(color), shape=pick(shape))


def make_dataset(n: int, rng: np.random.Generator, split: str = "train") -> Dict[str, object]:
    """split='train' excludes HELD_OUT (color, shape) combos; 'test' samples all; 'heldout' only held-out."""
    concepts = [(s, c, sh) for s in SIZES for c in COLORS for sh in SHAPES]
    if split == "train":
        concepts = [x for x in concepts if (x[1], x[2]) not in HELD_OUT]
    elif split == "heldout":
        concepts = [x for x in concepts if (x[1], x[2]) in HELD_OUT]
    imgs, caps, labels = [], [], []
    for _ in range(n):
        s, c, sh = concepts[rng.integers(len(concepts))]
        imgs.append(draw(s, c, sh, rng))
        caps.append(caption(s, c, sh, rng))
        labels.append((s, c, sh))
    return {"images": np.stack(imgs), "captions": caps, "labels": labels}


_TOK = re.compile(r"[a-z]+")


def build_vocab(captions: Sequence[str]) -> Dict[str, int]:
    words = sorted({w for c in captions for w in _TOK.findall(c.lower())})
    return {w: i for i, w in enumerate(words)}


def bow(texts: Sequence[str], vocab: Dict[str, int]) -> np.ndarray:
    X = np.zeros((len(texts), len(vocab)))
    for i, t in enumerate(texts):
        for w in _TOK.findall(t.lower()):
            if w in vocab:
                X[i, vocab[w]] = 1.0
    return X


def full_vocab() -> Dict[str, int]:
    """Vocab over every template word + synonym so zero-shot prompts are always in-vocabulary."""
    words = set()
    for t in TEMPLATES + ["a photo of a {shape}", "something {color}"]:
        words.update(w for w in _TOK.findall(t) if w not in ("size", "color", "shape"))
    for v in SYN.values():
        words.update(v)
    words.update(["photo", "something"])
    return {w: i for i, w in enumerate(sorted(words))}
