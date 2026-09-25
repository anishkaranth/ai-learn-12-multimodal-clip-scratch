#!/usr/bin/env python3
"""Toy-CLIP smoke: synthetic shapes + captions -> contrastive training -> retrieval + zero-shot -> results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from clip import encode, init_encoder, recall_at_k, train_clip
from data import COLORS, HELD_OUT, SHAPES, SYN, bow, full_vocab, make_dataset
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
CFG = {
    "image_shape": [8, 8, 3], "n_train": 3000, "n_test": 200, "n_heldout": 200, "img_hidden": 128, "txt_hidden": 64,
    "embed_dim": 32, "tau": 0.1, "epochs": 40, "batch": 64, "lr": 3e-3, "ks": [1, 5],
    "held_out_combos": [f"{c} {s}" for c, s in HELD_OUT],
}


def _compact(js: str) -> str:
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)


def _flat(d):
    return d["images"].reshape(len(d["images"]), -1)


def retrieval(e_img, e_txt, labels, ks):
    n = len(labels)
    S = e_img @ e_txt.T
    inst = np.eye(n, dtype=bool)
    conc = np.array([[a == b for b in labels] for a in labels])
    out = {}
    for k in ks:
        out[f"i2t_R@{k}_instance"] = recall_at_k(S, k, inst)
        out[f"t2i_R@{k}_instance"] = recall_at_k(S.T, k, inst)
        out[f"i2t_R@{k}_concept"] = recall_at_k(S, k, conc)
        out[f"t2i_R@{k}_concept"] = recall_at_k(S.T, k, conc)
    return out


def zero_shot(p, V, d):
    """Classify images by cosine to text prompts. 'single' = one prompt/class; 'ensemble' = mean over synonyms."""
    e, _ = encode(p, _flat(d), "img_")
    ys = np.array([SHAPES.index(l[2]) for l in d["labels"]])
    yc = np.array([COLORS.index(l[1]) for l in d["labels"]])
    out = {}
    for task, classes, y, tmpl in [("shape", SHAPES, ys, "a photo of a {}"), ("color", COLORS, yc, "something {}")]:
        single, _ = encode(p, bow([tmpl.format(c) for c in classes], V), "txt_")
        ens = []
        for c in classes:
            et, _ = encode(p, bow([tmpl.format(s) for s in SYN[c]], V), "txt_")
            m = et.mean(0)
            ens.append(m / np.linalg.norm(m))
        ens = np.stack(ens)
        out[f"{task}_acc_single_prompt"] = float(((e @ single.T).argmax(1) == y).mean())
        out[f"{task}_acc_prompt_ensemble"] = float(((e @ ens.T).argmax(1) == y).mean())
        out[f"{task}_chance"] = 1 / len(classes)
    return out


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    tr = make_dataset(CFG["n_train"], rng, "train")
    te = make_dataset(CFG["n_test"], rng, "test")
    ho = make_dataset(CFG["n_heldout"], rng, "heldout")
    V = full_vocab()
    p = {**init_encoder(_flat(tr).shape[1], CFG["img_hidden"], CFG["embed_dim"], rng, "img_"),
         **init_encoder(len(V), CFG["txt_hidden"], CFG["embed_dim"], rng, "txt_")}
    n_params = int(sum(v.size for v in p.values()))

    def evaluate(params, d):
        ei, _ = encode(params, _flat(d), "img_")
        et, _ = encode(params, bow(d["captions"], V), "txt_")
        return retrieval(ei, et, d["labels"], CFG["ks"])

    untrained = {"test": evaluate(p, te), "zero_shot_test": zero_shot(p, V, te)}
    loss_hist = train_clip(p, _flat(tr), bow(tr["captions"], V), epochs=CFG["epochs"], batch=CFG["batch"],
                           lr=CFG["lr"], tau=CFG["tau"], rng=rng)
    # random-score baseline through the same recall code
    rr = np.random.default_rng(SEED + 1)
    n = len(te["labels"])
    conc = np.array([[a == b for b in te["labels"]] for a in te["labels"]])
    R = rr.normal(size=(n, n))
    random_base = {}
    for k in CFG["ks"]:
        random_base[f"R@{k}_instance"] = recall_at_k(R, k, np.eye(n, dtype=bool))
        random_base[f"R@{k}_concept"] = recall_at_k(R, k, conc)
        random_base[f"R@{k}_instance_expected"] = k / n

    metrics = {
        "project": "ai-learn-12-multimodal-clip-scratch", "seed": SEED, "config": CFG,
        "data": {"n_train": len(tr["labels"]), "n_test": n, "n_heldout": len(ho["labels"]), "vocab_size": len(V),
                 "n_concepts_total": 32, "n_concepts_train": 32 - 2 * len(HELD_OUT)},
        "model": {"trainable_params": n_params},
        "training": {"loss_curve": loss_hist, "initial_loss": loss_hist[0], "final_loss": loss_hist[-1],
                     "log_batch_size": round(float(np.log(CFG["batch"])), 4)},
        "retrieval_test": evaluate(p, te), "retrieval_heldout": evaluate(p, ho), "retrieval_random": random_base,
        "retrieval_untrained": untrained["test"],
        "zero_shot_test": zero_shot(p, V, te), "zero_shot_heldout": zero_shot(p, V, ho),
        "zero_shot_untrained": untrained["zero_shot_test"],
    }
    metrics["runtime_sec"] = round(time.perf_counter() - t0, 3)
    RESULTS.mkdir(exist_ok=True)
    metrics["plots"] = make_plots(RESULTS, metrics)
    (RESULTS / "metrics.json").write_text(_compact(json.dumps(metrics, indent=2)) + "\n", encoding="utf-8")
    rt, zt, zh = metrics["retrieval_test"], metrics["zero_shot_test"], metrics["zero_shot_heldout"]
    shot = {
        "project": metrics["project"], "seed": SEED,
        "config": {k: CFG[k] for k in ("embed_dim", "tau", "epochs", "batch", "lr", "n_train")},
        "key_metrics": {
            "final_infonce_loss": loss_hist[-1],
            "i2t_R@1_concept": rt["i2t_R@1_concept"], "t2i_R@1_concept": rt["t2i_R@1_concept"],
            "i2t_R@5_concept": rt["i2t_R@5_concept"], "random_R@1_concept": random_base["R@1_concept"],
            "i2t_R@5_instance": rt["i2t_R@5_instance"], "random_R@5_instance": random_base["R@5_instance"],
            "zero_shot_shape_acc_ensemble": zt["shape_acc_prompt_ensemble"],
            "zero_shot_color_acc_ensemble": zt["color_acc_prompt_ensemble"],
            "heldout_zero_shot_shape_acc": zh["shape_acc_prompt_ensemble"],
        },
        "runtime_sec": metrics["runtime_sec"],
    }
    (RESULTS / "JSON.shot").write_text(json.dumps(shot, indent=2) + "\n", encoding="utf-8")
    write_results_md(RESULTS / "RESULTS.md", metrics)
    print(json.dumps(shot, indent=2))


if __name__ == "__main__":
    main()
