"""Matplotlib SVG plots + RESULTS.md writer for the toy-CLIP smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "ai-learn-12", "svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    names = []
    c = m["training"]["loss_curve"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(range(1, len(c) + 1), c, color="#126782", lw=1.6, label="train InfoNCE")
    ax.axhline(m["training"]["log_batch_size"], color="#999", ls="--", lw=1, label="log(batch) = chance")
    ax.set_xlabel("epoch")
    ax.set_ylabel("symmetric InfoNCE loss")
    ax.set_title("Contrastive training loss")
    ax.legend(fontsize=8)
    names.append(_save(fig, out / "infonce_loss.svg"))

    keys = ["i2t_R@1_concept", "t2i_R@1_concept", "i2t_R@5_concept", "t2i_R@5_concept", "i2t_R@5_instance", "t2i_R@5_instance"]
    rnd = m["retrieval_random"]
    rvals = [rnd["R@1_concept"], rnd["R@1_concept"], rnd["R@5_concept"], rnd["R@5_concept"], rnd["R@5_instance"], rnd["R@5_instance"]]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for off, lab, vals, col in [(-0.27, "random", rvals, "#adb5bd"), (0, "untrained", [m["retrieval_untrained"][k] for k in keys], "#8ecae6"),
                                (0.27, "trained CLIP", [m["retrieval_test"][k] for k in keys], "#126782")]:
        ax.bar(x + off, vals, 0.27, label=lab, color=col)
    ax.set_xticks(x, [k.replace("_", "\n", 1) for k in keys], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall")
    ax.set_title(f"Test retrieval (N={m['data']['n_test']})")
    ax.legend(fontsize=8)
    names.append(_save(fig, out / "retrieval_recall.svg"))

    groups = ["shape", "color"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(2)
    series = [("chance", [m["zero_shot_test"][f"{g}_chance"] for g in groups], "#adb5bd"),
              ("single prompt", [m["zero_shot_test"][f"{g}_acc_single_prompt"] for g in groups], "#8ecae6"),
              ("prompt ensemble", [m["zero_shot_test"][f"{g}_acc_prompt_ensemble"] for g in groups], "#126782"),
              ("held-out combos (ens.)", [m["zero_shot_heldout"][f"{g}_acc_prompt_ensemble"] for g in groups], "#e76f51")]
    for i, (lab, vals, col) in enumerate(series):
        ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, label=lab, color=col)
    ax.set_xticks(x, ["zero-shot shape (4-way)", "zero-shot color (4-way)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Zero-shot classification via text prompts")
    ax.legend(fontsize=7, loc="upper left")
    names.append(_save(fig, out / "zero_shot.svg"))
    return names


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_results_md(path: Path, m: Dict[str, Any]) -> None:
    rt, rh, rr, ru = m["retrieval_test"], m["retrieval_heldout"], m["retrieval_random"], m["retrieval_untrained"]
    zt, zh, zu = m["zero_shot_test"], m["zero_shot_heldout"], m["zero_shot_untrained"]
    L = [
        "# Results: toy CLIP from scratch (smoke run)", "",
        f"Seed {m['seed']}. Runtime {m['runtime_sec']}s on CPU. {m['data']['n_train']} train pairs "
        f"({m['data']['n_concepts_train']}/32 concepts, held-out colour-shape combos: {', '.join(m['config']['held_out_combos'])}), "
        f"{m['data']['n_test']} test pairs, {m['data']['n_heldout']} held-out-combo pairs. "
        f"Vocab {m['data']['vocab_size']}, embed dim {m['config']['embed_dim']}, tau {m['config']['tau']}, "
        f"{m['model']['trainable_params']} params.", "",
        f"InfoNCE loss: {m['training']['initial_loss']} (epoch 1) to {m['training']['final_loss']} (epoch {m['config']['epochs']}); "
        f"chance level is log(batch) = {m['training']['log_batch_size']}.", "",
        "## Retrieval (test gallery)", "",
        "*instance* means the exact paired item must be retrieved. Many items share a concept, so this is a harsh metric. "
        "*concept* means any item with the same (size, colour, shape).", "",
        "| metric | random | untrained | trained | held-out combos (trained) |", "|---|---|---|---|---|",
    ]
    for k in (1, 5):
        for d in ("i2t", "t2i"):
            for kind in ("concept", "instance"):
                key = f"{d}_R@{k}_{kind}"
                L.append(f"| {key} | {_pct(rr[f'R@{k}_{kind}'])} | {_pct(ru[key])} | **{_pct(rt[key])}** | {_pct(rh[key])} |")
    L += ["", "## Zero-shot classification (prompts: `a photo of a {shape}`, `something {colour}`)", "",
          "| task | chance | untrained (ens.) | single prompt | prompt ensemble | held-out combos (ens.) |", "|---|---|---|---|---|---|"]
    for g in ("shape", "color"):
        L.append(f"| {g} | {_pct(zt[g + '_chance'])} | {_pct(zu[g + '_acc_prompt_ensemble'])} | {_pct(zt[g + '_acc_single_prompt'])} | "
                 f"**{_pct(zt[g + '_acc_prompt_ensemble'])}** | {_pct(zh[g + '_acc_prompt_ensemble'])} |")
    L += ["", "## Takeaways", "",
          "- Contrastive training pulls matched image/caption pairs together. Concept-level recall ends far above random.",
          "- Instance-level recall stays low because many captions describe the same concept, and the model has no signal to separate near-duplicates.",
          "- Zero-shot works through text prompts alone. Colour is easier than shape for an 8x8 MLP encoder.",
          "- Held-out colour-shape combos (never seen together in training) still get classified, which shows the embedding composes attributes.",
          "- Caveat: the held-out gallery holds only 8 concepts (4 combos x 2 sizes) versus up to 32 in the test gallery, so its concept recall is easier and not directly comparable.", "",
          "Plots: " + ", ".join(f"`{p}`" for p in m["plots"]), ""]
    path.write_text("\n".join(L), encoding="utf-8")
