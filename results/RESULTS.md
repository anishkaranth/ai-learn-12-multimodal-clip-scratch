# Results: toy CLIP from scratch (smoke run)

Seed 42. Runtime 1.305s on CPU. 3000 train pairs (24/32 concepts, held-out colour-shape combos: yellow triangle, blue cross, red circle, green square), 200 test pairs, 200 held-out-combo pairs. Vocab 33, embed dim 32, tau 0.1, 33088 params.

InfoNCE loss: 3.6371 (epoch 1) to 0.46227 (epoch 40); chance level is log(batch) = 4.1589.

## Retrieval (test gallery)

*instance* means the exact paired item must be retrieved. Many items share a concept, so this is a harsh metric. *concept* means any item with the same (size, colour, shape).

| metric | random | untrained | trained | held-out combos (trained) |
|---|---|---|---|---|
| i2t_R@1_concept | 4.0% | 3.0% | **66.5%** | 82.5% |
| i2t_R@1_instance | 1.0% | 0.5% | **12.0%** | 2.0% |
| t2i_R@1_concept | 4.0% | 6.0% | **71.0%** | 82.5% |
| t2i_R@1_instance | 1.0% | 1.0% | **12.0%** | 3.0% |
| i2t_R@5_concept | 14.5% | 14.0% | **89.0%** | 94.0% |
| i2t_R@5_instance | 1.5% | 1.5% | **37.5%** | 13.0% |
| t2i_R@5_concept | 14.5% | 13.0% | **94.5%** | 99.5% |
| t2i_R@5_instance | 1.5% | 2.0% | **37.5%** | 13.5% |

## Zero-shot classification (prompts: `a photo of a {shape}`, `something {colour}`)

| task | chance | untrained (ens.) | single prompt | prompt ensemble | held-out combos (ens.) |
|---|---|---|---|---|---|
| shape | 25.0% | 27.5% | 56.5% | **66.5%** | 75.0% |
| color | 25.0% | 14.5% | 80.0% | **88.5%** | 91.5% |

## Takeaways

- Contrastive training pulls matched image/caption pairs together. Concept-level recall ends far above random.
- Instance-level recall stays low because many captions describe the same concept, and the model has no signal to separate near-duplicates.
- Zero-shot works through text prompts alone. Colour is easier than shape for an 8x8 MLP encoder.
- Held-out colour-shape combos (never seen together in training) still get classified, which shows the embedding composes attributes.
- Caveat: the held-out gallery holds only 8 concepts (4 combos x 2 sizes) versus up to 32 in the test gallery, so its concept recall is easier and not directly comparable.

Plots: `infonce_loss.svg`, `retrieval_recall.svg`, `zero_shot.svg`
