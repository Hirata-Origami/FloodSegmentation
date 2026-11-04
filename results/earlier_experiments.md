# Earlier experiments (numbers copied from the run logs in `results/logs/`)

All IoU values are water-class IoU on Sen1Floods11 hand-labelled chips (label -1 ignored). "LOEO" = leave-one-event-out over the 11 events.

## 01 Baseline encoder ensemble with weak-label pretraining (official split)
Nine models (resnet34, resnext50_32x4d, mit_b2 x 3 seeds), weak-label pretraining with held-out and Bolivia weak chips removed.

| Model | valid | test (95% CI) | Bolivia |
|---|---|---|---|
| ALL 9-model ensemble | 0.687 | 0.671 (0.557-0.756) | 0.546 |
| resnet34, seed 0, no weak pretraining | 0.667 | 0.693 | 0.457 |
| resnet34, seed 0, with weak pretraining | 0.668 | 0.654 | 0.376 |

Weak-label pretraining did not help (same validation IoU, lower test IoU; single seed).

## 02 Terrain ablation (JRC seasonality, NASADEM elevation, slope, HAND as extra input channels)
| | S1 only | S1 + HAND + slope | S1 + all terrain |
|---|---|---|---|
| LOEO macro IoU (resnet34, 1 seed) | 0.564 | 0.570 (+0.006, CI -0.004 to +0.018) | 0.573 (+0.011, CI -0.004 to +0.027) |
| Official split test IoU, 9-model ensemble | 0.687 (0.576-0.770) | not run | 0.686 (0.563-0.778) |
| Bolivia IoU | 0.597 | not run | 0.432 |

Paired official-split difference (S1 + all terrain minus S1): -0.001 (95% CI -0.034 to +0.027). Terrain as an input did not help.

## 03 Computer-vision tricks (LOEO, resnet34, 1 seed, one change at a time)
| Recipe | macro IoU | macro precision |
|---|---|---|
| base (BCE + Dice, 256 px crops, +-1 dB shift) | 0.575 | 0.697 |
| stronger contrast/shift augmentation | 0.567 | 0.707 |
| Tversky loss (false-positive weighted) | 0.573 | 0.770 |
| 512 px full-chip training | 0.566 | 0.714 |
| 512 px + chip-level water gate head | 0.574 | 0.701 |

Removing small connected components changed macro IoU by less than 0.001. The chip-level water gate was 81% accurate and did not improve results. The Tversky precision gain seen here turned out to be a shifted operating point: at matched recall it was less precise than the Dice control in the later run (see main README).

The "base" macro IoU here (0.575) and for S1 in the later runs (0.568) are the same recipe with different random seeds, which gives a feel for seed noise (about 0.007).
