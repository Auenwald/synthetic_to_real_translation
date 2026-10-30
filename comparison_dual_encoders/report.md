# Dual-encoder comparison

Candidate epochs: 1–30; primary temporal window: 6-30.
Seeds: [0, 1, 2]. Values: mean ± seed SD (ddof=1).

Target maxima and gaps below are retrospective, computed within each seed before averaging.

| View | Architecture | State | Target | Temporal mean | Temporal SD | Range | Target at source selection | Target maximum | Maximum − selected |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| HSV | Static | online | Cityscapes | 46.601 ± 0.234 | 0.938 ± 0.356 | 3.744 ± 1.077 | 45.612 ± 0.522 | 48.533 ± 0.405 | 2.921 ± 0.821 |
| HSV | Static | online | BDD100K | 40.657 ± 0.424 | 1.224 ± 0.338 | 4.850 ± 1.804 | 39.360 ± 0.894 | 43.020 ± 0.353 | 3.660 ± 1.154 |
| HSV | Static | ema | Cityscapes | 46.939 ± 0.203 | 0.626 ± 0.282 | 2.313 ± 1.076 | 46.057 ± 0.599 | 48.020 ± 0.253 | 1.963 ± 0.791 |
| HSV | Static | ema | BDD100K | 40.810 ± 0.637 | 0.948 ± 0.318 | 3.380 ± 1.466 | 39.616 ± 0.740 | 42.306 ± 0.539 | 2.689 ± 1.131 |
| HSV | Bidirectional | online | Cityscapes | 46.677 ± 0.552 | 1.279 ± 0.155 | 4.735 ± 1.262 | 44.818 ± 0.537 | 49.497 ± 0.920 | 4.679 ± 1.197 |
| HSV | Bidirectional | online | BDD100K | 40.723 ± 0.491 | 1.613 ± 0.268 | 5.801 ± 0.556 | 38.658 ± 1.173 | 43.724 ± 0.672 | 5.066 ± 1.646 |
| HSV | Bidirectional | ema | Cityscapes | 47.130 ± 0.534 | 1.167 ± 0.274 | 3.698 ± 0.784 | 45.453 ± 0.509 | 49.030 ± 0.613 | 3.577 ± 0.980 |
| HSV | Bidirectional | ema | BDD100K | 41.182 ± 0.456 | 1.383 ± 0.358 | 4.399 ± 0.810 | 39.233 ± 0.360 | 43.256 ± 0.721 | 4.023 ± 0.987 |
| FFT | Static | online | Cityscapes | 46.726 ± 0.498 | 1.200 ± 0.224 | 5.393 ± 0.631 | 46.294 ± 0.467 | 49.085 ± 0.212 | 2.790 ± 0.481 |
| FFT | Static | online | BDD100K | 41.543 ± 0.157 | 1.238 ± 0.090 | 5.621 ± 0.721 | 41.094 ± 0.726 | 44.471 ± 0.504 | 3.377 ± 1.002 |
| FFT | Static | ema | Cityscapes | 47.411 ± 0.533 | 0.916 ± 0.237 | 3.116 ± 0.691 | 46.383 ± 0.394 | 49.015 ± 0.417 | 2.633 ± 0.547 |
| FFT | Static | ema | BDD100K | 42.083 ± 0.247 | 0.839 ± 0.238 | 3.087 ± 0.657 | 41.001 ± 0.268 | 43.723 ± 0.586 | 2.722 ± 0.766 |
| FFT | Bidirectional | online | Cityscapes | 45.256 ± 0.161 | 1.477 ± 0.235 | 6.021 ± 0.954 | 43.497 ± 1.358 | 48.351 ± 0.675 | 4.854 ± 1.949 |
| FFT | Bidirectional | online | BDD100K | 40.240 ± 0.455 | 1.459 ± 0.394 | 5.571 ± 1.533 | 38.393 ± 0.786 | 43.301 ± 1.383 | 4.907 ± 2.165 |
| FFT | Bidirectional | ema | Cityscapes | 45.731 ± 0.272 | 1.317 ± 0.231 | 4.846 ± 0.641 | 44.172 ± 0.246 | 48.495 ± 0.804 | 4.323 ± 0.694 |
| FFT | Bidirectional | ema | BDD100K | 40.635 ± 0.404 | 1.149 ± 0.476 | 4.194 ± 1.742 | 39.158 ± 0.161 | 42.850 ± 1.295 | 3.692 ± 1.392 |

The complete CSV files also contain peak/selection epochs, final target scores,
source-score changes between these epochs, all temporal windows, and paired
bidirectional-minus-static differences. No significance tests are performed.

A positive source change from target peak to source selection describes endpoint
differences; it does not establish a monotonically increasing source trajectory.
Mean curves and seed SD bands do not measure within-run temporal variability.
