# DP-SGD learning-rate tuning (2026-10-22)

These were exploratory runs made before the full experiment. Their checkpoints and metric files were deleted, and the numbers below were copied from the console output of `python -m src.train_private`.

Setting: seed 0, 10,000 members, 2,000 validation images, 30 epochs, expected batch 1,024, momentum 0.9, clipping norm 1.0, δ = 1e-5, target ε = 8 (achieved 7.999, noise multiplier 1.360).

| DP learning rate | final train loss | final validation accuracy |
|---|---|---|
| 1.0 | 2.1493 | 0.4600 |
| 0.5 | 2.1031 | 0.5035 |
| 0.25 | 1.6737 | 0.5385 |
| 0.1 | 1.4401 | 0.5355 |

The standard-model setting (lr 0.05, batch 128, 30 epochs) was checked once in the same session: validation accuracy 0.6590 in the final epoch.

Full run v1 (`full_v1_*`) used lr 0.25 for every ε. At ε = 1 (noise multiplier 7.19) training diverged: train loss rose from 2.29 to 4.13 and test accuracy was 0.17 on average. A single ε = 1 run with lr 0.1 (seed 0) reached validation accuracy 0.3355 and test accuracy 0.3178 without diverging. Because lr 0.1 and 0.25 were within 0.003 validation accuracy at ε = 8, lr 0.1 was used for all ε in the final run (v2).

None of these tuning runs is covered by the privacy accounting (see the IA-2 limitations).
