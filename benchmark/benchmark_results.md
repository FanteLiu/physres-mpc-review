# Benchmark Results

Generated from `train_data.csv` using `model_benchmark.py`.

## Dataset And Evaluation Protocol

- Samples: 1500 total.
- Chronological split: 1050 training samples (70%) and 450 test samples (30%).
- One-step-ahead prediction is evaluated on 449 test targets.
- Warmup: first 10 test predictions are excluded from metrics.
- Raw pressure unit in `train_data.csv`: hPa. Paper figures convert pressure to kPa.
- State variables: chamber pressure `P`, Peltier temperature `Tp`, liquid temperature `Tl`.
- Models: PhysRes, Linear AR, MLP, ESN, LSTM.

## Main Takeaways

- Pressure: PhysRes obtains the lowest RMSE, 3.58 hPa (0.358 kPa). This is 13.3% lower than Linear AR and 40.2% lower than LSTM.
- Peltier temperature: LSTM is marginally lower in RMSE (0.167 deg C) than PhysRes (0.173 deg C), while PhysRes remains essentially comparable and far better than MLP/ESN.
- Liquid temperature: PhysRes obtains the lowest RMSE, 0.0224 deg C, 21.0% lower than LSTM.
- Linear AR is not a reliable baseline on this dataset: it diverges for `Tp` and performs poorly for `Tl`.
- PhysRes provides the best overall trade-off among the tested models: best pressure prediction, best liquid-temperature prediction, near-best Peltier-temperature prediction, compact model size, and fast closed-form training.

## Pressure Prediction Metrics

| Model | RMSE (hPa) | RMSE (kPa) | NRMSE | R2 | MAE (hPa) |
|---|---:|---:|---:|---:|---:|
| PhysRes | 3.5816 | 0.3582 | 0.0042 | 0.9989 | 2.5444 |
| Linear_AR | 4.1304 | 0.4130 | 0.0049 | 0.9985 | 2.9968 |
| MLP | 47.7917 | 4.7792 | 0.0563 | 0.7972 | 34.9822 |
| ESN | 140.4638 | 14.0464 | 0.1656 | -0.7516 | 98.4986 |
| LSTM | 5.9854 | 0.5985 | 0.0071 | 0.9968 | 4.5828 |

## Peltier Temperature Prediction Metrics

| Model | RMSE (deg C) | NRMSE | R2 | MAE (deg C) |
|---|---:|---:|---:|---:|
| PhysRes | 0.1734 | 0.0073 | 0.9977 | 0.0861 |
| Linear_AR | 122665.8239 | 5169.7448 | Diverged | 43838.5137 |
| MLP | 0.4781 | 0.0202 | 0.9822 | 0.3894 |
| ESN | 0.3929 | 0.0166 | 0.9880 | 0.2857 |
| LSTM | 0.1674 | 0.0071 | 0.9978 | 0.0872 |

## Liquid Temperature Prediction Metrics

| Model | RMSE (deg C) | NRMSE | R2 | MAE (deg C) |
|---|---:|---:|---:|---:|
| PhysRes | 0.0224 | 0.0011 | 0.9997 | 0.0177 |
| Linear_AR | 4.1563 | 0.2097 | -7.8858 | 3.1389 |
| MLP | 0.2501 | 0.0126 | 0.9678 | 0.1976 |
| ESN | 0.1046 | 0.0053 | 0.9944 | 0.0839 |
| LSTM | 0.0284 | 0.0014 | 0.9996 | 0.0224 |

## Training, Inference, And Model Complexity

| Model | Train time (s) | Inference time (ms) | Parameters | Hyperparameters |
|---|---:|---:|---:|---:|
| PhysRes | 0.0201 | 0.0122 | 1001 | 3 |
| Linear_AR | 0.0230 | 0.0431 | 10 | 1 |
| MLP | 0.4821 | 0.0476 | 2433 | 3 |
| ESN | 0.2193 | 0.1906 | 41001 | 4 |
| LSTM | 0.0389 | 0.0692 | 11051 | 2 |

## Figure Notes

- Fig. 4 plots pressure in kPa after removing the 10-sample warmup region.
- Fig. 5 plots `Tp` and `Tl` after removing the 10-sample warmup region.
- Linear AR is omitted from Fig. 5 because its `Tp` prediction diverges and would make the temperature axes unreadable; its numerical failure is retained in the metrics table.

## Files Produced

- MATLAB-ready prediction data: `paper_prediction_data.csv`.
- MATLAB plotting script: `generate_fig4_fig5_matlab.m`.
- Fig. 4 output: `../figures/fig4_pressure_prediction_new.png` and `.pdf`.
- Fig. 5 output: `../figures/fig5_temperature_prediction_new.png` and `.pdf`.
