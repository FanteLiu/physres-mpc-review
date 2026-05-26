#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export benchmark results and MATLAB-ready prediction data for paper figures."""

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "benchmark_data.npz"
SUMMARY_FILE = ROOT / "benchmark_summary.csv"
PREDICTION_CSV = ROOT / "paper_prediction_data.csv"
RESULTS_MD = ROOT / "benchmark_results.md"
WARMUP = 10


def fmt(value, digits=4):
    if value is None:
        return "Diverged"
    try:
        if math.isnan(value) or math.isinf(value):
            return "Diverged"
    except TypeError:
        return str(value)
    return f"{value:.{digits}f}"


def load_summary_rows():
    rows = []
    with SUMMARY_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(row, key):
    value = row[key]
    if value == "":
        return float("nan")
    return float(value)


def write_prediction_csv(data):
    predictions = data["predictions"].item()
    y_p = data["y_P_test"]
    y_tp = data["y_Tp_test"]
    y_tl = data["y_Tl_test"]
    n = len(y_p)
    models = ["PhysRes", "Linear_AR", "MLP", "ESN", "LSTM"]

    headers = [
        "sample",
        "time_s",
        "P_actual_kPa",
        "Tp_actual_C",
        "Tl_actual_C",
    ]
    for model in models:
        headers.extend([
            f"P_{model}_kPa",
            f"Tp_{model}_C",
            f"Tl_{model}_C",
        ])

    with PREDICTION_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(n):
            row = [
                i,
                i,
                y_p[i] / 10.0,
                y_tp[i],
                y_tl[i],
            ]
            for model in models:
                row.extend([
                    predictions["P"][model][i] / 10.0,
                    predictions["Tp"][model][i],
                    predictions["Tl"][model][i],
                ])
            writer.writerow(row)


def metric_table(rows, variable, pressure=False):
    selected = [r for r in rows if r["Variable"] == variable]
    lines = []
    if pressure:
        lines.append("| Model | RMSE (hPa) | RMSE (kPa) | NRMSE | R2 | MAE (hPa) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in selected:
            rmse_hpa = to_float(r, "RMSE")
            mae_hpa = to_float(r, "MAE")
            lines.append(
                f"| {r['Model']} | {fmt(rmse_hpa)} | {fmt(rmse_hpa / 10.0)} | "
                f"{fmt(to_float(r, 'NRMSE'))} | {fmt(to_float(r, 'R2'))} | {fmt(mae_hpa)} |"
            )
    else:
        lines.append("| Model | RMSE (deg C) | NRMSE | R2 | MAE (deg C) |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in selected:
            lines.append(
                f"| {r['Model']} | {fmt(to_float(r, 'RMSE'))} | "
                f"{fmt(to_float(r, 'NRMSE'))} | {fmt(to_float(r, 'R2'))} | {fmt(to_float(r, 'MAE'))} |"
            )
    return "\n".join(lines)


def write_markdown(data, rows):
    train_times = data["train_times"].item()
    infer_times = data["infer_times"].item()
    param_counts = data["param_counts"].item()
    hyperparam_counts = data["hyperparam_counts"].item()
    models = ["PhysRes", "Linear_AR", "MLP", "ESN", "LSTM"]

    p_rows = {r["Model"]: r for r in rows if r["Variable"] == "P"}
    tp_rows = {r["Model"]: r for r in rows if r["Variable"] == "Tp"}
    tl_rows = {r["Model"]: r for r in rows if r["Variable"] == "Tl"}

    phys_p = to_float(p_rows["PhysRes"], "RMSE")
    linear_p = to_float(p_rows["Linear_AR"], "RMSE")
    lstm_p = to_float(p_rows["LSTM"], "RMSE")
    phys_tl = to_float(tl_rows["PhysRes"], "RMSE")
    lstm_tl = to_float(tl_rows["LSTM"], "RMSE")
    phys_tp = to_float(tp_rows["PhysRes"], "RMSE")
    lstm_tp = to_float(tp_rows["LSTM"], "RMSE")

    lines = [
        "# Benchmark Results",
        "",
        "Generated from `train_data.csv` using `model_benchmark.py`.",
        "",
        "## Dataset And Evaluation Protocol",
        "",
        "- Samples: 1500 total.",
        "- Chronological split: 1050 training samples (70%) and 450 test samples (30%).",
        "- One-step-ahead prediction is evaluated on 449 test targets.",
        f"- Warmup: first {WARMUP} test predictions are excluded from metrics.",
        "- Raw pressure unit in `train_data.csv`: hPa. Paper figures convert pressure to kPa.",
        "- State variables: chamber pressure `P`, Peltier temperature `Tp`, liquid temperature `Tl`.",
        "- Models: PhysRes, Linear AR, MLP, ESN, LSTM.",
        "",
        "## Main Takeaways",
        "",
        f"- Pressure: PhysRes obtains the lowest RMSE, {phys_p:.2f} hPa ({phys_p / 10.0:.3f} kPa). This is {(linear_p - phys_p) / linear_p * 100.0:.1f}% lower than Linear AR and {(lstm_p - phys_p) / lstm_p * 100.0:.1f}% lower than LSTM.",
        f"- Peltier temperature: LSTM is marginally lower in RMSE ({lstm_tp:.3f} deg C) than PhysRes ({phys_tp:.3f} deg C), while PhysRes remains essentially comparable and far better than MLP/ESN.",
        f"- Liquid temperature: PhysRes obtains the lowest RMSE, {phys_tl:.4f} deg C, {(lstm_tl - phys_tl) / lstm_tl * 100.0:.1f}% lower than LSTM.",
        "- Linear AR is not a reliable baseline on this dataset: it diverges for `Tp` and performs poorly for `Tl`.",
        "- PhysRes provides the best overall trade-off among the tested models: best pressure prediction, best liquid-temperature prediction, near-best Peltier-temperature prediction, compact model size, and fast closed-form training.",
        "",
        "## Pressure Prediction Metrics",
        "",
        metric_table(rows, "P", pressure=True),
        "",
        "## Peltier Temperature Prediction Metrics",
        "",
        metric_table(rows, "Tp"),
        "",
        "## Liquid Temperature Prediction Metrics",
        "",
        metric_table(rows, "Tl"),
        "",
        "## Training, Inference, And Model Complexity",
        "",
        "| Model | Train time (s) | Inference time (ms) | Parameters | Hyperparameters |",
        "|---|---:|---:|---:|---:|",
    ]

    for model in models:
        lines.append(
            f"| {model} | {train_times[model]:.4f} | {infer_times[model]:.4f} | "
            f"{param_counts[model]} | {hyperparam_counts[model]} |"
        )

    lines.extend([
        "",
        "## Figure Notes",
        "",
        "- Fig. 4 plots pressure in kPa after removing the 10-sample warmup region.",
        "- Fig. 5 plots `Tp` and `Tl` after removing the 10-sample warmup region.",
        "- Linear AR is omitted from Fig. 5 because its `Tp` prediction diverges and would make the temperature axes unreadable; its numerical failure is retained in the metrics table.",
        "",
        "## Files Produced",
        "",
        f"- MATLAB-ready prediction data: `{PREDICTION_CSV.name}`.",
        "- MATLAB plotting script: `generate_fig4_fig5_matlab.m`.",
        "- Fig. 4 output: `../figures/fig4_pressure_prediction_new.png` and `.pdf`.",
        "- Fig. 5 output: `../figures/fig5_temperature_prediction_new.png` and `.pdf`.",
    ])

    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    data = np.load(DATA_FILE, allow_pickle=True)
    rows = load_summary_rows()
    write_prediction_csv(data)
    write_markdown(data, rows)
    print(f"Wrote {PREDICTION_CSV}")
    print(f"Wrote {RESULTS_MD}")


if __name__ == "__main__":
    main()
