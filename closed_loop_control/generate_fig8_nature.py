from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "closed_loop_control" / "control_data.mat"
OUT_DIR = ROOT / "figures"
OUT_BASENAME = OUT_DIR / "fig8_closed_loop_nature"


def load_control_data():
    mat = loadmat(DATA_PATH)
    return {
        "time": mat["t"].ravel().astype(float),
        "On-Off": mat["P_bang"].ravel().astype(float),
        "PID": mat["P_pid"].ravel().astype(float),
        "PhysRes-MPC": mat["P_mpc"].ravel().astype(float),
    }


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.3,
            "axes.labelsize": 6.6,
            "axes.titlesize": 6.3,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.55,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def make_figure():
    data = load_control_data()
    t = data["time"]
    target = 130.0

    colors = {
        "PhysRes-MPC": "#C8403D",
        "PID": "#2B6F8E",
        "On-Off": "#7A7F87",
    }
    order = ["PhysRes-MPC", "PID", "On-Off"]

    metrics = {
        "MAE\n(kPa)": {"PhysRes-MPC": 2.25, "PID": 2.75, "On-Off": 3.50},
        "Overshoot\n(%)": {"PhysRes-MPC": 18.0, "PID": 20.8, "On-Off": 21.8},
        "Steady-state\nerror (%)": {"PhysRes-MPC": 0.74, "PID": 1.80, "On-Off": 2.01},
    }

    fig = plt.figure(figsize=(3.5, 3.15), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[2.5, 1.0],
        hspace=0.50,
        wspace=0.55,
        left=0.13,
        right=0.985,
        top=0.93,
        bottom=0.16,
    )

    ax = fig.add_subplot(gs[0, :])
    for name in order:
        lw = 0.85 if name == "PhysRes-MPC" else 0.70
        z = 4 if name == "PhysRes-MPC" else 3
        ax.plot(t, data[name], color=colors[name], lw=lw, label=name, zorder=z)

    ax.axhline(target, color="#202020", lw=0.50, ls=(0, (3, 2)), zorder=1)

    ax.set_xlim(0, 400)
    ax.set_ylim(96, 139)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_yticks([100, 110, 120, 130, 140])
    ax.set_xlabel("Time (s)", labelpad=1.5)
    ax.set_ylabel("Pressure (kPa)", labelpad=2)
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        handlelength=1.8,
        columnspacing=1.4,
    )
    ax.text(-0.105, 1.06, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=7.2)

    for col, (metric_name, values) in enumerate(metrics.items()):
        mx = fig.add_subplot(gs[1, col])
        y_positions = np.arange(len(order))[::-1]
        vals = [values[name] for name in order]
        for yi, name, val in zip(y_positions, order, vals):
            mx.hlines(yi, 0, val, color=colors[name], lw=0.9, alpha=0.58)
            mx.plot(val, yi, "o", color=colors[name], ms=3.0, zorder=3)
            mx.text(
                val,
                yi + 0.22,
                f"{val:g}",
                ha="center",
                va="bottom",
                fontsize=5.2,
                color=colors[name],
            )

        xmax = max(vals) * 1.24
        mx.set_xlim(0, xmax)
        mx.set_ylim(-0.55, 2.55)
        mx.set_title(metric_name, pad=2.0)
        mx.set_yticks(y_positions)
        if col == 0:
            mx.set_yticklabels(order)
        else:
            mx.set_yticklabels([])
        mx.tick_params(axis="y", length=0, pad=1.5)
        mx.set_xticks([0, max(vals)])
        mx.set_xticklabels(["0", f"{max(vals):g}"])
        mx.spines["left"].set_visible(False)
        mx.grid(axis="x", color="#E2E2E2", lw=0.45)
        mx.set_axisbelow(True)
        if col == 0:
            mx.text(-0.44, 1.18, "(b)", transform=mx.transAxes, fontweight="bold", fontsize=7.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".png": {"dpi": 600},
        ".pdf": {},
        ".svg": {},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(OUT_BASENAME.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    configure_matplotlib()
    make_figure()
