import csv
import re
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/data"
ASSETS = ROOT / "assets"
COLORS = ["#7C8C9B", "#2B729E", "#CC6846"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "ruact",
    }
)


def read(name):
    with (DATA / name).open() as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(
        ASSETS / (name + ".svg"),
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "RuACT"},
    )
    fig.savefig(
        ASSETS / (name + ".png"),
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "RuACT"},
    )
    svg = ASSETS / (name + ".svg")
    content = re.sub(r"<!--.*?-->", "", svg.read_text(), flags=re.S)
    svg.write_text("\n".join(line.rstrip() for line in content.splitlines()) + "\n")
    plt.close(fig)


def teacher():
    rows = read("teacher_recovery.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.2), layout="constrained")
    for method, color, marker in zip(
        ["GRPO", "RuACT-Teacher", "RuACT"], COLORS, ["o", "s", "D"]
    ):
        data = [r for r in rows if r["method"] == method]
        ax.plot(
            [int(r["step"]) for r in data],
            [float(r["recovery_percent"]) for r in data],
            marker=marker,
            markersize=4,
            lw=2,
            color=color,
            label=method,
        )
    ax.set(
        xlabel="Training step",
        ylabel="Recovered criteria (%)",
        xlim=(-3, 205),
        ylim=(0, 48),
        xticks=[0, 40, 80, 120, 160, 200],
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "teacher_recovery")


def critic():
    rows = read("critic_dispersion.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharey=True, layout="constrained")
    for ax, method in zip(axes, ["RuACT-Critic", "RuACT"]):
        data = [r for r in rows if r["method"] == method]
        for key, label, color, marker in [
            ("original_std", "Original reward", COLORS[0], "o"),
            ("combined_std", "With child rubrics", COLORS[2], "s"),
        ]:
            xs, ys = [], []
            for end in range(20, 201, 20):
                block = [r for r in data if end - 20 < int(r["step"]) <= end]
                n = sum(int(r["groups"]) for r in block)
                xs.append(end)
                ys.append(sum(float(r[key]) * int(r["groups"]) for r in block) / n)
            ax.plot(
                xs, ys, marker=marker, markersize=4, color=color, lw=1.8, label=label
            )
        ax.set(
            title=method,
            xlabel="Training step",
            xticks=[20, 60, 100, 140, 180],
            ylim=(0.10, 0.15),
        )
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Within-group reward standard deviation")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "critic_dispersion")


def ablations():
    rows = read("ablation_scores.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout="constrained")
    panels = ["Gate", "Filter", "Teacher target", "Critic target"]
    for ax, panel in zip(axes.flat, panels):
        data = [r for r in rows if r["comparison"] == panel]
        variants = list(dict.fromkeys(r["variant"] for r in data))
        for index, (variant, color) in enumerate(zip(variants, [COLORS[1], COLORS[2]])):
            ys = [
                float(
                    next(
                        r["score_percent"]
                        for r in data
                        if r["variant"] == variant and r["benchmark"] == b
                    )
                )
                for b in ["RaR-Medicine", "HealthBench"]
            ]
            xpos = np.arange(2) + (index - 0.5) * 0.32
            bars = ax.bar(xpos, ys, width=0.30, label=variant, color=color)
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in ys], padding=3, fontsize=9)
        qualifier = " (joint workflow)" if panel == "Filter" else ""
        ax.set(
            title=panel + qualifier + "\n" + data[0]["evaluator"],
            ylabel="Score (%)",
            ylim=(0, 70),
            xticks=[0, 1],
            xticklabels=["RaR-Medicine", "HealthBench"],
        )
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "ablations")


if __name__ == "__main__":
    teacher()
    critic()
    ablations()
