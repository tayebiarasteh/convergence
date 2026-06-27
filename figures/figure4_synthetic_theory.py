"""
figures/figure4_synthetic_theory.py
Created on June 27, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Figure4SyntheticTheory:
    THEORY_CSV = os.environ.get(
        "CONV_THEORY_CSV", "/PATH/theory_synthetic.csv"
    )
    OUT_PDF = "/PATH/figure4_synthetic_theory.pdf"

    PAIR_COLOR = {"ssl_vs_ssl": "#2166AC", "sup_vs_sup": "#E08214", "sup_vs_ssl": "#7A7A7A"}
    PAIR_LABEL = {"ssl_vs_ssl": "SSL pair", "sup_vs_sup": "Supervised pair",
                  "sup_vs_ssl": "Mixed pair"}
    PAIR_ORDER = ["ssl_vs_ssl", "sup_vs_ssl", "sup_vs_sup"]        # for curves
    STRIP_ORDER = ["ssl_vs_ssl", "sup_vs_ssl", "sup_vs_sup"]       # for strips
    GAP_COLOR = "#333333"

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        self._set_rcparams()
        self._load_data()
        self.fig = None

    @staticmethod
    def _set_rcparams():
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 18.0,
            "axes.labelsize": 18.0,
            "axes.titlesize": 19.5,
            "xtick.labelsize": 15.0,
            "ytick.labelsize": 15.0,
            "legend.fontsize": 16.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.15,
            "xtick.major.width": 1.05,
            "ytick.major.width": 1.05,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

    @staticmethod
    def _clean(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    @staticmethod
    def _panel_label(ax, letter, title, x=-0.115, y=1.04, dx=0.060):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=25, fontweight="bold",
                ha="left", va="bottom")
        ax.text(x + dx, y, title, transform=ax.transAxes, fontsize=18.5,
                ha="left", va="bottom")

    @staticmethod
    def _resolve(name, env=None):
        if env and os.environ.get(env):
            return os.environ[env]
        for d in ("/PATH/project_data", "/PATH"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return os.path.join("/PATH", name)

    def _load_data(self):
        df = pd.read_csv(self._resolve("theory_synthetic.csv", "CONV_THEORY_CSV"))
        self.df = df[df["experiment"] == "E8"].copy()

    def _mknn_curve(self):
        s = self.df[self.df["metric"] == "mknn_over_seeds"]
        out = {}
        for pt in self.PAIR_ORDER:
            d = s[s["pair_type"] == pt].sort_values("alpha")
            out[pt] = (d["alpha"].values, d["value_mean"].values,
                       d["value_ci_low"].values, d["value_ci_high"].values)
        return out

    def _gap_curve(self):
        d = self.df[self.df["metric"] == "alignment_gap_sup_minus_ssl"].sort_values("alpha")
        return (d["alpha"].values, d["value_mean"].values,
                d["value_ci_low"].values, d["value_ci_high"].values,
                float(d["p_fdr"].iloc[0]))

    def _per_seed(self, metric):
        s = self.df[(self.df["sub_analysis"] == "synthetic_alignment") &
                    (self.df["metric"] == metric)]
        return {pt: s[s["pair_type"] == pt]["value_mean"].values for pt in self.STRIP_ORDER}

    def _proc_curve(self):
        s = self.df[(self.df["sub_analysis"] == "synthetic_alignment") &
                    (self.df["metric"] == "procrustes")]
        out = {}
        for pt in self.PAIR_ORDER:
            d = s[s["pair_type"] == pt]
            g = d.groupby("alpha")["value_mean"].agg(["mean", "std"]).reset_index()
            out[pt] = (g["alpha"].values, g["mean"].values, g["std"].fillna(0).values)
        return out

    def build(self):
        self.fig = plt.figure(figsize=(16.5, 18.4), facecolor="white")

        gs_leg = self.fig.add_gridspec(1, 1, left=0.06, right=0.98, top=0.995, bottom=0.958)
        gs_r1 = self.fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.905, bottom=0.660,
                                      wspace=0.30)
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.585, bottom=0.345,
                                      wspace=0.32)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.270, bottom=0.050,
                                      wspace=0.32)

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_metric_agreement(self.fig.add_subplot(gs_r1[0, 0]), "a", "Alignment metrics agree")
        self._panel_mknn(self.fig.add_subplot(gs_r1[0, 1]), "b", "mKNN vs label informativeness")
        self._panel_gap(self.fig.add_subplot(gs_r2[0, 0]), "c", "Supervised minus self-supervised gap")
        self._panel_strip(self.fig.add_subplot(gs_r2[0, 1]), "d", "Per-seed mKNN by pairing",
                          "mknn", "mKNN (%)")
        self._panel_proc(self.fig.add_subplot(gs_r3[0, 0]), "e", "Procrustes vs label informativeness")
        self._panel_strip(self.fig.add_subplot(gs_r3[0, 1]), "f", "Per-seed Procrustes by pairing",
                          "procrustes", "Procrustes disparity", lower_better=True)
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        handles = [Line2D([0], [0], marker="o", lw=0, markersize=15,
                          mfc=self.PAIR_COLOR[p], mec="white", mew=0.8,
                          label=self.PAIR_LABEL[p]) for p in self.PAIR_ORDER]
        ax.text(0.005, 0.5, "Encoder pairing", transform=ax.transAxes,
                fontsize=16.5, fontweight="normal", ha="left", va="center")
        leg = ax.legend(handles, [h.get_label() for h in handles], ncol=3,
                        frameon=False, loc="center left", bbox_to_anchor=(0.165, 0.5),
                        columnspacing=1.8, handletextpad=0.5,
                        prop={"size": 16.0, "weight": "normal"})
        ax.add_artist(leg)

    def _panel_metric_agreement(self, ax, letter, title):
        self._clean(ax)
        s = self.df[self.df["sub_analysis"] == "synthetic_alignment"]
        mk = (s[s["metric"] == "mknn"][["alpha", "seed", "pair_type", "value_mean"]]
              .rename(columns={"value_mean": "mknn"}))
        pr = (s[s["metric"] == "procrustes"][["alpha", "seed", "pair_type", "value_mean"]]
              .rename(columns={"value_mean": "proc"}))
        m = mk.merge(pr, on=["alpha", "seed", "pair_type"])
        for pt in self.STRIP_ORDER:
            d = m[m["pair_type"] == pt]
            ax.scatter(d["mknn"], d["proc"], s=58, color=self.PAIR_COLOR[pt],
                       alpha=0.6, edgecolor="white", linewidth=0.6, zorder=3)
        r = float(np.corrcoef(m["mknn"], m["proc"])[0, 1])
        ax.text(0.04, 0.06, f"Pearson r = {r:.2f}\nn = {len(m)} runs",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=14.0, color="#222222")
        ax.text(0.96, 0.95, "low disparity =\nmore aligned", transform=ax.transAxes,
                ha="right", va="top", fontsize=12.5, color="#777777")
        ax.set_xlabel("mKNN (%)")
        ax.set_ylabel("Procrustes disparity")
        ax.set_xlim(0, float(m["mknn"].max()) * 1.12)
        ax.set_ylim(0, float(m["proc"].max()) * 1.22)
        self._panel_label(ax, letter, title, x=-0.165, y=1.04, dx=0.066)

    def _panel_mknn(self, ax, letter, title):
        self._clean(ax)
        cur = self._mknn_curve()
        for pt in self.PAIR_ORDER:
            a, m, lo, hi = cur[pt]
            c = self.PAIR_COLOR[pt]
            ax.fill_between(a, lo, hi, color=c, alpha=0.16, zorder=1)
            ax.plot(a, m, "-o", color=c, lw=2.4, ms=8, mfc=c, mec="white", mew=0.9, zorder=3)
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(r"Label informativeness $\alpha$")
        ax.set_ylabel("mKNN (%)")
        ax.set_ylim(0, 12)
        ax.set_xlim(0.04, 1.06)
        self._panel_label(ax, letter, title, x=-0.155, y=1.04, dx=0.066)

    def _panel_gap(self, ax, letter, title):
        self._clean(ax)
        a, m, lo, hi, p = self._gap_curve()
        ax.axhline(0, color="#999999", ls=(0, (4, 3)), lw=1.5, zorder=1)
        ax.axhspan(-8, 0, color="#E08214", alpha=0.05, zorder=0)
        ax.fill_between(a, lo, hi, color=self.GAP_COLOR, alpha=0.16, zorder=1)
        ax.plot(a, m, "-o", color=self.GAP_COLOR, lw=2.4, ms=8, mfc=self.GAP_COLOR,
                mec="white", mew=0.9, zorder=3)
        ptxt = "all p < 0.001" if p < 0.001 else f"all p = {p:.3f}"
        ax.text(0.97, 0.06, ptxt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=14.5, color="#222222")
        ax.text(0.04, 0.10, "supervised less aligned", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=13.5, color="#9a6212")
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(r"Label informativeness $\alpha$")
        ax.set_ylabel("Supervised $-$ self-supervised\nmKNN gap (%)")
        ax.set_ylim(-8, 1.0)
        ax.set_xlim(0.04, 1.06)
        self._panel_label(ax, letter, title, x=-0.185, y=1.04, dx=0.072)

    def _panel_strip(self, ax, letter, title, metric, ylabel, lower_better=False):
        self._clean(ax)
        data = self._per_seed(metric)
        rng = np.random.default_rng(5)
        xs = np.arange(len(self.STRIP_ORDER))
        for x, pt in zip(xs, self.STRIP_ORDER):
            v = data[pt]
            c = self.PAIR_COLOR[pt]
            ax.scatter(x + rng.uniform(-0.16, 0.16, len(v)), v, s=46, color=c,
                       alpha=0.55, edgecolor="white", linewidth=0.5, zorder=3)
            mu = float(np.mean(v))
            ax.plot([x - 0.28, x + 0.28], [mu, mu], color="#1a1a1a", lw=2.6, zorder=5)
            ax.text(x + 0.32, mu, f"{mu:.1f}" if metric == "mknn" else f"{mu:.2f}",
                    va="center", ha="left", fontsize=13.5, color="#1a1a1a")
        ax.set_xticks(xs)
        ax.set_xticklabels([self.PAIR_LABEL[p] for p in self.STRIP_ORDER], fontsize=14.5)
        for tl, p in zip(ax.get_xticklabels(), self.STRIP_ORDER):
            tl.set_color(self.PAIR_COLOR[p])
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, len(xs) - 0.4)
        if lower_better:
            allv = np.concatenate([data[p] for p in self.STRIP_ORDER])
            ax.set_ylim(0, allv.max() * 1.28)
            ax.text(0.97, 0.95, "lower = more aligned", transform=ax.transAxes,
                    ha="right", va="top", fontsize=13.0, color="#555555")
        else:
            allv = np.concatenate([data[p] for p in self.STRIP_ORDER])
            ax.set_ylim(0, allv.max() * 1.15)
        self._panel_label(ax, letter, title, x=-0.165, y=1.04, dx=0.066)

    def _panel_proc(self, ax, letter, title):
        self._clean(ax)
        cur = self._proc_curve()
        for pt in self.PAIR_ORDER:
            a, m, sd = cur[pt]
            c = self.PAIR_COLOR[pt]
            ax.fill_between(a, m - sd, m + sd, color=c, alpha=0.16, zorder=1)
            ax.plot(a, m, "-o", color=c, lw=2.4, ms=8, mfc=c, mec="white", mew=0.9, zorder=3)
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(r"Label informativeness $\alpha$")
        ax.set_ylabel("Procrustes disparity")
        ax.text(0.97, 0.95, "lower = more aligned", transform=ax.transAxes,
                ha="right", va="top", fontsize=13.0, color="#555555")
        ax.set_ylim(0, 1.35)
        ax.set_xlim(0.04, 1.06)
        self._panel_label(ax, letter, title, x=-0.165, y=1.04, dx=0.066)

    def save(self):
        self.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.out_pdf, format="pdf", facecolor="white", bbox_inches="tight")
        return self


if __name__ == "__main__":
    Figure4SyntheticTheory().build().save()
