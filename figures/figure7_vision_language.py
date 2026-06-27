"""
figures/figure7_vision_language.py
Created on June 27, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


class Figure7VisionLanguage:
    OUT_PDF = "/PATH/figure7_vision_language.pdf"

    CKNNA = "#1C6E68"   # CKNNA metric
    MKNN = "#E08214"    # mKNN metric
    G_IMG = "#117733"   # image-image within-modality (contrast)
    G_FLOOR = "#9E9E9E"  # random-init floor (contrast)
    G_TXT = "#762A83"   # image-text cross-modal (contrast)
    SEQ = LinearSegmentedColormap.from_list(
        "teal_seq", ["#F2FAF9", "#BCE3DF", "#7FC6BF", "#3E9A92", "#1C6E68", "#0C3B38"])

    POOLS = [1000, 5000, 20000]
    ENC_NAME = {
        "rad_dino": "RAD-DINO", "txrv_densenet": "TXRV", "dinov2_large": "DINOv2",
        "dinov3_l": "DINOv3", "clip_vitl14": "CLIP", "siglip2_large": "SigLIP2",
        "biomedclip_image": "BiomedCLIP", "medgemma_vision": "MedGemma",
        "llava_med_vision": "LLaVA-Med", "llava_onevision_vision": "LLaVA-OV",
        "uni": "UNI", "uni2": "UNI2-h", "virchow": "Virchow", "virchow2": "Virchow2",
        "phikon_v2": "Phikon-v2", "prov_gigapath": "GigaPath", "conch_image": "CONCH",
        "retfound": "RETFound",
    }
    TXT_NAME = {
        "biomedclip_text": "BiomedCLIP", "conch_text": "CONCH", "medcpt_article": "MedCPT-art",
        "medcpt_query": "MedCPT-qry", "medgemma_27b_text": "MedGemma-27B",
        "pubmedbert": "PubMedBERT", "sapbert": "SapBERT",
    }

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        self._set_rcparams()
        self._load_data()
        self.fig = None

    @staticmethod
    def _set_rcparams():
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 18.0, "axes.labelsize": 17.5,
            "axes.titlesize": 19.0, "xtick.labelsize": 14.5, "ytick.labelsize": 14.5,
            "legend.fontsize": 15.5, "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": 1.15, "xtick.major.width": 1.05, "ytick.major.width": 1.05,
            "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        })

    @staticmethod
    def _clean(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    @staticmethod
    def _panel_label(ax, letter, title, x=-0.16, y=1.04, dx=0.090):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=24, fontweight="bold",
                ha="left", va="bottom")
        ax.text(x + dx, y, title, transform=ax.transAxes, fontsize=18.0, ha="left", va="bottom")

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
        df = pd.read_csv(self._resolve("convergence_alignment.csv"))
        self.e2 = df[df["experiment"] == "E2"].copy()
        e1 = df[df["experiment"] == "E1"]
        self.within = e1[(e1["sub_analysis"] == "within_cxr") & (e1["n"] == 20000)]["value_mean"].dropna()
        self.floor = e1[(e1["sub_analysis"] == "floor_cxr") & (e1["n"] == 20000)]["value_mean"].dropna()

    def _pool_traces(self, metric):
        d = self.e2[(self.e2["metric"] == metric) & (self.e2["k"] == 10)]
        piv = d.pivot_table(index=["unit_a", "unit_b"], columns="n", values="value_mean")
        return piv[[p for p in self.POOLS if p in piv.columns]]

    def _dist_at(self, metric, pool):
        return self.e2[(self.e2["metric"] == metric) & (self.e2["k"] == 10) &
                       (self.e2["n"] == pool)]["value_mean"].dropna().values

    def build(self):
        self.fig = plt.figure(figsize=(18.0, 18.6), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.06, right=0.98, top=0.992, bottom=0.952)
        gs_r1 = self.fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.905, bottom=0.655,
                                      wspace=0.27)
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.575, bottom=0.325,
                                      width_ratios=[1.0, 1.05], wspace=0.27)
        gs_r3 = self.fig.add_gridspec(1, 1, left=0.110, right=0.940, top=0.250, bottom=0.075)

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_pool(self.fig.add_subplot(gs_r1[0, 0]), "a", "mKNN decays with pool size",
                         "mknn", self.MKNN, floor=False)
        self._panel_pool(self.fig.add_subplot(gs_r1[0, 1]), "b", "CKNNA decays with pool size",
                         "cknna", self.CKNNA, floor=True)
        self._panel_dist(self.fig.add_subplot(gs_r2[0, 0]), "c", "Every pair sits at floor")
        self._panel_contrast(self.fig.add_subplot(gs_r2[0, 1]), "d", "Cross-modal vs within-modality")
        self._panel_heatmap(self.fig.add_subplot(gs_r3[0, 0]), "e", "Image-text alignment landscape")
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        items = [("CKNNA", self.CKNNA), ("mKNN", self.MKNN)]
        handles = [Line2D([0], [0], marker="s", lw=0, markersize=16, mfc=c, mec="white",
                          mew=0.8, label=lab) for lab, c in items]
        ax.text(0.30, 0.5, "Alignment metric", transform=ax.transAxes, fontsize=16.5,
                fontweight="normal", ha="right", va="center")
        leg = ax.legend(handles, [h.get_label() for h in handles], ncol=2, frameon=False,
                        loc="center left", bbox_to_anchor=(0.33, 0.5), columnspacing=2.0,
                        handletextpad=0.5, prop={"size": 16.5, "weight": "normal"})
        ax.add_artist(leg)

    def _panel_pool(self, ax, letter, title, metric, color, floor):
        self._clean(ax)
        piv = self._pool_traces(metric)
        x = np.array(piv.columns, dtype=float)
        for _, row in piv.iterrows():
            ax.plot(x, row.values, color=color, alpha=0.07, lw=1.0, zorder=2)
        med = piv.median(axis=0).values
        q1 = piv.quantile(0.25, axis=0).values
        q3 = piv.quantile(0.75, axis=0).values
        ax.fill_between(x, q1, q3, color=color, alpha=0.18, zorder=3)
        ax.plot(x, med, color=color, lw=3.0, marker="o", ms=9, zorder=5)
        for xi, mi in zip(x, med):
            ax.annotate(f"{mi:.1f}", xy=(xi, mi), xytext=(0, 11), textcoords="offset points",
                        ha="center", fontsize=13.0, color=color)
        if floor:
            fl = float(self.floor.mean())
            ax.axhline(fl, color="#777777", lw=1.5, ls=(0, (4, 3)), zorder=1)
            ax.text(x[-1], fl + 0.18, f"random-init floor {fl:.1f}", color="#777777",
                    fontsize=12.0, ha="right", va="bottom")
        ax.set_xscale("log")
        ax.set_xticks(self.POOLS)
        ax.set_xticklabels([f"{p:,}" for p in self.POOLS])
        ax.set_xlabel("Evaluation-pool size (cases)")
        ax.set_ylabel(f"{'mKNN' if metric=='mknn' else 'CKNNA'} alignment (%)")
        ax.set_ylim(0, max(piv.values.max(), (self.floor.mean() if floor else 0)) * 1.18)
        self._panel_label(ax, letter, title, x=-0.165, y=1.04, dx=0.090)

    def _panel_dist(self, ax, letter, title):
        self._clean(ax)
        groups = [("mKNN", self._dist_at("mknn", 1000), self.MKNN),
                  ("CKNNA", self._dist_at("cknna", 1000), self.CKNNA)]
        rng = np.random.default_rng(5)
        for i, (lab, v, c) in enumerate(groups):
            parts = ax.violinplot(v, positions=[i], widths=0.7, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(c); b.set_alpha(0.30); b.set_edgecolor(c); b.set_linewidth(1.4)
            ax.scatter(i + rng.uniform(-0.10, 0.10, len(v)), v, s=18, color=c, alpha=0.45,
                       edgecolor="none", zorder=4)
            ax.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color="#1a1a1a", lw=2.4, zorder=5)
            ax.text(i, v.max() + 0.35, f"median {np.median(v):.1f}\nmax {v.max():.1f}",
                    ha="center", va="bottom", fontsize=12.5, color="#1a1a1a")
        fl = float(self.floor.mean())
        ax.plot([0.6, 1.4], [fl, fl], color="#777777", lw=1.5, ls=(0, (4, 3)), zorder=1)
        ax.text(1.42, fl, f"floor {fl:.1f}", color="#777777", fontsize=12.0, ha="left", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["mKNN", "CKNNA"], fontsize=15.0)
        ax.set_ylabel("Alignment at 1,000 cases (%)")
        ax.set_ylim(0, 9.2)
        self._panel_label(ax, letter, title, x=-0.175, y=1.04, dx=0.090)

    def _panel_contrast(self, ax, letter, title):
        self._clean(ax)
        data = [("Image\u2013image\n(within-modality)", self.within.values, self.G_IMG),
                ("Random\nfloor", self.floor.values, self.G_FLOOR),
                ("Image\u2013text\n(cross-modal)", self._dist_at("cknna", 20000), self.G_TXT)]
        rng = np.random.default_rng(8)
        for i, (lab, v, c) in enumerate(data):
            bp = ax.boxplot(v, positions=[i], widths=0.55, patch_artist=True, showfliers=False,
                            medianprops=dict(color="#1a1a1a", lw=2.0),
                            whiskerprops=dict(color="#666666", lw=1.4),
                            capprops=dict(color="#666666", lw=1.4))
            bp["boxes"][0].set(facecolor=c, alpha=0.55, edgecolor=c, linewidth=1.6)
            ax.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=14, color=c, alpha=0.30,
                       edgecolor="none", zorder=4)
            ax.text(i, np.median(v), f"  {np.median(v):.1f}", ha="left", va="center",
                    fontsize=13.0, color="#1a1a1a")
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([f"{lab}\n($n=${len(v)})" for lab, v, _ in data], fontsize=12.5)
        ax.set_ylabel("CKNNA at 20,000 cases (%)")
        ax.set_ylim(0, max(self.within.max(), 1) * 1.15)
        self._panel_label(ax, letter, title, x=-0.175, y=1.04, dx=0.090)

    def _panel_heatmap(self, ax, letter, title):
        self._clean(ax)
        d = self.e2[(self.e2["metric"] == "cknna") & (self.e2["k"] == 10) & (self.e2["n"] == 1000)]
        mat = d.pivot_table(index="unit_b", columns="unit_a", values="value_mean")
        col_order = mat.mean(axis=0).sort_values(ascending=False).index.tolist()
        row_order = mat.mean(axis=1).sort_values(ascending=False).index.tolist()
        M = mat.loc[row_order, col_order].values.astype(float)
        norm = Normalize(vmin=0, vmax=float(np.nanmax(M)))
        im = ax.imshow(M, cmap=self.SEQ, norm=norm, aspect="auto")
        nr, nc = M.shape
        for i in range(nr):
            for j in range(nc):
                shade = self.SEQ(norm(M[i, j]))
                lum = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        color="white" if lum < 0.55 else "#1a1a1a", fontsize=10.5)
        ax.set_xticks(range(nc))
        ax.set_xticklabels([self.ENC_NAME[c] for c in col_order], rotation=90, fontsize=11.5)
        ax.set_yticks(range(nr))
        ax.set_yticklabels([self.TXT_NAME[r] for r in row_order], fontsize=12.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Image encoder", fontsize=15.5)
        ax.set_ylabel("Text encoder", fontsize=15.5)
        cb = self.fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
        cb.set_label("CKNNA at 1,000 cases (%)", fontsize=13)
        cb.ax.tick_params(labelsize=11)
        self._panel_label(ax, letter, title, x=-0.085, y=1.05, dx=0.045)

    def save(self):
        self.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.out_pdf, format="pdf", facecolor="white", bbox_inches="tight")
        return self


if __name__ == "__main__":
    Figure7VisionLanguage().build().save()
