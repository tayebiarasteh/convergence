"""
figures/figure6_deployable.py
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


class Figure6Deployable:
    OUT_PDF = "/PATH/figure6_deployable.pdf"

    TRANSFER = "#1C6E68"   # achieved / transferred
    ORACLE = "#E08214"     # within-source oracle / in-distribution
    BASELINE = "#9E9E9E"   # identity baseline / chance
    SEQ = LinearSegmentedColormap.from_list(
        "teal_seq", ["#F2FAF9", "#BCE3DF", "#7FC6BF", "#3E9A92", "#1C6E68", "#0C3B38"])

    SITE_ORDER = ["mimic", "chexpert", "nih_cxr14", "padchest", "vindr_cxr", "vindr_pcxr"]
    SITE_NAME = {"mimic": "MIMIC", "chexpert": "CheXpert", "nih_cxr14": "CXR14",
                 "padchest": "PadChest", "vindr_cxr": "VinDr-CXR", "vindr_pcxr": "VinDr-PCXR"}
    FINDING_NAME = {
        "atelectasis": "Atelectasis", "cardiomegaly": "Cardiomegaly",
        "consolidation": "Consolidation", "edema": "Edema",
        "enlarged_cardiomediastinum": "Enl. cardiomediastinum", "fracture": "Fracture",
        "lung_lesion": "Lung lesion", "lung_opacity": "Lung opacity",
        "pleural_effusion": "Pleural effusion", "pleural_other": "Pleural other",
        "pneumonia": "Pneumonia", "pneumothorax": "Pneumothorax",
        "support_devices": "Support devices",
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
        df = pd.read_csv(self._resolve("deployable_artifact.csv"))
        self.ce = df[df["sub_analysis"] == "cross_encoder_probe"].copy()
        self.cs = df[df["sub_analysis"] == "cross_site_probe"].copy()
        self.st = df[df["sub_analysis"] == "stitching"].copy()
        self.dr = df[df["metric"] == "drift_auroc"].copy()

    def build(self):
        self.fig = plt.figure(figsize=(18.0, 20.0), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.06, right=0.98, top=0.992, bottom=0.952)
        gs_r1 = self.fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.900, bottom=0.660,
                                      width_ratios=[1.1, 1.0], wspace=0.30)
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.085, right=0.975, top=0.580, bottom=0.330,
                                      width_ratios=[1.25, 1.0], wspace=0.32)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.250, bottom=0.050,
                                      width_ratios=[1.0, 1.05], wspace=0.30)

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_retention(self.fig.add_subplot(gs_r1[0, 0]), "a", "Cross-encoder classifier retention")
        self._panel_transfer_oracle(self.fig.add_subplot(gs_r1[0, 1]), "b", "Transferred vs oracle AUROC")
        self._panel_crosssite(self.fig.add_subplot(gs_r2[0, 0]), "c", "Cross-site transfer by finding")
        self._panel_persite(self.fig.add_subplot(gs_r2[0, 1]), "d", "Transfer AUROC per site")
        self._panel_stitching(self.fig.add_subplot(gs_r3[0, 0]), "e", "Feature stitching recovers the oracle")
        self._panel_drift(self.fig.add_subplot(gs_r3[0, 1]), "f", "Consensus-deviation drift detection")
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        items = [("Transferred", self.TRANSFER), ("Oracle / in-distribution", self.ORACLE),
                 ("Baseline / chance", self.BASELINE)]
        handles = [Line2D([0], [0], marker="s", lw=0, markersize=16, mfc=c, mec="white",
                          mew=0.8, label=lab) for lab, c in items]
        leg = ax.legend(handles, [h.get_label() for h in handles], ncol=3, frameon=False,
                        loc="center", bbox_to_anchor=(0.5, 0.5), columnspacing=2.4,
                        handletextpad=0.5, prop={"size": 16.5, "weight": "normal"})
        ax.add_artist(leg)

    def _panel_retention(self, ax, letter, title):
        self._clean(ax)
        v = self.ce["retention_mean"].dropna().values.astype(float)
        med, q1, q3, mean = np.median(v), np.percentile(v, 25), np.percentile(v, 75), v.mean()
        ax.hist(v, bins=np.arange(40, 106, 2.5), color=self.TRANSFER, alpha=0.85,
                edgecolor="white", linewidth=0.5, zorder=2)
        ax.axvspan(q1, q3, color=self.TRANSFER, alpha=0.13, zorder=1)
        ax.axvline(med, color="#0C3B38", lw=2.4, zorder=4)
        ax.axvline(100, color="#888888", lw=1.4, ls=(0, (4, 3)), zorder=3)
        ymax = ax.get_ylim()[1]
        ax.text(med - 1.2, ymax * 0.92, f"median {med:.1f}%", color="#0C3B38",
                fontsize=14.0, ha="right", va="top")
        ax.text(0.035, 0.74, f"IQR {q1:.1f}\u2013{q3:.1f}\nmean {mean:.1f}%\n$n=${len(v):,} transfers",
                transform=ax.transAxes, fontsize=13.5, ha="left", va="top", color="#222222")
        ax.text(101.5, ymax * 0.55, "oracle\n(100%)", color="#888888", fontsize=12.0,
                ha="left", va="center")
        ax.set_xlabel("Retention (% of within-encoder oracle AUROC)")
        ax.set_ylabel("Cross-encoder transfers")
        ax.set_xlim(40, 106)
        self._panel_label(ax, letter, title, x=-0.155, y=1.04, dx=0.090)

    def _panel_transfer_oracle(self, ax, letter, title):
        self._clean(ax)
        pair = self.ce.groupby(["unit_b", "unit_c"]).agg(
            tx=("auroc_xenc", "mean"), orc=("oracle_mean", "mean")).reset_index()
        x = pair["orc"].values.astype(float)
        y = pair["tx"].values.astype(float) * 100.0
        ax.scatter(x, y, s=42, color=self.TRANSFER, alpha=0.45, edgecolor="white",
                   linewidth=0.3, zorder=3)
        lo, hi = 48, 91
        ax.plot([lo, hi], [lo, hi], color="#444444", lw=1.8, ls=(0, (5, 2)), zorder=2)
        ax.text(90.3, 89.2, "transferred = oracle", color="#444444", fontsize=12.5,
                ha="right", va="bottom", rotation=33, rotation_mode="anchor")
        ax.set_xlabel("Within-target oracle AUROC")
        ax.set_ylabel("Transferred AUROC")
        ax.set_xlim(79.5, 90.5)
        ax.set_ylim(48, 90)
        self._panel_label(ax, letter, title, x=-0.165, y=1.04, dx=0.090)

    def _panel_crosssite(self, ax, letter, title):
        self._clean(ax)
        mat = self.cs.pivot_table(index="unit_a", columns="unit_c", values="value_mean",
                                  aggfunc="mean")
        cols = [c for c in self.SITE_ORDER if c in mat.columns]
        # order findings by mean transfer (descending) for readability
        row_order = mat[cols].mean(axis=1).sort_values(ascending=False).index.tolist()
        M = mat.loc[row_order, cols].values.astype(float)
        norm = Normalize(vmin=45, vmax=95)
        im = ax.imshow(M, cmap=self.SEQ, norm=norm, aspect="auto")
        nr, nc = M.shape
        for i in range(nr):
            for j in range(nc):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "\u00b7", ha="center", va="center", color="#CCCCCC", fontsize=14)
                    continue
                shade = self.SEQ(norm(M[i, j]))
                lum = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        color="white" if lum < 0.55 else "#1a1a1a", fontsize=11.0)
        ax.set_xticks(range(nc))
        ax.set_xticklabels([self.SITE_NAME[c] for c in cols], rotation=35, ha="right", fontsize=12.5)
        for tl, c in zip(ax.get_xticklabels(), cols):
            if c == "mimic":
                tl.set_color(self.ORACLE)
        ax.set_yticks(range(nr))
        ax.set_yticklabels([self.FINDING_NAME[r] for r in row_order], fontsize=11.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = self.fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label("Transfer AUROC", fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.text(0.0, 1.015, "MIMIC column is in-distribution", transform=ax.transAxes,
                fontsize=12.0, color=self.ORACLE, ha="left", va="bottom")
        self._panel_label(ax, letter, title, x=-0.235, y=1.065, dx=0.115)

    def _panel_persite(self, ax, letter, title):
        self._clean(ax)
        means, ns, colors, labels = [], [], [], []
        for s in self.SITE_ORDER:
            d = self.cs[self.cs["unit_c"] == s]["value_mean"].dropna()
            if not len(d):
                continue
            means.append(d.mean()); ns.append(len(d))
            colors.append(self.ORACLE if s == "mimic" else self.TRANSFER)
            labels.append(self.SITE_NAME[s])
        xs = np.arange(len(means))
        ax.bar(xs, means, width=0.66, color=colors, edgecolor="white", linewidth=1.0, zorder=3)
        ext = self.cs[self.cs["unit_c"] != "mimic"]["value_mean"].dropna().mean()
        ax.axhline(ext, color="#444444", lw=1.6, ls=(0, (5, 2)), zorder=4)
        for x, m, n in zip(xs, means, ns):
            tag = f"{m:.1f}" + (f"\n$n=${n}" if n <= 1 else "")
            ax.text(x, m + 1.2, tag, ha="center", va="bottom", fontsize=12.5, color="#1a1a1a")
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=13.0)
        for tl, lab in zip(ax.get_xticklabels(), labels):
            if lab == "MIMIC":
                tl.set_color(self.ORACLE)
        ax.set_ylabel("Mean transfer AUROC")
        ax.set_ylim(0, 92)
        self._panel_label(ax, letter, title, x=-0.175, y=1.04, dx=0.090)

    def _panel_stitching(self, ax, letter, title):
        self._clean(ax)
        order = ["affine", "oracle", "identity"]
        names = {"affine": "Affine map", "oracle": "Oracle", "identity": "Identity baseline"}
        cmap = {"affine": self.TRANSFER, "oracle": self.ORACLE, "identity": self.BASELINE}
        data = [self.st[self.st["method"] == m]["value_mean"].dropna().values for m in order]
        bp = ax.boxplot(data, positions=np.arange(len(order)), widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color="#1a1a1a", lw=2.0),
                        whiskerprops=dict(color="#666666", lw=1.4),
                        capprops=dict(color="#666666", lw=1.4))
        rng = np.random.default_rng(3)
        for i, (m, d) in enumerate(zip(order, data)):
            bp["boxes"][i].set(facecolor=cmap[m], alpha=0.55, edgecolor=cmap[m], linewidth=1.6)
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(d)), d, s=12, color=cmap[m],
                       alpha=0.20, edgecolor="none", zorder=4)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([f"{names[m]}\n($n=${len(d)})" for m, d in zip(order, data)], fontsize=13.5)
        ax.set_ylabel("Stitched AUROC")
        ax.set_ylim(10, 102)
        self._panel_label(ax, letter, title, x=-0.175, y=1.04, dx=0.090)

    def _panel_drift(self, ax, letter, title):
        self._clean(ax)
        self.dr["site"] = self.dr["sub_analysis"].str.replace("mimic_vs_", "", regex=False)
        d = self.dr.set_index("site").reindex(
            [s for s in self.SITE_ORDER if s != "mimic"]).dropna(subset=["value_mean"])
        xs = np.arange(len(d))
        ax.bar(xs, d["value_mean"].values, width=0.6, color=self.BASELINE,
               edgecolor="white", linewidth=1.0, zorder=3)
        ax.axhline(50, color="#B2182B", lw=1.8, ls=(0, (5, 2)), zorder=4)
        ax.text(len(d) - 0.5, 52.0, "chance (50)", color="#B2182B", fontsize=13.0,
                ha="right", va="bottom")
        for x, (_, r) in zip(xs, d.iterrows()):
            ax.text(x, r["value_mean"] - 3.0, f"{r['value_mean']:.1f}", ha="center", va="top",
                    fontsize=12.5, color="white")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"vs {self.SITE_NAME[s]}" for s in d.index], rotation=35,
                           ha="right", fontsize=13.0)
        ax.set_ylabel("Drift-detector AUROC")
        ax.set_ylim(0, 100)
        self._panel_label(ax, letter, title, x=-0.155, y=1.04, dx=0.090)

    def save(self):
        self.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.out_pdf, format="pdf", facecolor="white", bbox_inches="tight")
        return self


if __name__ == "__main__":
    Figure6Deployable().build().save()
