"""
figures/figure3_convergence_map.py
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
from matplotlib.colors import LinearSegmentedColormap, Normalize


class Figure3ConvergenceMap:
    ALIGN_CSV = os.environ.get(
        "CONV_ALIGN_CSV", "/PATH/convergence_alignment.csv"
    )
    OUT_PDF = "/PATH/figure3_convergence_map.pdf"

    FAMILY_COLORS = {
        "cxr_specialist": "#E08214",
        "general_ssl": "#2166AC",
        "image_text": "#762A83",
        "vlm_tower": "#117733",
        "histopathology": "#B2182B",
        "fundus": "#444444",
    }
    FAMILY_LABEL = {
        "cxr_specialist": "CXR specialist",
        "general_ssl": "General SSL",
        "image_text": "Image-text",
        "vlm_tower": "VLM tower",
        "histopathology": "Histopathology",
        "fundus": "Fundus",
    }
    FAMILY_ORDER = ["cxr_specialist", "general_ssl", "image_text",
                    "vlm_tower", "histopathology", "fundus"]
    # encoder -> (display name, family); order within list defines axis order
    ENCODERS = [
        ("rad_dino", "RAD-DINO", "cxr_specialist"),
        ("txrv_densenet", "TXRV", "cxr_specialist"),
        ("dinov2_large", "DINOv2", "general_ssl"),
        ("dinov3_l", "DINOv3", "general_ssl"),
        ("clip_vitl14", "CLIP", "image_text"),
        ("siglip2_large", "SigLIP2", "image_text"),
        ("biomedclip_image", "BiomedCLIP", "image_text"),
        ("medgemma_vision", "MedGemma", "vlm_tower"),
        ("llava_med_vision", "LLaVA-Med", "vlm_tower"),
        ("llava_onevision_vision", "LLaVA-OV", "vlm_tower"),
        ("uni", "UNI", "histopathology"),
        ("uni2", "UNI2-h", "histopathology"),
        ("virchow", "Virchow", "histopathology"),
        ("virchow2", "Virchow2", "histopathology"),
        ("phikon_v2", "Phikon-v2", "histopathology"),
        ("prov_gigapath", "GigaPath", "histopathology"),
        ("conch_image", "CONCH", "histopathology"),
        ("retfound", "RETFound", "fundus"),
    ]
    POOLS = [
        ("within_cxr", "Chest radiography"),
        ("within_histo", "Histopathology"),
        ("within_derm", "Dermatology"),
        ("within_fundus", "Fundus"),
        ("within_mammo", "Mammography"),
    ]
    POOL_SHORT = {"within_cxr": "CXR", "within_histo": "Histo", "within_derm": "Derm",
                  "within_fundus": "Fundus", "within_mammo": "Mammo", "floor_cxr": "Floor"}

    SEQ = LinearSegmentedColormap.from_list(
        "convseq",
        ["#F2FAF9", "#BCE3DF", "#7FC6BF", "#3E9A92", "#1C6E68", "#0C3B38"],
    )
    HEAT_VMAX = 80.0          # shared across the five E1 matrices (histo max 76.1)
    WITHIN_COLOR = "#1C6E68"  # pooled within-modality accent
    FLOOR_COLOR = "#9E9E9E"

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
    def _panel_label(ax, letter, title, x=-0.14, y=1.04, dx=0.052):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=25, fontweight="bold",
                ha="left", va="bottom")
        ax.text(x + dx, y, title, transform=ax.transAxes, fontsize=18.5,
                ha="left", va="bottom")

    def _load_data(self):
        df = pd.read_csv(self.ALIGN_CSV)
        e1 = df[df["experiment"] == "E1"].copy()
        self.e1 = e1
        self.enc_ids = [e[0] for e in self.ENCODERS]
        self.enc_name = {e[0]: e[1] for e in self.ENCODERS}
        self.enc_fam = {e[0]: e[2] for e in self.ENCODERS}
        self.idx = {e[0]: i for i, e in enumerate(self.ENCODERS)}

        # summary rows (per-pool mean and CI, plus floor)
        self.summary = e1[e1["comparison_type"] == "within_vs_cross_summary"].copy()
        self.test = e1[e1["comparison_type"] == "within_vs_cross_test"].iloc[0]

    def _pairs(self, sub):
        return self.e1[(self.e1["sub_analysis"] == sub) & self.e1["unit_a"].notna()]

    def _matrix(self, sub):
        M = np.full((18, 18), np.nan)
        for _, r in self._pairs(sub).iterrows():
            if r["unit_a"] in self.idx and r["unit_b"] in self.idx:
                i, j = self.idx[r["unit_a"]], self.idx[r["unit_b"]]
                M[i, j] = r["value_mean"]
                M[j, i] = r["value_mean"]
        return M

    def _summary_row(self, sub):
        r = self.summary[self.summary["sub_analysis"] == sub].iloc[0]
        return r["value_mean"], r["value_ci_low"], r["value_ci_high"]

    def _per_encoder_cxr(self):
        deg = {e: [] for e in self.enc_ids}
        for _, r in self._pairs("within_cxr").iterrows():
            if r["unit_a"] in deg:
                deg[r["unit_a"]].append(r["value_mean"])
            if r["unit_b"] in deg:
                deg[r["unit_b"]].append(r["value_mean"])
        rows = [(e, float(np.mean(v))) for e, v in deg.items() if v]
        return sorted(rows, key=lambda t: t[1])  # ascending (for barh bottom->top)

    def build(self):
        self.fig = plt.figure(figsize=(19.0, 21.4), facecolor="white")

        gs_leg = self.fig.add_gridspec(1, 1, left=0.055, right=0.985, top=0.995, bottom=0.964)
        gs_r1 = self.fig.add_gridspec(1, 3, left=0.075, right=0.965, top=0.926, bottom=0.692,
                                      wspace=0.30)
        gs_r2 = self.fig.add_gridspec(1, 3, left=0.075, right=0.965, top=0.621, bottom=0.386,
                                      width_ratios=[1.0, 1.0, 1.12], wspace=0.34)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.314, bottom=0.055,
                                      width_ratios=[1.0, 1.15], wspace=0.26)

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        subs = [s for s, _ in self.POOLS]
        titles = [t for _, t in self.POOLS]
        letters = ["a", "b", "c", "d", "e"]
        axes_h = [self.fig.add_subplot(gs_r1[0, 0]), self.fig.add_subplot(gs_r1[0, 1]),
                  self.fig.add_subplot(gs_r1[0, 2]), self.fig.add_subplot(gs_r2[0, 0]),
                  self.fig.add_subplot(gs_r2[0, 1])]
        for ax, sub, ttl, lt in zip(axes_h, subs, titles, letters):
            self._panel_heatmap(ax, sub, lt, ttl, cbar=(lt == "e"))

        self._panel_pools(self.fig.add_subplot(gs_r2[0, 2]), "f", "Within-pool alignment")
        self._panel_dist(self.fig.add_subplot(gs_r3[0, 0]), "g", "Within-modality vs floor")
        self._panel_encoders(self.fig.add_subplot(gs_r3[0, 1]), "h",
                             "Per-encoder alignment, chest radiography")
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        fam_handles = [
            Line2D([0], [0], marker="s", lw=0, markersize=16,
                   mfc=self.FAMILY_COLORS[f], mec="white", mew=0.8,
                   label=self.FAMILY_LABEL[f])
            for f in self.FAMILY_ORDER
        ]
        ax.text(0.005, 0.52, "Encoder family", transform=ax.transAxes,
                fontsize=16.5, fontweight="normal", ha="left", va="center")
        leg = ax.legend(fam_handles, [h.get_label() for h in fam_handles],
                        ncol=6, frameon=False, loc="center left",
                        bbox_to_anchor=(0.135, 0.52), columnspacing=1.35,
                        handletextpad=0.4,
                        prop={"size": 16.0, "weight": "normal"})
        ax.add_artist(leg)

    def _panel_heatmap(self, ax, sub, letter, title, cbar=False):
        self._clean(ax)
        M = self._matrix(sub)
        norm = Normalize(vmin=0.0, vmax=self.HEAT_VMAX)
        im = ax.imshow(M, cmap=self.SEQ, norm=norm, aspect="equal")
        names = [self.enc_name[e] for e in self.enc_ids]
        ax.set_xticks(range(18)); ax.set_yticks(range(18))
        ax.set_xticklabels(names, rotation=90, fontsize=9.2)
        ax.set_yticklabels(names, fontsize=9.2)
        for tl, e in zip(ax.get_xticklabels(), self.enc_ids):
            tl.set_color(self.FAMILY_COLORS[self.enc_fam[e]])
        for tl, e in zip(ax.get_yticklabels(), self.enc_ids):
            tl.set_color(self.FAMILY_COLORS[self.enc_fam[e]])
        # family block separators
        bounds = []
        last = None
        for i, e in enumerate(self.enc_ids):
            f = self.enc_fam[e]
            if last is not None and f != last:
                bounds.append(i - 0.5)
            last = f
        for b in bounds:
            ax.axhline(b, color="white", lw=1.8)
            ax.axvline(b, color="white", lw=1.8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        self._panel_label(ax, letter, title, x=-0.20, y=1.045, dx=0.090)
        if cbar:
            cb = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("CKNNA (%)", fontsize=15)
            cb.ax.tick_params(labelsize=13)

    def _panel_pools(self, ax, letter, title):
        self._clean(ax)
        order = ["within_histo", "within_derm", "within_fundus",
                 "within_mammo", "within_cxr"]
        ys = np.arange(len(order))[::-1]
        for y, sub in zip(ys, order):
            m, lo, hi = self._summary_row(sub)
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", ms=12,
                        color=self.WITHIN_COLOR, ecolor=self.WITHIN_COLOR,
                        elinewidth=2.0, capsize=5, zorder=4)
            ax.text(hi + 1.2, y, f"{m:.1f}", va="center", ha="left",
                    fontsize=14, color="#1a1a1a")
        # floor
        fm, flo, fhi = self._summary_row("floor_cxr")
        yf = -1.2
        ax.errorbar(fm, yf, xerr=[[fm - flo], [fhi - fm]], fmt="s", ms=11,
                    color=self.FLOOR_COLOR, ecolor=self.FLOOR_COLOR,
                    elinewidth=2.0, capsize=5, zorder=4)
        ax.text(fhi + 1.2, yf, f"{fm:.1f}", va="center", ha="left",
                fontsize=14, color="#1a1a1a")
        ax.axvline(fm, color=self.FLOOR_COLOR, ls=(0, (4, 3)), lw=1.4, zorder=1)
        ax.text(fm + 0.6, len(order) - 0.4, "floor", color="#777777",
                fontsize=13.5, ha="left", va="center")
        labels = [self.POOL_SHORT[s] for s in order] 
        ax.set_yticks(list(ys) + [yf])
        ax.set_yticklabels([self.POOL_SHORT[s] for s in order] + ["Floor"], fontsize=14.5)
        ax.set_ylim(-2.0, len(order) - 0.4)
        ax.set_xlim(0, 46)
        ax.set_xlabel("Within-pool CKNNA (%)")
        self._panel_label(ax, letter, title, x=-0.26, y=1.045, dx=0.075)

    def _panel_dist(self, ax, letter, title):
        self._clean(ax)
        within = pd.concat([self._pairs(s)["value_mean"] for s, _ in self.POOLS]).values
        floor = self._pairs("floor_cxr")["value_mean"].values
        data = [within, floor]
        parts = ax.violinplot(data, positions=[0, 1], widths=0.8,
                              showmeans=False, showextrema=False)
        for pc, col in zip(parts["bodies"], [self.WITHIN_COLOR, self.FLOOR_COLOR]):
            pc.set_facecolor(col); pc.set_alpha(0.30); pc.set_edgecolor(col)
            pc.set_linewidth(1.4)
        rng = np.random.default_rng(3)
        for x, d, col in zip([0, 1], data, [self.WITHIN_COLOR, self.FLOOR_COLOR]):
            ax.scatter(x + rng.uniform(-0.10, 0.10, len(d)), d, s=14,
                       color=col, alpha=0.35, edgecolor="none", zorder=2)
            ax.plot([x - 0.30, x + 0.30], [np.mean(d), np.mean(d)], color="#1a1a1a",
                    lw=2.6, zorder=5)
            ax.text(x + 0.34, np.mean(d), f"{np.mean(d):.1f}", ha="left", va="center",
                    fontsize=14.5, color="#1a1a1a", zorder=6)
        diff = float(self.test["within_minus_other_mean"])
        p = float(self.test["p_fdr"])
        na, nb = int(self.test["n_a"]), int(self.test["n_b"])
        ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.5, 0.97, f"$\\Delta$ = {diff:.1f},  {ptxt}", transform=ax.transAxes,
                ha="center", va="top", fontsize=15.5, color="#222222")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Within-modality\n(n = {na})", f"Floor\n(n = {nb})"],
                           fontsize=14.5)
        ax.set_ylim(0, max(within.max(), floor.max()) * 1.10)
        ax.set_ylabel("Pairwise CKNNA (%)")
        self._panel_label(ax, letter, title, x=-0.205, y=1.045, dx=0.07)

    def _panel_encoders(self, ax, letter, title):
        self._clean(ax)
        rows = self._per_encoder_cxr()  # ascending
        ys = np.arange(len(rows))
        for y, (e, m) in zip(ys, rows):
            ax.barh(y, m, height=0.66, color=self.FAMILY_COLORS[self.enc_fam[e]],
                    edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(m + 0.3, y, f"{m:.1f}", va="center", ha="left", fontsize=12.5,
                    color="#1a1a1a")
        ax.set_yticks(ys)
        ax.set_yticklabels([self.enc_name[e] for e, _ in rows], fontsize=12.8)
        for tl, (e, _) in zip(ax.get_yticklabels(), rows):
            tl.set_color(self.FAMILY_COLORS[self.enc_fam[e]])
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlim(0, max(m for _, m in rows) * 1.16)
        ax.set_xlabel("Mean CKNNA to the other 17 encoders (%)")
        self._panel_label(ax, letter, title, x=-0.175, y=1.025, dx=0.052)

    def save(self):
        self.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.out_pdf, format="pdf", facecolor="white", bbox_inches="tight")
        return self


if __name__ == "__main__":
    Figure3ConvergenceMap().build().save()
