"""
figures/figure5_consensus_geometry.py
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


class Figure5ConsensusGeometry:
    OUT_PDF = "/PATH/figure5_consensus_geometry.pdf"

    # encoder families (same as Figure 3)
    FAMILY_COLORS = {
        "cxr_specialist": "#E08214", "general_ssl": "#2166AC", "image_text": "#762A83",
        "vlm_tower": "#117733", "histopathology": "#B2182B", "fundus": "#444444",
    }
    FAMILY_LABEL = {
        "cxr_specialist": "CXR specialist", "general_ssl": "General SSL",
        "image_text": "Image-text", "vlm_tower": "VLM tower",
        "histopathology": "Histopathology", "fundus": "Fundus",
    }
    FAMILY_ORDER = ["cxr_specialist", "general_ssl", "image_text",
                    "vlm_tower", "histopathology", "fundus"]
    ENC_NAME = {
        "rad_dino": "RAD-DINO", "txrv_densenet": "TXRV", "dinov2_large": "DINOv2",
        "dinov3_l": "DINOv3", "clip_vitl14": "CLIP", "siglip2_large": "SigLIP2",
        "biomedclip_image": "BiomedCLIP", "medgemma_vision": "MedGemma",
        "llava_med_vision": "LLaVA-Med", "llava_onevision_vision": "LLaVA-OV",
        "uni": "UNI", "uni2": "UNI2-h", "virchow": "Virchow", "virchow2": "Virchow2",
        "phikon_v2": "Phikon-v2", "prov_gigapath": "GigaPath", "conch_image": "CONCH",
        "retfound": "RETFound",
    }
    ENC_FAMILY = {
        "rad_dino": "cxr_specialist", "txrv_densenet": "cxr_specialist",
        "dinov2_large": "general_ssl", "dinov3_l": "general_ssl",
        "clip_vitl14": "image_text", "siglip2_large": "image_text",
        "biomedclip_image": "image_text", "medgemma_vision": "vlm_tower",
        "llava_med_vision": "vlm_tower", "llava_onevision_vision": "vlm_tower",
        "uni": "histopathology", "uni2": "histopathology", "virchow": "histopathology",
        "virchow2": "histopathology", "phikon_v2": "histopathology",
        "prov_gigapath": "histopathology", "conch_image": "histopathology",
        "retfound": "fundus",
    }
    # subgroup categories (panel h, iterated for plotting)
    CAT_ORDER = ["sex", "race", "ethnicity", "insurance"]

    ACCENT = "#1C6E68"
    GRAY = "#9E9E9E"
    FINDING_COLOR = "#555555"
    FIT_COLOR = "#B2182B"

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
        ax.text(x + dx, y, title, transform=ax.transAxes, fontsize=18.0,
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
        self.struct = pd.read_csv(self._resolve("convergence_structure.csv"))
        self.frac = pd.read_csv(self._resolve("fracture_robustness.csv"))
        # per-encoder structure (residual in residual_raw, AUROC in value_mean)
        pe = self.struct[self.struct["sub_analysis"] == "per_encoder_data"].copy()
        pe["residual"] = pe["residual_raw"]
        pe["family"] = pe["unit_a"].map(self.ENC_FAMILY)
        self.pe = pe
        # per-finding (mknn) and subgroup rows
        self.pf = self.frac[self.frac["sub_analysis"] == "per_finding_alignment"].copy()

    def _mantel(self, sub):
        r = self.struct[(self.struct["experiment"] == "E3") &
                        (self.struct["sub_analysis"] == sub)].iloc[0]
        return (r["stat_estimate"], r["stat_ci_low"], r["stat_ci_high"],
                r["p_fdr"], bool(r["significant_fdr05"]))

    def _spear(self, sub):
        r = self.struct[(self.struct["experiment"] == "E4") &
                        (self.struct["sub_analysis"] == sub) &
                        (self.struct["metric"] == "residual_vs_axis")].iloc[0]
        return r["stat_estimate"], r["p_fdr"]

    def _frac_reg(self):
        r = self.frac[self.frac["sub_analysis"] == "fracture_regression"].iloc[0]
        return (r["stat_estimate"], r["p_raw"], r["partial_spearman_estimate"],
                r["partial_p_raw"])

    def _subgroups(self):
        rows = []
        for cat in self.CAT_ORDER:
            d = self.frac[(self.frac["sub_analysis"] == cat) &
                          (self.frac["metric"] == "mknn_per_subgroup")]
            for _, r in d.iterrows():
                rows.append((cat, r["unit_a"], r["value_mean"], r["value_ci_low"],
                             r["value_ci_high"], r["n_cases"]))
        return pd.DataFrame(rows, columns=["cat", "group", "mean", "lo", "hi", "n"])

    def build(self):
        self.fig = plt.figure(figsize=(19.0, 20.8), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.06, right=0.98, top=0.995, bottom=0.930)
        gs_r1 = self.fig.add_gridspec(1, 3, left=0.075, right=0.975, top=0.880, bottom=0.640,
                                      width_ratios=[0.85, 1.0, 1.0], wspace=0.40)
        gs_r2 = self.fig.add_gridspec(1, 3, left=0.075, right=0.975, top=0.560, bottom=0.320,
                                      width_ratios=[1.0, 1.05, 1.0], wspace=0.42)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.075, right=0.975, top=0.245, bottom=0.050,
                                      width_ratios=[1.0, 1.15], wspace=0.30)

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_mantel(self.fig.add_subplot(gs_r1[0, 0]), "a", "Recovers clinical co-occurrence")
        self._panel_scale(self.fig.add_subplot(gs_r1[0, 1]), "b", "Residual vs model size",
                          "log_params", "log10(params_m)", logx=True, xlabel="Parameters (millions)")
        self._panel_scale(self.fig.add_subplot(gs_r1[0, 2]), "c", "Residual vs downstream AUROC",
                          "value_mean", "downstream_auroc", xlabel="Linear-probe AUROC (%)")
        self._panel_scale(self.fig.add_subplot(gs_r2[0, 0]), "d", "Residual vs release year",
                          "year", "release_year", xlabel="Release year")
        self._panel_residual_rank(self.fig.add_subplot(gs_r2[0, 1]), "e", "Residual distance, ranked")
        self._panel_finding(self.fig.add_subplot(gs_r2[0, 2]), "f", "Alignment vs finding prevalence",
                            xcol="log_prev", xlabel=r"Finding log$_{10}$ prevalence", show_reg=True)
        self._panel_finding(self.fig.add_subplot(gs_r3[0, 0]), "g", "Alignment vs finding learnability",
                            xcol="learnability_auroc", xlabel="Finding learnability AUROC (%)")
        self._panel_subgroups(self.fig.add_subplot(gs_r3[0, 1]), "h", "Alignment across patient subgroups")
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        fam = [Line2D([0], [0], marker="o", lw=0, markersize=15, mfc=self.FAMILY_COLORS[f],
                      mec="white", mew=0.8, label=self.FAMILY_LABEL[f]) for f in self.FAMILY_ORDER]
        ax.text(0.005, 0.5, "Encoder family (panels b\u2013e)", transform=ax.transAxes,
                fontsize=16.0, fontweight="normal", ha="left", va="center")
        l1 = ax.legend(fam, [h.get_label() for h in fam], ncol=6, frameon=False,
                       loc="center left", bbox_to_anchor=(0.235, 0.5), columnspacing=1.3,
                       handletextpad=0.4, prop={"size": 15.5, "weight": "normal"})
        ax.add_artist(l1)

    def _panel_mantel(self, ax, letter, title):
        self._clean(ax)
        labels = ["Comorbidity", "ICD-10"]
        keys = ["comorbidity_jaccard", "icd10_hierarchy"]
        cols = [self.ACCENT, self.GRAY]
        xs = np.arange(2)
        for x, k, c in zip(xs, keys, cols):
            est, lo, hi, p, sig = self._mantel(k)
            ax.bar(x, est, width=0.6, color=c, edgecolor="white", linewidth=1.0, zorder=3)
            ax.errorbar(x, est, yerr=[[est - lo], [hi - est]], fmt="none", ecolor="#333333",
                        elinewidth=1.6, capsize=6, zorder=4)
            star = "$p<0.001$" if (sig and p < 0.001) else f"$p={p:.2f}$"
            ax.text(x, hi + 0.03, f"{est:.2f}\n{star}", ha="center", va="bottom",
                    fontsize=13.5, color="#1a1a1a")
        ax.axhline(0, color="#999999", lw=1.0, zorder=1)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=14.5)
        ax.set_ylim(-0.2, 0.92)
        ax.set_ylabel("Spearman r of distance matrices")
        self._panel_label(ax, letter, title, x=-0.30, y=1.04, dx=0.135)

    def _panel_scale(self, ax, letter, title, xcol, sub, logx=False, xlabel=""):
        self._clean(ax)
        d = self.pe.dropna(subset=["residual", xcol])
        x = d[xcol].values.astype(float)
        y = d["residual"].values.astype(float)
        xfit = np.log10(d["params_m"].values.astype(float)) if logx else x
        for _, r in d.iterrows():
            xv = r["params_m"] if logx else r[xcol]
            ax.scatter(xv, r["residual"], s=85, color=self.FAMILY_COLORS[r["family"]],
                       edgecolor="white", linewidth=0.7, zorder=3)
        b, a = np.polyfit(xfit, y, 1)
        xx = np.linspace(xfit.min(), xfit.max(), 50)
        xplot = 10 ** xx if logx else xx
        ax.plot(xplot, a + b * xx, color="#333333", lw=2.0, ls=(0, (5, 2)), zorder=2)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Residual distance to consensus")
        self._panel_label(ax, letter, title, x=-0.215, y=1.04, dx=0.090)

    def _panel_residual_rank(self, ax, letter, title):
        self._clean(ax)
        d = self.pe.dropna(subset=["residual"]).sort_values("residual")
        xs = np.arange(len(d))
        for x, (_, r) in zip(xs, d.iterrows()):
            ax.bar(x, r["residual"], width=0.72, color=self.FAMILY_COLORS[r["family"]],
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([self.ENC_NAME[u] for u in d["unit_a"]], rotation=90, fontsize=11.0)
        for tl, u in zip(ax.get_xticklabels(), d["unit_a"]):
            tl.set_color(self.FAMILY_COLORS[self.ENC_FAMILY[u]])
        ax.set_ylabel("Residual distance to consensus")
        ax.set_xlim(-0.7, len(d) - 0.3)
        self._panel_label(ax, letter, title, x=-0.175, y=1.04, dx=0.090)

    def _panel_finding(self, ax, letter, title, xcol, xlabel, show_reg=False):
        self._clean(ax)
        d = self.pf.dropna(subset=[xcol, "value_mean"])
        x = d[xcol].values.astype(float)
        y = d["value_mean"].values.astype(float)
        ax.scatter(x, y, s=55, color=self.FINDING_COLOR, alpha=0.6,
                   edgecolor="white", linewidth=0.4, zorder=3)
        b, a = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, a + b * xx, color=self.FIT_COLOR, lw=2.2, zorder=4)
        if show_reg:
            pass  # stats reported in caption
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Per-finding mKNN alignment (%)")
        self._panel_label(ax, letter, title, x=-0.185, y=1.04, dx=0.090)

    def _panel_subgroups(self, ax, letter, title):
        self._clean(ax)
        sg = self._subgroups()
        for _, r in sg.iterrows():
            ax.errorbar(r["n"], r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                        fmt="o", ms=11, color=self.ACCENT, ecolor=self.ACCENT,
                        elinewidth=1.5, capsize=4, zorder=3)
        # label the extreme small-n groups (category named for clarity)
        cat_named = {("Native American", "race"): "Native American",
                     ("Patient Refused", "ethnicity"): "Ethnicity not disclosed",
                     ("Pacific Islander", "race"): "Pacific Islander",
                     ("Other", "insurance"): "Other insurance",
                     ("Medicaid", "insurance"): "Medicaid"}
        for _, r in sg.iterrows():
            key = (r["group"], r["cat"])
            if key in cat_named:
                ax.annotate(f"{cat_named[key]} (n={int(r['n'])})", xy=(r["n"], r["mean"]),
                            xytext=(8, 4), textcoords="offset points", fontsize=11.5,
                            color="#333333", ha="left", va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("Subgroup size (cases)")
        ax.set_ylabel("Subgroup mKNN alignment (%)")
        ax.set_ylim(0, 66)
        self._panel_label(ax, letter, title, x=-0.155, y=1.04, dx=0.075)

    def save(self):
        self.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.out_pdf, format="pdf", facecolor="white", bbox_inches="tight")
        return self


if __name__ == "__main__":
    Figure5ConsensusGeometry().build().save()
