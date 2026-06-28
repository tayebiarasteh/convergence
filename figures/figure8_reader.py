"""
figures/figure8_reader.py
Created on June 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


class Figure8Reader:
    OUT_PDF = "/PATH/figure8_reader.pdf"

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

    READER1 = "#1C6E68"     # main gold set (480)
    READER2 = "#7FC6BF"     # extension (150)
    CHANCE = "#B2182B"      # chance / no-skill reference
    BAND = "#3E9A92"        # interpretation-band base

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        self.disp = {e: d for e, d, f in self.ENCODERS}
        self.fam = {e: f for e, d, f in self.ENCODERS}
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
        df = pd.read_csv(self._resolve("reader_study_results.csv"))
        g = df[df["section"] == "gold_label_metrics"].set_index("reader")
        self.gold = g
        self.inter = df[df["section"] == "inter_reader_agreement"].iloc[0]
        self.trip = df[df["section"] == "triplet_grounding"].copy()

    def build(self):
        self.fig = plt.figure(figsize=(18.0, 20.4), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.06, right=0.98, top=0.992, bottom=0.930)
        gs_r1 = self.fig.add_gridspec(1, 1, left=0.165, right=0.905, top=0.885, bottom=0.640)
        gs_r2 = self.fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.560, bottom=0.330,
                                      width_ratios=[1.05, 1.0], wspace=0.30)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.250, bottom=0.045,
                                      width_ratios=[1.0, 1.12], wspace=0.30)

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_triplet_encoders(self.fig.add_subplot(gs_r1[0, 0]), "a",
                                     "Encoders do not predict radiologist similarity triplets")
        self._panel_goldlabels(self.fig.add_subplot(gs_r2[0, 0]), "b",
                               "Automated read vs radiologist gold labels")
        self._panel_interreader(self.fig.add_subplot(gs_r2[0, 1]), "c",
                                "Inter-reader agreement (150 shared)")
        self._panel_distrust(self.fig.add_subplot(gs_r3[0, 0]), "d",
                             "Reader-flagged distrust and image quality")
        self._panel_family(self.fig.add_subplot(gs_r3[0, 1]), "e",
                           "Encoder families sit at chance")
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        fam_handles = [Patch(facecolor=self.FAMILY_COLORS[f], edgecolor="white",
                             label=self.FAMILY_LABEL[f]) for f in self.FAMILY_ORDER]
        leg1 = ax.legend(fam_handles, [h.get_label() for h in fam_handles], ncol=6,
                         frameon=False, loc="center", bbox_to_anchor=(0.5, 0.78),
                         columnspacing=1.8, handletextpad=0.5,
                         prop={"size": 15.5, "weight": "normal"})
        ax.add_artist(leg1)
        rd_handles = [Patch(facecolor=self.READER1, edgecolor="white", label="Reader 1 (480 radiographs)"),
                      Patch(facecolor=self.READER2, edgecolor="white", label="Reader 2 extension (150)")]
        leg2 = ax.legend(rd_handles, [h.get_label() for h in rd_handles], ncol=2,
                         frameon=False, loc="center", bbox_to_anchor=(0.5, 0.24),
                         columnspacing=2.6, handletextpad=0.5,
                         prop={"size": 15.5, "weight": "normal"})
        ax.add_artist(leg2)

    def _panel_triplet_encoders(self, ax, letter, title):
        self._clean(ax)
        t = self.trip.copy()
        t["fam"] = t["encoder"].map(self.fam)
        t = t.sort_values("triplet_acc_mean", ascending=True)  # bottom-up -> highest on top
        ys = np.arange(len(t))
        for y, (_, r) in zip(ys, t.iterrows()):
            c = self.FAMILY_COLORS[r["fam"]]
            ax.plot([r.triplet_acc_ci_low, r.triplet_acc_ci_high], [y, y], color=c, lw=2.2,
                    solid_capstyle="round", zorder=2, alpha=0.9)
            ax.plot(r.triplet_acc_mean, y, "o", color=c, markersize=9, mec="white", mew=0.8, zorder=3)
        ax.axvline(50, color=self.CHANCE, lw=1.8, ls=(0, (5, 2)), zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([self.disp[e] for e in t["encoder"]], fontsize=12.5)
        for tl, e in zip(ax.get_yticklabels(), t["encoder"]):
            tl.set_color(self.FAMILY_COLORS[self.fam[e]])
        ax.set_ylim(-0.7, len(t) - 0.3)
        ax.set_xlim(37, 58)
        ax.set_xlabel("Triplet-prediction accuracy (%)")
        mean = self.trip["triplet_acc_mean"].mean()
        med = self.trip["triplet_acc_mean"].median()
        ax.text(0.985, 0.045,
                f"0 of 18 above chance\nmean {mean:.1f}, median {med:.1f}\n$n=300$ triplets per encoder",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=13.0, color="#222222")
        ax.text(50.25, len(t) - 0.6, "chance (50)", color=self.CHANCE, fontsize=13.0,
                ha="left", va="top")
        self._panel_label(ax, letter, title, x=-0.155, y=1.025, dx=0.070)

    def _panel_goldlabels(self, ax, letter, title):
        self._clean(ax)
        metrics = [("accuracy", "Accuracy"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity")]
        readers = [("reader1", self.READER1), ("reader2_extension", self.READER2)]
        w = 0.36
        xbase = np.arange(len(metrics))
        for j, (rk, col) in enumerate(readers):
            row = self.gold.loc[rk]
            xs = xbase + (j - 0.5) * w
            for i, (mk, _) in enumerate(metrics):
                m = float(row[f"{mk}_mean"]); lo = float(row[f"{mk}_ci_low"]); hi = float(row[f"{mk}_ci_high"])
                ax.bar(xs[i], m, width=w, color=col, edgecolor="white", linewidth=1.0, zorder=3)
                ax.plot([xs[i], xs[i]], [lo, hi], color="#333333", lw=1.5, zorder=4)
                ax.text(xs[i], hi + 1.4, f"{m:.1f}", ha="center", va="bottom", fontsize=12.0, color="#1a1a1a")
        ax.axhline(50, color="#888888", lw=1.4, ls=(0, (4, 3)), zorder=2)
        ax.text(2.46, 51.0, "no skill (50)", color="#888888", fontsize=11.5, ha="right", va="bottom")
        ax.set_xticks(xbase)
        ax.set_xticklabels([lab for _, lab in metrics], fontsize=13.5)
        ax.set_ylabel("Agreement with gold labels (%)")
        ax.set_ylim(0, 104)
        self._panel_label(ax, letter, title, x=-0.165, y=1.04, dx=0.090)

    def _panel_interreader(self, ax, letter, title):
        self._clean(ax)
        bands = [(0.0, 0.2, "slight"), (0.2, 0.4, "fair"), (0.4, 0.6, "moderate"),
                 (0.6, 0.8, "substantial"), (0.8, 1.0, "almost\nperfect")]
        alphas = [0.10, 0.18, 0.30, 0.20, 0.12]
        for (a, b, name), al in zip(bands, alphas):
            ax.axvspan(a, b, ymin=0.46, ymax=0.92, color=self.BAND, alpha=al, zorder=1)
            ax.text((a + b) / 2, 1.62, name, ha="center", va="center", fontsize=10.5,
                    color="#3a3a3a")
        # kappa row (y=1)
        k = float(self.inter["kappa_estimate"]); klo = float(self.inter["kappa_ci_low"]); khi = float(self.inter["kappa_ci_high"])
        ax.plot([klo, khi], [1, 1], color=self.READER1, lw=3.0, solid_capstyle="round", zorder=3)
        ax.plot(k, 1, "o", color=self.READER1, markersize=13, mec="white", mew=1.0, zorder=4)
        ax.text(k, 1.28, f"{k:.3f} [{klo:.3f}, {khi:.3f}]", ha="center", va="bottom", fontsize=12.5,
                color="#1a1a1a")
        # percent-agreement row (y=0), normalized to 0-1
        pa = float(self.inter["percent_agreement_mean"]) / 100.0
        palo = float(self.inter["percent_agreement_ci_low"]) / 100.0
        pahi = float(self.inter["percent_agreement_ci_high"]) / 100.0
        ax.plot([palo, pahi], [0, 0], color="#444444", lw=3.0, solid_capstyle="round", zorder=3)
        ax.plot(pa, 0, "s", color="#444444", markersize=12, mec="white", mew=1.0, zorder=4)
        ax.text(pa, 0.28, f"{pa*100:.1f} [{palo*100:.1f}, {pahi*100:.1f}]", ha="center", va="bottom",
                fontsize=12.5, color="#1a1a1a")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Percent\nagreement", "Cohen's\nweighted $\\kappa$"], fontsize=13.0)
        ax.set_ylim(-0.6, 2.05)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Agreement (kappa; percent agreement / 100)")
        self._panel_label(ax, letter, title, x=-0.235, y=1.04, dx=0.105)

    def _panel_distrust(self, ax, letter, title):
        self._clean(ax)
        groups = [("distrust", "Distrust rate"), ("suboptimal", "Suboptimal or worse")]
        readers = [("reader1", self.READER1), ("reader2_extension", self.READER2)]
        w = 0.36
        xbase = np.arange(len(groups))
        for j, (rk, col) in enumerate(readers):
            row = self.gold.loc[rk]
            xs = xbase + (j - 0.5) * w
            # distrust (with CI)
            dm = float(row["distrust_rate_mean"]); dlo = float(row["distrust_rate_ci_low"]); dhi = float(row["distrust_rate_ci_high"])
            ax.bar(xs[0], dm, width=w, color=col, edgecolor="white", linewidth=1.0, zorder=3)
            ax.plot([xs[0], xs[0]], [dlo, dhi], color="#333333", lw=1.5, zorder=4)
            ax.text(xs[0], dhi + 1.0, f"{dm:.1f}", ha="center", va="bottom", fontsize=12.0, color="#1a1a1a")
            # suboptimal fraction x100 (no CI)
            sub = float(row["frac_suboptimal_or_worse"]) * 100.0
            ax.bar(xs[1], sub, width=w, color=col, edgecolor="white", linewidth=1.0, zorder=3)
            ax.text(xs[1], sub + 1.0, f"{sub:.0f}", ha="center", va="bottom", fontsize=12.0, color="#1a1a1a")
        ax.set_xticks(xbase)
        ax.set_xticklabels([lab for _, lab in groups], fontsize=13.5)
        ax.set_ylabel("Percent of radiographs (%)")
        ax.set_ylim(0, 70)
        self._panel_label(ax, letter, title, x=-0.165, y=1.04, dx=0.090)

    def _panel_family(self, ax, letter, title):
        self._clean(ax)
        t = self.trip.copy()
        t["fam"] = t["encoder"].map(self.fam)
        rng = np.random.default_rng(7)
        xs = np.arange(len(self.FAMILY_ORDER))
        for i, f in enumerate(self.FAMILY_ORDER):
            sub = t[t["fam"] == f]["triplet_acc_mean"].values.astype(float)
            c = self.FAMILY_COLORS[f]
            fmean = sub.mean()
            ax.plot([i - 0.30, i + 0.30], [fmean, fmean], color=c, lw=3.4, solid_capstyle="round", zorder=3)
            jit = rng.uniform(-0.13, 0.13, len(sub))
            ax.scatter(i + jit, sub, s=46, color=c, alpha=0.55, edgecolor="white", linewidth=0.5, zorder=4)
        ax.axhline(50, color=self.CHANCE, lw=1.8, ls=(0, (5, 2)), zorder=1)
        ax.text(len(self.FAMILY_ORDER) - 0.5, 50.4, "chance (50)", color=self.CHANCE,
                fontsize=12.5, ha="right", va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels([self.FAMILY_LABEL[f].replace(" ", "\n") for f in self.FAMILY_ORDER], fontsize=12.0)
        for tl, f in zip(ax.get_xticklabels(), self.FAMILY_ORDER):
            tl.set_color(self.FAMILY_COLORS[f])
        ax.set_ylabel("Triplet-prediction accuracy (%)")
        ax.set_ylim(37, 58)
        ax.set_xlim(-0.6, len(self.FAMILY_ORDER) - 0.4)
        self._panel_label(ax, letter, title, x=-0.150, y=1.04, dx=0.090)

    def save(self):
        self.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.out_pdf, format="pdf", facecolor="white", bbox_inches="tight")
        print(f"wrote {self.out_pdf}")
        return self


if __name__ == "__main__":
    Figure8Reader().build().save()
