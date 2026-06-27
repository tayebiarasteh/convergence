"""
figures/figure2_controlled_driver.py
Created on June 27, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize


class Figure2ControlledDriver:
    ALIGN_CSV = os.environ.get(
        "CONV_ALIGN_CSV", "/PATH/convergence_alignment.csv"
    )
    OUT_PDF = "/PATH/figure2_controlled_driver.pdf"

    OBJ_COLORS = {
        "ssl": "#2166AC",          # self-supervised
        "supervised": "#E08214",   # clinical-label supervised
        "image_text": "#762A83",   # image-text contrastive
    }
    CROSS_COLOR = "#9E9E9E"        # different-objective / neutral
    OBJ_ORDER = ["ssl", "supervised", "image_text"]
    OBJ_LABEL = {"ssl": "SSL", "supervised": "Supervised", "image_text": "Image-text"}
    BB_ORDER = ["vit_s", "vit_b"]
    BB_LABEL = {"vit_s": "ViT-S", "vit_b": "ViT-B"}
    MOD_MARKER = {"cxr": "o", "histo": "s"}
    MOD_LABEL = {"cxr": "Chest radiography", "histo": "Histopathology"}

    SEQ = LinearSegmentedColormap.from_list(
        "convseq",
        ["#F2FAF9", "#BCE3DF", "#7FC6BF", "#3E9A92", "#1C6E68", "#0C3B38"],
    )
    HEAT_VMAX = 65.0  # shared upper bound across all heatmaps (histo max 63.4)

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        self._set_rcparams()
        self._load_data()
        self.fig = None

    @staticmethod
    def _resolve(name, env=None):
        if env and os.environ.get(env):
            return os.environ[env]
        for d in ("/PATH/project_data", "/PATH"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return os.path.join("/PATH", name)

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
    def _panel_label(ax, letter, title, x=-0.14, y=1.045, dx=0.052):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=25, fontweight="bold",
                ha="left", va="bottom")
        ax.text(x + dx, y, title, transform=ax.transAxes, fontsize=18.5,
                ha="left", va="bottom")

    @staticmethod
    def _obj(u):
        return u.split("__")[1]

    @staticmethod
    def _bb(u):
        return u.split("__")[2]

    def _key(self, u):
        return (self._obj(u), self._bb(u))

    def _load_data(self):
        df = pd.read_csv(self._resolve("convergence_alignment.csv", "CONV_ALIGN_CSV"))
        e5 = df[(df["experiment"] == "E5") &
                (df["sub_analysis"] == "controlled_driver")].copy()
        if e5.empty:
            raise ValueError("No E5 controlled_driver rows found in CSV")
        e5["oa"] = e5["unit_a"].map(self._obj)
        e5["ba"] = e5["unit_a"].map(self._bb)
        e5["ob"] = e5["unit_b"].map(self._obj)
        e5["bb"] = e5["unit_b"].map(self._bb)
        e5["same_obj"] = e5["same_objective"].astype(str).isin(["True", "true", "1"])
        self.e5 = e5

        # encoder order used on heatmap axes
        self.enc_order = [(o, b) for o in self.OBJ_ORDER for b in self.BB_ORDER]
        self.enc_tick = [f"{self.OBJ_LABEL[o]}\n{self.BB_LABEL[b]}" for (o, b) in self.enc_order]

        # paired (mknn, cknna) per pair, for panel f
        self.paired = e5.pivot_table(
            index=["modality_context", "unit_a", "unit_b", "same_obj"],
            columns="metric", values="value_mean").reset_index()
        self.paired["oa"] = self.paired["unit_a"].map(self._obj)
        self.paired["ob"] = self.paired["unit_b"].map(self._obj)

    def _ck(self, mod):
        return self.e5[(self.e5["modality_context"] == mod) & (self.e5["metric"] == "cknna")]

    def _matrix6(self, mod):
        """6x6 symmetric CKNNA matrix in enc_order; diagonal NaN."""
        idx = {k: i for i, k in enumerate(self.enc_order)}
        M = np.full((6, 6), np.nan)
        for _, r in self._ck(mod).iterrows():
            i, j = idx[(r["oa"], r["ba"])], idx[(r["ob"], r["bb"])]
            M[i, j] = r["value_mean"]
            M[j, i] = r["value_mean"]
        return M

    def _matrix3(self, mod):
        """3x3 objective-pairing mean CKNNA (diagonal = matched-objective)."""
        oi = {o: i for i, o in enumerate(self.OBJ_ORDER)}
        acc = {(i, j): [] for i in range(3) for j in range(3)}
        for _, r in self._ck(mod).iterrows():
            i, j = oi[r["oa"]], oi[r["ob"]]
            acc[(i, j)].append(r["value_mean"])
            acc[(j, i)].append(r["value_mean"])
        M = np.full((3, 3), np.nan)
        for (i, j), v in acc.items():
            if v:
                M[i, j] = float(np.mean(v))
        return M

    def _matched(self, mod):
        """matched-objective CKNNA mean and std per objective (the same-objective pair)."""
        ck = self._ck(mod)
        out = {}
        for o in self.OBJ_ORDER:
            row = ck[(ck["same_obj"]) & (ck["oa"] == o)]
            if len(row):
                out[o] = (float(row["value_mean"].iloc[0]), float(row["value_std"].iloc[0]))
        return out

    def _same_diff(self, mod):
        ck = self._ck(mod)
        same = ck[ck["same_obj"]]
        diff = ck[~ck["same_obj"]]
        return same, diff

    def build(self):
        self.fig = plt.figure(figsize=(18.0, 20.4), facecolor="white")

        gs_leg = self.fig.add_gridspec(1, 1, left=0.055, right=0.985, top=0.995, bottom=0.945)
        gs_r1 = self.fig.add_gridspec(1, 3, left=0.070, right=0.975, top=0.905, bottom=0.660,
                                      width_ratios=[1.02, 1.02, 1.05], wspace=0.42)
        gs_r2 = self.fig.add_gridspec(1, 3, left=0.070, right=0.975, top=0.575, bottom=0.350,
                                      width_ratios=[1.0, 1.0, 1.12], wspace=0.40)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.115, right=0.885, top=0.275, bottom=0.045,
                                      width_ratios=[1.0, 1.0], wspace=0.40)

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_heatmap6(self.fig.add_subplot(gs_r1[0, 0]), "cxr", "a",
                             "Controlled encoders, chest radiography")
        self._panel_heatmap6(self.fig.add_subplot(gs_r1[0, 1]), "histo", "b",
                             "Controlled encoders, histopathology", cbar=True)
        self._panel_samediff(self.fig.add_subplot(gs_r1[0, 2]), "c",
                             "Shared vs different objective")
        self._panel_matched(self.fig.add_subplot(gs_r2[0, 0]), "cxr", "d",
                            "Matched objective, chest radiography")
        self._panel_matched(self.fig.add_subplot(gs_r2[0, 1]), "histo", "e",
                            "Matched objective, histopathology")
        self._panel_metric(self.fig.add_subplot(gs_r2[0, 2]), "f",
                           "mKNN vs CKNNA agreement")
        self._panel_heatmap3(self.fig.add_subplot(gs_r3[0, 0]), "cxr", "g",
                             "Objective pairing, chest radiography")
        self._panel_heatmap3(self.fig.add_subplot(gs_r3[0, 1]), "histo", "h",
                             "Objective pairing, histopathology", cbar=True)
        return self

    def _draw_legend(self, ax):
        ax.axis("off")
        obj_handles = [
            Line2D([0], [0], marker="s", lw=0, markersize=16,
                   mfc=self.OBJ_COLORS[o], mec="white", mew=0.8, label=self.OBJ_LABEL[o])
            for o in self.OBJ_ORDER
        ]
        obj_handles.append(
            Line2D([0], [0], marker="s", lw=0, markersize=16,
                   mfc=self.CROSS_COLOR, mec="white", mew=0.8, label="Different objective")
        )
        mod_handles = [
            Line2D([0], [0], marker=self.MOD_MARKER[m], lw=0, markersize=14,
                   mfc="#555555", mec="white", mew=0.8, label=self.MOD_LABEL[m])
            for m in ["cxr", "histo"]
        ]

        ax.text(0.005, 0.74, "Training objective", transform=ax.transAxes,
                fontsize=16.5, fontweight="normal", ha="left", va="center")
        leg1 = ax.legend(obj_handles, [h.get_label() for h in obj_handles],
                         ncol=4, frameon=False, loc="center left",
                         bbox_to_anchor=(0.155, 0.74), columnspacing=1.5, handletextpad=0.45,
                         prop={"size": 16.0, "weight": "normal"})
        ax.add_artist(leg1)

        ax.text(0.005, 0.27, "Modality", transform=ax.transAxes,
                fontsize=16.5, fontweight="normal", ha="left", va="center")
        leg2 = ax.legend(mod_handles, [h.get_label() for h in mod_handles],
                         ncol=2, frameon=False, loc="center left",
                         bbox_to_anchor=(0.155, 0.27), columnspacing=1.5, handletextpad=0.45,
                         prop={"size": 16.0, "weight": "normal"})
        ax.add_artist(leg2)

    def _panel_heatmap6(self, ax, mod, letter, title, cbar=False):
        self._clean(ax)
        M = self._matrix6(mod)
        norm = Normalize(vmin=0.0, vmax=self.HEAT_VMAX)
        im = ax.imshow(M, cmap=self.SEQ, norm=norm, aspect="equal")
        n = 6
        for i in range(n):
            for j in range(n):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "\u2014", ha="center", va="center",
                            color="#BBBBBB", fontsize=14)
                    continue
                shade = self.SEQ(norm(M[i, j]))
                lum = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        color="white" if lum < 0.55 else "#1a1a1a", fontsize=13.5)
        cap = [self.BB_LABEL[b] for (_, b) in self.enc_order]
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(cap, fontsize=12.5)
        ax.set_yticklabels(cap, fontsize=12.5)
        for tl, (o, _) in zip(ax.get_xticklabels(), self.enc_order):
            tl.set_color(self.OBJ_COLORS[o])
        for tl, (o, _) in zip(ax.get_yticklabels(), self.enc_order):
            tl.set_color(self.OBJ_COLORS[o])
        # objective-block separators
        for s in (1.5, 3.5):
            ax.axhline(s, color="white", lw=2.4)
            ax.axvline(s, color="white", lw=2.4)
        # objective block labels under the x-axis (rows carry the same objective by color)
        for c, o in zip((0.5, 2.5, 4.5), self.OBJ_ORDER):
            ax.text(c, -0.205, self.OBJ_LABEL[o], transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=14.0, color=self.OBJ_COLORS[o],
                    clip_on=False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        self._panel_label(ax, letter, title, x=-0.16, y=1.07, dx=0.090)
        if cbar:
            cb = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("CKNNA (%)", fontsize=15)
            cb.ax.tick_params(labelsize=13)

    def _panel_samediff(self, ax, letter, title):
        self._clean(ax)
        rng = np.random.default_rng(7)
        centers = {"cxr": (0.0, 1.1), "histo": (2.7, 3.8)}  # (same x, diff x)
        ymax = 0
        for mod in ["cxr", "histo"]:
            same, diff = self._same_diff(mod)
            xs, xd = centers[mod]
            # different-objective pairs (gray)
            yv = diff["value_mean"].values
            ax.scatter(xd + rng.uniform(-0.13, 0.13, len(yv)), yv, s=70,
                       marker=self.MOD_MARKER[mod], facecolor=self.CROSS_COLOR,
                       edgecolor="white", linewidth=0.6, alpha=0.85, zorder=3)
            # same-objective pairs colored by objective
            for _, r in same.iterrows():
                ax.scatter(xs + rng.uniform(-0.10, 0.10), r["value_mean"], s=145,
                           marker=self.MOD_MARKER[mod],
                           facecolor=self.OBJ_COLORS[r["oa"]], edgecolor="white",
                           linewidth=0.9, zorder=5)
            m_same = float(same["value_mean"].mean())
            m_diff = float(diff["value_mean"].mean())
            for xc, mv in [(xs, m_same), (xd, m_diff)]:
                ax.plot([xc - 0.26, xc + 0.26], [mv, mv], color="#222222", lw=2.6, zorder=6)
            ax.annotate("", xy=(xs, m_same), xytext=(xd, m_diff),
                        arrowprops=dict(arrowstyle="-", color="#222222", lw=1.0, ls=(0, (4, 3))),
                        zorder=2)
            ax.text((xs + xd) / 2, max(m_same, m_diff) + 4.0,
                    f"$\\Delta$ {m_same - m_diff:+.1f}", ha="center", va="bottom",
                    fontsize=15, color="#222222")
            ymax = max(ymax, np.nanmax(np.concatenate([same["value_mean"].values,
                                                       diff["value_mean"].values])))
        ax.set_xticks([centers["cxr"][0], centers["cxr"][1],
                       centers["histo"][0], centers["histo"][1]])
        ax.set_xticklabels(["Shared", "Different", "Shared", "Different"], fontsize=14.5)
        ax.set_xlim(-0.6, 4.4)
        ax.set_ylim(0, ymax + 12)
        ax.set_ylabel("CKNNA (%)")
        # modality group labels centered under each pair
        ax.text(0.23, -0.135, "Chest radiography", transform=ax.transAxes,
                ha="center", va="top", fontsize=14.5)
        ax.text(0.77, -0.135, "Histopathology", transform=ax.transAxes,
                ha="center", va="top", fontsize=14.5)
        self._panel_label(ax, letter, title, x=-0.155, y=1.045, dx=0.090)

    def _panel_matched(self, ax, mod, letter, title):
        self._clean(ax)
        mt = self._matched(mod)
        xs = np.arange(len(self.OBJ_ORDER))
        for x, o in zip(xs, self.OBJ_ORDER):
            mean, sd = mt[o]
            ax.bar(x, mean, width=0.62, color=self.OBJ_COLORS[o],
                   edgecolor="white", linewidth=1.0, zorder=3)
            ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor="#333333",
                        elinewidth=1.6, capsize=5, zorder=4)
            ax.text(x, mean + sd + (1.6 if mod == "cxr" else 1.8), f"{mean:.1f}",
                    ha="center", va="bottom", fontsize=15, color="#1a1a1a")
        ax.set_xticks(xs)
        ax.set_xticklabels([self.OBJ_LABEL[o] for o in self.OBJ_ORDER], fontsize=14.5)
        for tl, o in zip(ax.get_xticklabels(), self.OBJ_ORDER):
            tl.set_color(self.OBJ_COLORS[o])
        top = max(v[0] + v[1] for v in mt.values())
        ax.set_ylim(0, top * 1.22)
        ax.set_ylabel("Matched-objective CKNNA (%)")
        self._panel_label(ax, letter, title, x=-0.175, y=1.045, dx=0.090)

    def _panel_metric(self, ax, letter, title):
        self._clean(ax)
        d = self.paired.dropna(subset=["mknn", "cknna"]).copy()
        lim = 67
        ax.plot([0, lim], [0, lim], ls=(0, (4, 3)), color="#999999", lw=1.4, zorder=1)
        ax.text(lim - 2, lim - 2, "y = x", color="#777777", fontsize=14,
                ha="right", va="top", rotation=45, rotation_mode="anchor")
        for _, r in d.iterrows():
            same = bool(r["same_obj"])
            col = self.OBJ_COLORS[r["oa"]] if same else self.CROSS_COLOR
            ax.scatter(r["mknn"], r["cknna"], s=130 if same else 80,
                       marker=self.MOD_MARKER[r["modality_context"]],
                       facecolor=col, edgecolor="white", linewidth=0.8,
                       alpha=1.0 if same else 0.85, zorder=5 if same else 3)
        r_pear = float(np.corrcoef(d["mknn"], d["cknna"])[0, 1])
        ax.text(0.04, 0.94, f"Pearson r = {r_pear:.2f}\nn = {len(d)} pairs",
                transform=ax.transAxes, fontsize=15, ha="left", va="top", color="#222222")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("mKNN (%)"); ax.set_ylabel("CKNNA (%)")
        ax.set_aspect("equal")
        self._panel_label(ax, letter, title, x=-0.175, y=1.045, dx=0.090)

    def _panel_heatmap3(self, ax, mod, letter, title, cbar=False):
        self._clean(ax)
        M = self._matrix3(mod)
        norm = Normalize(vmin=0.0, vmax=self.HEAT_VMAX)
        im = ax.imshow(M, cmap=self.SEQ, norm=norm, aspect="equal")
        for i in range(3):
            for j in range(3):
                shade = self.SEQ(norm(M[i, j]))
                lum = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        color="white" if lum < 0.55 else "#1a1a1a", fontsize=16.5)
        labs = [self.OBJ_LABEL[o] for o in self.OBJ_ORDER]
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(labs, fontsize=14.5)
        ax.set_yticklabels(labs, fontsize=14.5)
        for tl, o in zip(ax.get_xticklabels(), self.OBJ_ORDER):
            tl.set_color(self.OBJ_COLORS[o])
        for tl, o in zip(ax.get_yticklabels(), self.OBJ_ORDER):
            tl.set_color(self.OBJ_COLORS[o])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        self._panel_label(ax, letter, title, x=-0.22, y=1.06, dx=0.085)
        if cbar:
            cb = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
            cb.set_label("Mean CKNNA (%)", fontsize=15)
            cb.ax.tick_params(labelsize=13)

    def save(self):
        self.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.out_pdf, format="pdf", facecolor="white", bbox_inches="tight")
        return self


if __name__ == "__main__":
    Figure2ControlledDriver().build().save()
