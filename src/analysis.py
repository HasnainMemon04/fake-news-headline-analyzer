"""Exploratory Data Analysis (EDA) and labelled figure generation.

Implements `EDAAnalyzer` and plotting routines required by Idea 19, Lab 10,
and Section 7 of `docs/proposal.md`:
- Headline length and word-count distributions with IQR outlier bounds.
- Top frequent non-stopword vocabulary terms in Real vs. Fake headlines.
- Source provenance and subject audit (explicitly documenting missing publisher IDs).
- Labelled matplotlib figures saved to `outputs/figures/` for the report and UI.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src import LABEL_NAMES
from src.explanations import tokenize_headline_words
from src.topics import TopicClusterer

REAL_COLOR = "#1b6ca8"
FAKE_COLOR = "#d9534f"
BASELINE_COLOR = "#7f8c8d"
NB_COLOR = "#2b8cbe"
LR_COLOR = "#153859"


class EDAAnalyzer:
    """Computes exploratory statistics and renders labelled matplotlib charts."""

    def __init__(self, output_figures_dir: str | Path = "outputs/figures") -> None:
        self._figures_dir = Path(output_figures_dir)
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._summary: dict[str, Any] = {}

    @property
    def figures_dir(self) -> Path:
        """Directory where generated PNG charts are saved."""
        return self._figures_dir

    @property
    def summary(self) -> dict[str, Any]:
        """Return computed EDA summary dictionary."""
        return dict(self._summary)

    def compute_eda(
        self,
        clean_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        audit_meta: dict[str, Any],
        top_n_words: int = 15,
    ) -> dict[str, Any]:
        """Compute grouped length statistics, frequent words, and provenance summary."""
        if clean_df.empty or dev_df.empty:
            raise ValueError("Cannot run EDA on empty dataframes.")

        length_stats: dict[str, Any] = {}
        for label_val, label_name in LABEL_NAMES.items():
            sub = clean_df[clean_df["label"] == label_val]
            words = sub["word_count"].to_numpy()
            chars = sub["char_length"].to_numpy()
            q1_w, q3_w = np.percentile(words, [25, 75])
            iqr_w = q3_w - q1_w
            length_stats[label_name] = {
                "count": int(len(sub)),
                "word_count_mean": float(np.mean(words)),
                "word_count_std": float(np.std(words)),
                "word_count_median": float(np.median(words)),
                "word_count_q1": float(q1_w),
                "word_count_q3": float(q3_w),
                "word_count_min": int(np.min(words)),
                "word_count_max": int(np.max(words)),
                "word_outliers_above_upper_fence": int(np.sum(words > (q3_w + 1.5 * iqr_w))),
                "char_length_mean": float(np.mean(chars)),
                "char_length_std": float(np.std(chars)),
                "char_length_median": float(np.median(chars)),
                "char_length_min": int(np.min(chars)),
                "char_length_max": int(np.max(chars)),
            }

        # Compute top frequent non-stopword terms on development partition
        top_words: dict[str, list[dict[str, Any]]] = {}
        for label_val, label_name in LABEL_NAMES.items():
            sub_dev = dev_df[dev_df["label"] == label_val]
            counter: Counter[str] = Counter()
            for title in sub_dev["title"].astype(str):
                tokens = [
                    tok
                    for tok in tokenize_headline_words(title)
                    if tok not in ENGLISH_STOP_WORDS and not tok.isdigit()
                ]
                counter.update(tokens)
            top_words[label_name] = [
                {"term": term, "count": int(count)}
                for term, count in counter.most_common(top_n_words)
            ]

        subject_counts: dict[str, dict[str, int]] = {}
        for label_val, label_name in LABEL_NAMES.items():
            sub = clean_df[clean_df["label"] == label_val]
            subject_counts[label_name] = {
                str(k): int(v) for k, v in sub["subject"].value_counts().items()
            }

        provenance_summary = {
            "publisher_column_present": False,
            "missing_publisher_rate": 1.0,
            "documented_real_provenance": "Reuters.com (ISOT official documentation)",
            "documented_fake_provenance": (
                "Multiple unreliable websites flagged by Politifact and Wikipedia "
                "(individual publisher identifiers omitted from public CSVs)"
            ),
            "subject_counts_by_class": subject_counts,
            "note": (
                "CSV schema contains only title, text, subject, and date. "
                "Subject values partition almost perfectly by file origin and are "
                "excluded from model inputs to prevent trivial metadata leakage."
            ),
        }

        self._summary = {
            "dataset_counts": {
                "raw_rows": audit_meta.get("raw_rows", 0),
                "cleaned_unique_titles": int(len(clean_df)),
                "development_rows": int(len(dev_df)),
                "reserved_test_rows": int(len(clean_df) - len(dev_df)),
                "normalized_duplicates_removed": audit_meta.get(
                    "normalized_duplicates_removed", 0
                ),
                "conflicting_records_removed": audit_meta.get(
                    "conflicting_normalized_title_records", 0
                ),
            },
            "length_statistics_by_class": length_stats,
            "word_count_iqr_bounds_overall": audit_meta.get("word_count_iqr_bounds", {}),
            "top_words_development": top_words,
            "source_and_provenance_audit": provenance_summary,
        }
        return self.summary

    def plot_eda_figures(self, clean_df: pd.DataFrame) -> dict[str, str]:
        """Render and save EDA charts to `outputs/figures/`."""
        if not self._summary:
            raise RuntimeError("Call compute_eda() before plot_eda_figures().")

        saved_paths: dict[str, str] = {}

        # Figure 1: Headline Length & Word Count Distributions + Outlier Boxplots
        fig1, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
        real_words = clean_df[clean_df["label"] == 0]["word_count"]
        fake_words = clean_df[clean_df["label"] == 1]["word_count"]

        bins = np.arange(1, 36, 1)
        axes[0].hist(
            real_words.clip(upper=35),
            bins=bins,
            alpha=0.65,
            label=f"Real (mean={real_words.mean():.1f}, median={real_words.median():.0f})",
            color=REAL_COLOR,
            edgecolor="white",
        )
        axes[0].hist(
            fake_words.clip(upper=35),
            bins=bins,
            alpha=0.65,
            label=f"Fake (mean={fake_words.mean():.1f}, median={fake_words.median():.0f})",
            color=FAKE_COLOR,
            edgecolor="white",
        )
        iqr_info = self._summary.get("word_count_iqr_bounds_overall", {})
        upper_fence = iqr_info.get("upper_fence", 23.5)
        axes[0].axvline(
            upper_fence,
            color="#2c3e50",
            linestyle="--",
            linewidth=1.5,
            label=f"IQR Upper Fence ({upper_fence:.1f} words)",
        )
        axes[0].set_title("Headline Word Count Distribution by Class", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Normalized Word Count per Headline (clipped at 35)", fontsize=10)
        axes[0].set_ylabel("Number of Unique Headlines", fontsize=10)
        axes[0].legend(frameon=True, fontsize=9)
        axes[0].grid(axis="y", linestyle=":", alpha=0.6)

        real_chars = clean_df[clean_df["label"] == 0]["char_length"]
        fake_chars = clean_df[clean_df["label"] == 1]["char_length"]
        bp = axes[1].boxplot(
            [real_chars, fake_chars],
            tick_labels=["Real News\n(True.csv)", "Fake News\n(Fake.csv)"],
            patch_artist=True,
            widths=0.45,
            flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.35},
        )
        for patch, color in zip(bp["boxes"], [REAL_COLOR, FAKE_COLOR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axes[1].set_title("Headline Character Length & Outliers by Class", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Character Length (raw title)", fontsize=10)
        axes[1].grid(axis="y", linestyle=":", alpha=0.6)

        fig1.suptitle(
            "Exploratory Data Analysis: Headline Length Distributions (N = 38,720 Cleaned Titles)",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        fig1.tight_layout()
        path1 = self._figures_dir / "eda_length_distributions.png"
        fig1.savefig(path1, bbox_inches="tight")
        plt.close(fig1)
        saved_paths["eda_length_distributions"] = str(path1)

        # Figure 2: Top Frequent Words (Real vs Fake) & Source/Subject Audit
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5.5), dpi=150)
        top_real = self._summary["top_words_development"]["Real"][:12][::-1]
        top_fake = self._summary["top_words_development"]["Fake"][:12][::-1]

        axes2[0].barh(
            [item["term"] for item in top_real],
            [item["count"] for item in top_real],
            color=REAL_COLOR,
            alpha=0.85,
        )
        axes2[0].set_title("Top 12 Terms in Real Headlines (Dev)", fontsize=11, fontweight="bold")
        axes2[0].set_xlabel("Document Frequency", fontsize=10)
        axes2[0].grid(axis="x", linestyle=":", alpha=0.6)

        axes2[1].barh(
            [item["term"] for item in top_fake],
            [item["count"] for item in top_fake],
            color=FAKE_COLOR,
            alpha=0.85,
        )
        axes2[1].set_title("Top 12 Terms in Fake Headlines (Dev)", fontsize=11, fontweight="bold")
        axes2[1].set_xlabel("Document Frequency", fontsize=10)
        axes2[1].grid(axis="x", linestyle=":", alpha=0.6)

        subj_real = self._summary["source_and_provenance_audit"]["subject_counts_by_class"]["Real"]
        subj_fake = self._summary["source_and_provenance_audit"]["subject_counts_by_class"]["Fake"]
        all_subjects = list(dict.fromkeys(list(subj_real.keys()) + list(subj_fake.keys())))
        y_pos = np.arange(len(all_subjects))
        r_vals = [subj_real.get(s, 0) for s in all_subjects]
        f_vals = [subj_fake.get(s, 0) for s in all_subjects]

        axes2[2].barh(y_pos, r_vals, color=REAL_COLOR, alpha=0.85, label="Real (Reuters provenance)")
        axes2[2].barh(
            y_pos,
            f_vals,
            left=r_vals,
            color=FAKE_COLOR,
            alpha=0.85,
            label="Fake (Unattributed sites)",
        )
        axes2[2].set_yticks(y_pos)
        axes2[2].set_yticklabels(all_subjects, fontsize=9)
        axes2[2].set_title("Audit: CSV Subject Metadata by Class\n(Publisher ID Missing = 100%)", fontsize=11, fontweight="bold")
        axes2[2].set_xlabel("Number of Cleaned Headlines", fontsize=10)
        axes2[2].legend(loc="lower right", fontsize=8.5)
        axes2[2].grid(axis="x", linestyle=":", alpha=0.6)

        fig2.suptitle(
            "Exploratory Data Analysis: Lexical Frequencies & Source Provenance Audit",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        fig2.tight_layout()
        path2 = self._figures_dir / "eda_top_words_and_provenance.png"
        fig2.savefig(path2, bbox_inches="tight")
        plt.close(fig2)
        saved_paths["eda_top_words_and_provenance"] = str(path2)

        return saved_paths

    def plot_model_comparison_and_confusion(
        self,
        metrics_by_model: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Render held-out model comparison bar chart and confusion matrices."""
        saved_paths: dict[str, str] = {}

        # Figure 3: Model Comparison Bar Chart on Held-out Test Set (N = 7,744)
        fig3, ax3 = plt.subplots(figsize=(10.5, 5.5), dpi=150)
        metric_keys = [
            ("accuracy", "Accuracy"),
            ("macro_f1", "Macro-F1"),
            ("real_f1", "Real F1 (Class 0)"),
            ("fake_f1", "Fake F1 (Class 1)"),
        ]
        x = np.arange(len(metric_keys))
        width = 0.25
        colors = [BASELINE_COLOR, NB_COLOR, LR_COLOR]

        for idx, model_entry in enumerate(metrics_by_model):
            vals = [float(model_entry[key]) for key, _ in metric_keys]
            offset = (idx - 1) * width
            bars = ax3.bar(
                x + offset,
                vals,
                width=width,
                label=model_entry["model_name"],
                color=colors[idx % len(colors)],
                edgecolor="white",
            )
            for bar, val in zip(bars, vals):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.012,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

        ax3.set_xticks(x)
        ax3.set_xticklabels([label for _, label in metric_keys], fontsize=10, fontweight="bold")
        ax3.set_ylim(0.0, 1.16)
        ax3.set_ylabel("Score on Locked Held-Out Test Set (N = 7,744)", fontsize=10)
        ax3.set_title(
            "Held-Out Test Performance: Majority Baseline vs. Naive Bayes vs. Logistic Regression",
            fontsize=12,
            fontweight="bold",
        )
        ax3.legend(loc="upper center", ncols=len(metrics_by_model), frameon=True, fontsize=9.5)
        ax3.grid(axis="y", linestyle=":", alpha=0.6)
        fig3.tight_layout()
        path3 = self._figures_dir / "model_comparison_metrics.png"
        fig3.savefig(path3, bbox_inches="tight")
        plt.close(fig3)
        saved_paths["model_comparison_metrics"] = str(path3)

        # Figure 4: Confusion Matrices Side-by-Side
        fig4, axes4 = plt.subplots(1, len(metrics_by_model), figsize=(14.5, 4.6), dpi=150)
        if len(metrics_by_model) == 1:
            axes4 = [axes4]

        for ax, model_entry in zip(axes4, metrics_by_model):
            cm = np.asarray(model_entry["confusion_matrix"], dtype=int)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred: Real (0)", "Pred: Fake (1)"], fontsize=9.5)
            ax.set_yticklabels(["True: Real (0)", "True: Fake (1)"], fontsize=9.5)
            ax.set_title(
                f"{model_entry['model_name']}\n(Macro-F1 = {model_entry['macro_f1']:.4f})",
                fontsize=10.5,
                fontweight="bold",
            )
            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    pct = (cm[i, j] / cm.sum()) * 100.0 if cm.sum() > 0 else 0.0
                    ax.text(
                        j,
                        i,
                        f"{cm[i, j]:,}\n({pct:.1f}%)",
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "#153859",
                        fontsize=10,
                        fontweight="bold",
                    )
            fig4.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig4.suptitle(
            "Confusion Matrices on Locked Held-Out Test Set (N = 7,744, Seed = 42)",
            fontsize=13,
            fontweight="bold",
            y=1.03,
        )
        fig4.tight_layout()
        path4 = self._figures_dir / "confusion_matrices.png"
        fig4.savefig(path4, bbox_inches="tight")
        plt.close(fig4)
        saved_paths["confusion_matrices"] = str(path4)

        return saved_paths

    def plot_clustering_figures(self, clusterer: TopicClusterer) -> dict[str, str]:
        """Render K-Means evaluation, cluster sizes, and 2D SVD projection plots."""
        saved_paths: dict[str, str] = {}
        evals = clusterer.k_evaluations
        summaries = clusterer.cluster_summaries

        # Figure 5: K-Means Silhouette Curve & Cluster Size Breakdown
        fig5, axes5 = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=150)

        ks = [item["k"] for item in evals]
        sils = [item["silhouette_score"] for item in evals]
        inertias = [item["inertia"] for item in evals]

        ax_left = axes5[0]
        ax_left.plot(
            ks,
            sils,
            marker="o",
            linewidth=2.2,
            color=LR_COLOR,
            label="Sampled Silhouette Score (n=1,000)",
        )
        for k_val, s_val in zip(ks, sils):
            ax_left.annotate(
                f"{s_val:.4f}",
                (k_val, s_val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        ax_left.set_xticks(ks)
        ax_left.set_xlabel("Number of Clusters (k)", fontsize=10)
        ax_left.set_ylabel("Silhouette Score", fontsize=10, color=LR_COLOR)
        ax_left.set_title("K-Means Model Selection Across k in {3, 5, 8}", fontsize=11.5, fontweight="bold")
        ax_left.grid(True, linestyle=":", alpha=0.6)

        ax_right_twin = ax_left.twinx()
        ax_right_twin.plot(
            ks,
            inertias,
            marker="s",
            linestyle="--",
            color=FAKE_COLOR,
            alpha=0.75,
            label="Within-Cluster Inertia",
        )
        ax_right_twin.set_ylabel("Inertia (Squared TF-IDF Distance)", fontsize=10, color=FAKE_COLOR)

        cluster_labels = [
            f"C{s.cluster_id + 1}: {', '.join(s.top_terms[:3])}" for s in summaries
        ][::-1]
        cluster_sizes = [s.size for s in summaries][::-1]
        bars = axes5[1].barh(cluster_labels, cluster_sizes, color=NB_COLOR, alpha=0.85)
        for bar, size_val in zip(bars, cluster_sizes):
            axes5[1].text(
                bar.get_width() + 150,
                bar.get_y() + bar.get_height() / 2.0,
                f"{size_val:,}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        axes5[1].set_xlim(0, max(cluster_sizes) * 1.18)
        axes5[1].set_title(
            f"Cluster Sizes & Top Centroid Terms (Selected k = {clusterer.selected_k})",
            fontsize=11.5,
            fontweight="bold",
        )
        axes5[1].set_xlabel("Number of Development Headlines", fontsize=10)
        axes5[1].grid(axis="x", linestyle=":", alpha=0.6)

        fig5.suptitle(
            "Unsupervised Learning: K-Means Lexical Topic Clustering on Development Headlines",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        fig5.tight_layout()
        path5 = self._figures_dir / "kmeans_clusters_analysis.png"
        fig5.savefig(path5, bbox_inches="tight")
        plt.close(fig5)
        saved_paths["kmeans_clusters_analysis"] = str(path5)

        # Figure 6: 2D SVD Projection Scatterplots (by Cluster ID & by Class Label)
        proj = clusterer.projection_2d
        if proj and "x" in proj:
            fig6, axes6 = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150)
            xs = np.asarray(proj["x"])
            ys = np.asarray(proj["y"])
            cids = np.asarray(proj["cluster_id"])
            alabels = np.asarray(proj["audit_label"])

            palette = ["#1b6ca8", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c", "#16a085", "#34495e", "#f1c40f"]
            for s in summaries:
                mask = cids == s.cluster_id
                axes6[0].scatter(
                    xs[mask],
                    ys[mask],
                    s=14,
                    alpha=0.65,
                    color=palette[s.cluster_id % len(palette)],
                    label=f"C{s.cluster_id + 1}: {'/'.join(s.top_terms[:2])}",
                )
            axes6[0].set_title("2D TruncatedSVD Projection Colored by K-Means Cluster", fontsize=11, fontweight="bold")
            axes6[0].set_xlabel("SVD Component 1", fontsize=10)
            axes6[0].set_ylabel("SVD Component 2", fontsize=10)
            axes6[0].legend(fontsize=8.5, loc="best", frameon=True)
            axes6[0].grid(True, linestyle=":", alpha=0.5)

            for lbl_val, lbl_name, col in [(0, "Real News (0)", REAL_COLOR), (1, "Fake News (1)", FAKE_COLOR)]:
                mask = alabels == lbl_val
                axes6[1].scatter(
                    xs[mask],
                    ys[mask],
                    s=14,
                    alpha=0.55,
                    color=col,
                    label=lbl_name,
                )
            axes6[1].set_title("2D TruncatedSVD Projection Colored by Truth Label (Audit)", fontsize=11, fontweight="bold")
            axes6[1].set_xlabel("SVD Component 1", fontsize=10)
            axes6[1].set_ylabel("SVD Component 2", fontsize=10)
            axes6[1].legend(fontsize=9, loc="best", frameon=True)
            axes6[1].grid(True, linestyle=":", alpha=0.5)

            fig6.suptitle(
                "Dimensionality Reduction: 2D Projection of Sampled Development Headlines (n = 1,500)",
                fontsize=13,
                fontweight="bold",
                y=1.02,
            )
            fig6.tight_layout()
            path6 = self._figures_dir / "kmeans_2d_projection.png"
            fig6.savefig(path6, bbox_inches="tight")
            plt.close(fig6)
            saved_paths["kmeans_2d_projection"] = str(path6)

        return saved_paths
