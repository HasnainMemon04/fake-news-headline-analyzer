"""End-to-end reproducible experiment runner for the Fake News Headline Analyzer.

CLI entrypoint:
    python -m src.experiments --seed 42 --data-dir data/raw --output-dir outputs

Implements `ExperimentRunner` following `docs/implementation-plan.md` and
`ml-best-practices`:
- Seeds Python `random`, `numpy`, and all estimators with `seed=42`.
- Executes `NewsDataset` validation, deduplication, and 5-fold stratified group split.
- Trains `MajorityBaselineAgent`, `NaiveBayesAgent`, and `LogisticAgent` strictly
  on the 30,976 development headlines using 3-fold grouped CV.
- Evaluates once on the locked 7,744 held-out test records, computing Accuracy,
  per-class Precision/Recall/F1, Macro-F1, 95% bootstrap confidence intervals,
  confusion matrices, slice-based error analysis, and runtime benchmarks.
- Fits `TopicClusterer` (K-Means for k in {3, 5, 8}) on development headlines.
- Saves metrics tables, JSON summaries, 6 labelled figures, and local model artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import random
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src import LABEL_NAMES, SEED
from src.agents import (
    HeadlineAgent,
    LogisticAgent,
    MajorityBaselineAgent,
    NaiveBayesAgent,
)
from src.analysis import EDAAnalyzer
from src.dataset import NewsDataset
from src.topics import TopicClusterer


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 500,
    seed: int = SEED,
) -> dict[str, tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for Accuracy and Macro-F1."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    acc_samples = np.empty(n_bootstraps, dtype=np.float64)
    f1_samples = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        indices = rng.randint(0, n, size=n)
        yt = y_true[indices]
        yp = y_pred[indices]
        acc_samples[i] = accuracy_score(yt, yp)
        f1_samples[i] = f1_score(yt, yp, average="macro", zero_division=0)

    return {
        "accuracy_95_ci": (
            float(np.percentile(acc_samples, 2.5)),
            float(np.percentile(acc_samples, 97.5)),
        ),
        "macro_f1_95_ci": (
            float(np.percentile(f1_samples, 2.5)),
            float(np.percentile(f1_samples, 97.5)),
        ),
    }


def evaluate_length_slices(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    """Evaluate model performance across headline word-count slices."""
    word_counts = test_df["word_count"].to_numpy()
    slices = [
        ("Short (<= 8 words)", word_counts <= 8),
        ("Medium (9-14 words)", (word_counts >= 9) & (word_counts <= 14)),
        ("Long (>= 15 words)", word_counts >= 15),
    ]
    results: list[dict[str, Any]] = []
    for name, mask in slices:
        count = int(np.sum(mask))
        if count == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        results.append(
            {
                "slice": name,
                "count": count,
                "accuracy": float(accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            }
        )
    return results


class ExperimentRunner:
    """Orchestrates reproducible training, evaluation, clustering, and reporting."""

    def __init__(
        self,
        data_dir: str | Path = "data/raw",
        output_dir: str | Path = "outputs",
        seed: int = SEED,
        *,
        verify_checksums: bool = False,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._output_dir = Path(output_dir)
        self._figures_dir = self._output_dir / "figures"
        self._results_dir = self._output_dir / "results"
        self._models_dir = self._results_dir / "models"
        self._seed = int(seed)
        self._verify_checksums = verify_checksums

    def run(self) -> dict[str, Any]:
        """Execute the full experiment pipeline and persist all artifacts."""
        overall_start = perf_counter()
        random.seed(self._seed)
        np.random.seed(self._seed)

        # 1. Load, clean, and split dataset
        dataset = NewsDataset(
            data_dir=self._data_dir,
            seed=self._seed,
            verify_checksums=self._verify_checksums,
        )
        dataset.load()
        clean_df = dataset.clean()
        dev_df, test_df = dataset.split(save_manifest=True)
        audit_summary = dataset.audit_summary

        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir.mkdir(parents=True, exist_ok=True)

        # 2. Exploratory Data Analysis & EDA Charts
        eda_analyzer = EDAAnalyzer(output_figures_dir=self._figures_dir)
        eda_summary = eda_analyzer.compute_eda(
            clean_df=clean_df,
            dev_df=dev_df,
            audit_meta=audit_summary,
        )
        eda_figures = eda_analyzer.plot_eda_figures(clean_df=clean_df)

        # 3. Train and Evaluate Baseline + Two Supervised Classifiers
        dev_texts = dev_df["title"].astype(str).tolist()
        dev_labels = dev_df["label"].to_numpy()
        dev_groups = dev_df["duplicate_group"].to_numpy()

        test_texts = test_df["title"].astype(str).tolist()
        test_labels = test_df["label"].to_numpy()

        agents: list[ tuple[str, HeadlineAgent] ] = [
            ("majority_baseline", MajorityBaselineAgent(seed=self._seed)),
            ("naive_bayes", NaiveBayesAgent(alphas=(0.5, 1.0), seed=self._seed)),
            ("logistic_regression", LogisticAgent(c_values=(0.5, 1.0, 2.0), seed=self._seed)),
        ]

        metrics_by_model: list[dict[str, Any]] = []
        cv_records: list[dict[str, Any]] = []
        error_examples: dict[str, list[dict[str, Any]]] = {}

        for slug, agent in agents:
            fit_start = perf_counter()
            agent.fit(dev_texts, dev_labels, groups=dev_groups)
            fit_seconds = float(perf_counter() - fit_start)

            pred_start = perf_counter()
            test_preds = agent.predict(test_texts)
            pred_seconds = float(perf_counter() - pred_start)

            report = classification_report(
                test_labels,
                test_preds,
                labels=[0, 1],
                target_names=["Real", "Fake"],
                output_dict=True,
                zero_division=0,
            )
            cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
            ci = compute_bootstrap_ci(test_labels, test_preds, n_bootstraps=500, seed=self._seed)
            slices = evaluate_length_slices(test_df, test_labels, test_preds)

            model_entry = {
                "slug": slug,
                "model_name": agent.name,
                "best_params": agent.best_params,
                "vocabulary_size": agent.vocabulary_size,
                "accuracy": float(accuracy_score(test_labels, test_preds)),
                "accuracy_95_ci": list(ci["accuracy_95_ci"]),
                "macro_f1": float(report["macro avg"]["f1-score"]),
                "macro_f1_95_ci": list(ci["macro_f1_95_ci"]),
                "macro_precision": float(report["macro avg"]["precision"]),
                "macro_recall": float(report["macro avg"]["recall"]),
                "real_precision": float(report["Real"]["precision"]),
                "real_recall": float(report["Real"]["recall"]),
                "real_f1": float(report["Real"]["f1-score"]),
                "fake_precision": float(report["Fake"]["precision"]),
                "fake_recall": float(report["Fake"]["recall"]),
                "fake_f1": float(report["Fake"]["f1-score"]),
                "confusion_matrix": cm.astype(int).tolist(),
                "fit_seconds": round(fit_seconds, 4),
                "predict_seconds": round(pred_seconds, 4),
                "slice_metrics": slices,
            }
            metrics_by_model.append(model_entry)

            for cv_row in agent.cv_results:
                cv_records.append({"model_name": agent.name, **cv_row})

            # Collect representative error analysis examples for NB and LR
            if slug in ("naive_bayes", "logistic_regression"):
                fp_indices = np.where((test_labels == 0) & (test_preds == 1))[0][:3]
                fn_indices = np.where((test_labels == 1) & (test_preds == 0))[0][:3]
                samples: list[dict[str, Any]] = []
                for err_type, idxs in [("False Positive (Real predicted as Fake)", fp_indices),
                                       ("False Negative (Fake predicted as Real)", fn_indices)]:
                    for idx in idxs:
                        title_str = str(test_texts[int(idx)])
                        expl = agent.explain(title_str)
                        top_terms = [
                            {
                                "term": c.term,
                                "signed_contribution": round(c.signed_contribution, 4),
                                "direction": c.direction,
                            }
                            for c in expl.contributions[:5]
                        ]
                        samples.append(
                            {
                                "error_type": err_type,
                                "headline": title_str,
                                "true_label": LABEL_NAMES[int(test_labels[int(idx)])],
                                "predicted_label": expl.predicted_class,
                                "fake_probability": round(float(expl.fake_probability or 0.0), 4),
                                "top_influential_terms": top_terms,
                            }
                        )
                error_examples[slug] = samples

            agent.save(self._models_dir / f"{slug}.joblib")

        # 4. Generate Model Comparison and Confusion Matrix Charts
        eval_figures = eda_analyzer.plot_model_comparison_and_confusion(metrics_by_model)

        # 5. Unsupervised K-Means Topic Clustering on Development Headlines
        clusterer = TopicClusterer(k_candidates=(3, 5, 8), seed=self._seed)
        clusterer.fit(dev_texts, audit_labels=dev_labels, preferred_k=5)
        clusterer.save(self._models_dir / "topic_clusterer.joblib")
        cluster_figures = eda_analyzer.plot_clustering_figures(clusterer)

        # 6. Export CSV tables and comprehensive JSON artifact
        metrics_df = pd.DataFrame(
            [
                {
                    "Model": m["model_name"],
                    "Selected Hyperparameters": json.dumps(m["best_params"]),
                    "Accuracy": round(m["accuracy"], 4),
                    "Macro-F1": round(m["macro_f1"], 4),
                    "Real Precision": round(m["real_precision"], 4),
                    "Real Recall": round(m["real_recall"], 4),
                    "Real F1": round(m["real_f1"], 4),
                    "Fake Precision": round(m["fake_precision"], 4),
                    "Fake Recall": round(m["fake_recall"], 4),
                    "Fake F1": round(m["fake_f1"], 4),
                    "Train Time (s)": m["fit_seconds"],
                    "Predict Time (s)": m["predict_seconds"],
                }
                for m in metrics_by_model
            ]
        )
        metrics_df.to_csv(self._results_dir / "metrics_summary.csv", index=False)

        if cv_records:
            pd.DataFrame(cv_records).to_csv(
                self._results_dir / "cv_grid_results.csv", index=False
            )

        cluster_df = pd.DataFrame(
            [
                {
                    "Cluster ID": s.cluster_id + 1,
                    "Topic Descriptor": s.label_name,
                    "Size (Dev)": s.size,
                    "Share (%)": round(s.percentage, 2),
                    "Top 8 Terms": ", ".join(s.top_terms[:8]),
                    "Representative Headline": (
                        s.representative_headlines[0] if s.representative_headlines else ""
                    ),
                    "Post-hoc Real Share (%)": round(s.real_share_audit * 100.0, 1),
                    "Post-hoc Fake Share (%)": round(s.fake_share_audit * 100.0, 1),
                }
                for s in clusterer.cluster_summaries
            ]
        )
        cluster_df.to_csv(self._results_dir / "cluster_summary.csv", index=False)

        all_figures = {**eda_figures, **eval_figures, **cluster_figures}
        full_results = {
            "project": "Explainable Fake News Headline Analyzer",
            "course": "AL-2002 Artificial Intelligence Laboratory (Fall 2026)",
            "seed": self._seed,
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "scikit_learn_version": sklearn.__version__,
            },
            "dataset_audit": audit_summary,
            "eda_summary": eda_summary,
            "cv_grid_results": cv_records,
            "heldout_test_metrics": metrics_by_model,
            "error_analysis_examples": error_examples,
            "kmeans_evaluation": {
                "selected_k": clusterer.selected_k,
                "k_candidates_evaluated": clusterer.k_evaluations,
                "clusters": [s.to_dict() for s in clusterer.cluster_summaries],
            },
            "figures": all_figures,
            "total_elapsed_seconds": round(perf_counter() - overall_start, 2),
        }

        results_json_path = self._results_dir / "experiment_results.json"
        results_json_path.write_text(
            json.dumps(full_results, indent=2) + "\n", encoding="utf8"
        )
        return full_results


def main() -> None:
    """Parse CLI arguments and run all project experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing True.csv and Fake.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write figures/ and results/",
    )
    parser.add_argument(
        "--allow-unverified-data",
        action="store_true",
        help="Allow custom CSVs whose SHA-256 digests differ from the pinned ISOT files",
    )
    args = parser.parse_args()

    try:
        runner = ExperimentRunner(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            verify_checksums=not args.allow_unverified_data,
        )
        results = runner.run()
        print(
            f"Completed experiments in {results['total_elapsed_seconds']}s. "
            f"Results saved to '{args.output_dir / 'results'}' and figures to '{args.output_dir / 'figures'}'."
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Experiment execution failed: {exc}") from exc


if __name__ == "__main__":
    main()
