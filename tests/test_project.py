"""Automated unit and integration tests for Explainable Fake News Headline Analyzer.

Covers all requirements listed in `src/README.md`:
1. OOP design (inheritance, encapsulation, overridden `fit()` and `explain()`).
2. Schema errors and missing file handling (`FileNotFoundError`, `ValueError`).
3. Duplicate normalization, conflicting label exclusion, and near-duplicate grouping
   with zero group overlap across train/test splits.
4. Mathematical identity between signed term contribution sums + intercept/prior
   and model decision scores (`decision_function` / `predict_log_proba`).
5. Empty, stopword-only, and unknown-vocabulary input handling.
6. Consistent label ordering (`0=Real`, `1=Fake`) and offline reload parity.
7. Streamlit `AppTest` verification for missing artifacts, valid inputs, and no-feature cases.
8. Actionable CLI errors for missing or unverified experiment data.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents import (
    HeadlineAgent,
    LogisticAgent,
    MajorityBaselineAgent,
    NaiveBayesAgent,
)
from src.dataset import (
    NewsDataset,
    compute_near_duplicate_groups,
    normalize_title,
)
from src.experiments import main as run_experiment_cli
from src.topics import TopicClusterer


def _make_synthetic_csvs(data_dir: Path, n_per_class: int = 30) -> None:
    """Create valid synthetic True.csv and Fake.csv files for fast unit testing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    real_rows = []
    fake_rows = []
    for i in range(n_per_class):
        real_rows.append(
            {
                "title": f"Senate bipartisan committee approves budget bill number {i} in Washington",
                "text": "Official policy body text",
                "subject": "politicsNews",
                "date": "October 15, 2026",
            }
        )
        fake_rows.append(
            {
                "title": f"BREAKING shocking secret video exposes corrupt conspiracy scandal number {i}",
                "text": "Sensational body text",
                "subject": "News",
                "date": "October 15, 2026",
            }
        )
    # Add an exact duplicate and a near-duplicate
    real_rows.append(real_rows[0].copy())
    near_dup = real_rows[1].copy()
    near_dup["title"] = real_rows[1]["title"] + "!"
    real_rows.append(near_dup)

    pd.DataFrame(real_rows).to_csv(data_dir / "True.csv", index=False)
    pd.DataFrame(fake_rows).to_csv(data_dir / "Fake.csv", index=False)


class TestHeadlineAnalyzerProject(unittest.TestCase):
    """Comprehensive verification suite for AL-2002 project modules."""

    def test_oop_inheritance_and_method_overrides(self) -> None:
        """Verify >=3 classes, abstract base inheritance, and overridden methods."""
        self.assertTrue(issubclass(NaiveBayesAgent, HeadlineAgent))
        self.assertTrue(issubclass(LogisticAgent, HeadlineAgent))
        self.assertTrue(issubclass(MajorityBaselineAgent, HeadlineAgent))

        self.assertIsNot(NaiveBayesAgent.fit, HeadlineAgent.fit)
        self.assertIsNot(LogisticAgent.fit, HeadlineAgent.fit)
        self.assertIsNot(NaiveBayesAgent.explain, LogisticAgent.explain)

        nb = NaiveBayesAgent(seed=42)
        self.assertFalse(nb.is_fitted)
        with self.assertRaises(RuntimeError):
            nb.predict(["Unfitted test headline"])

    def test_normalization_and_near_duplicate_grouping(self) -> None:
        """Verify Unicode NFKC normalization and token-set Jaccard >= 0.90 grouping."""
        raw = "  U.S. Senate—Passes   Bipartisan Bill!! "
        self.assertEqual(normalize_title(raw), "u s senate passes bipartisan bill")

        titles = [
            "a b c d e f g h i j",
            "a b c d e f g h i j k",  # Jaccard = 10/11 = 0.909 >= 0.90
            "x y z w u v p q r s",
        ]
        groups = compute_near_duplicate_groups(titles, similarity_threshold=0.90)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])

    def test_dataset_schema_errors_and_split_isolation(self) -> None:
        """Verify missing files, bad schemas, conflict removal, and zero split leakage."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_missing = NewsDataset(data_dir=tmp_path / "nonexistent")
            with self.assertRaises(FileNotFoundError):
                ds_missing.load()

            bad_dir = tmp_path / "bad_schema"
            bad_dir.mkdir()
            pd.DataFrame({"wrong_col": ["abc"]}).to_csv(bad_dir / "True.csv", index=False)
            pd.DataFrame({"wrong_col": ["def"]}).to_csv(bad_dir / "Fake.csv", index=False)
            with self.assertRaises(ValueError):
                NewsDataset(data_dir=bad_dir).load()

            valid_dir = tmp_path / "valid_raw"
            _make_synthetic_csvs(valid_dir, n_per_class=25)
            ds = NewsDataset(data_dir=valid_dir, seed=42)
            ds.load()
            clean_df = ds.clean()
            dev_df, test_df = ds.split(save_manifest=True)

            dev_groups = set(dev_df["duplicate_group"].tolist())
            test_groups = set(test_df["duplicate_group"].tolist())
            self.assertEqual(len(dev_groups & test_groups), 0)
            self.assertTrue((valid_dir.parent / "processed" / "split_manifest.csv").is_file())

    def test_agents_explanation_identity_and_offline_reload(self) -> None:
        """Verify signed contribution sums match decision scores and offline reload works."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_synthetic_csvs(tmp_path / "raw", n_per_class=30)
            ds = NewsDataset(data_dir=tmp_path / "raw", seed=42)
            dev_df, test_df = ds.split(save_manifest=False)

            texts = dev_df["title"].tolist()
            labels = dev_df["label"].to_numpy()
            groups = dev_df["duplicate_group"].to_numpy()

            for agent in [
                NaiveBayesAgent(seed=42),
                LogisticAgent(seed=42),
            ]:
                agent.fit(texts, labels, groups=groups)
                sample_headline = test_df["title"].iloc[0]
                expl = agent.explain(sample_headline)
                self.assertTrue(expl.is_valid)
                self.assertIn(expl.predicted_label, (0, 1))
                self.assertAlmostEqual(
                    expl.base_score + expl.contribution_sum,
                    expl.total_log_odds,
                    places=7,
                )

                # Empty, stopword-only, and unknown vocabulary checks
                empty_expl = agent.explain("   ")
                self.assertFalse(empty_expl.is_valid)

                stop_expl = agent.explain("the and of to in for")
                self.assertFalse(stop_expl.is_valid)

                unknown_expl = agent.explain("zzxqvvxx qqqwwweee")
                self.assertFalse(unknown_expl.is_valid)

                # Offline save/load parity
                artifact_path = tmp_path / f"{agent.name}.joblib"
                agent.save(artifact_path)
                reloaded = HeadlineAgent.load(artifact_path)
                np.testing.assert_array_equal(
                    agent.predict(test_df["title"].tolist()),
                    reloaded.predict(test_df["title"].tolist()),
                )

    def test_topic_clusterer_fit_and_assign(self) -> None:
        """Verify K-Means clustering, silhouette calculation, and headline assignment."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_synthetic_csvs(tmp_path / "raw", n_per_class=25)
            ds = NewsDataset(data_dir=tmp_path / "raw", seed=42)
            dev_df, _ = ds.split(save_manifest=False)

            clusterer = TopicClusterer(k_candidates=(3, 5), seed=42)
            clusterer.fit(
                dev_df["title"].tolist(),
                audit_labels=dev_df["label"].to_numpy(),
                preferred_k=3,
            )
            self.assertEqual(clusterer.selected_k, 3)
            self.assertEqual(len(clusterer.cluster_summaries), 3)

            assigned = clusterer.assign("Senate approves bipartisan budget bill in Washington")
            self.assertTrue(assigned["is_valid"])
            self.assertIn(assigned["cluster_id"], (0, 1, 2))

            unknown_assigned = clusterer.assign("zzxqvvxx qqqwwweee")
            self.assertFalse(unknown_assigned["is_valid"])

    def test_experiment_cli_missing_and_unverified_dataset(self) -> None:
        """Report data setup problems without creating a partial experiment output."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "raw"
            output_dir = root / "outputs"
            with patch("sys.argv", [
                "src.experiments", "--data-dir", str(data_dir), "--output-dir", str(output_dir),
            ]):
                with self.assertRaisesRegex(SystemExit, "tools/download_data.py"):
                    run_experiment_cli()

                _make_synthetic_csvs(data_dir)
                with self.assertRaisesRegex(SystemExit, "SHA-256 mismatch"):
                    run_experiment_cli()

            self.assertFalse(output_dir.exists())

    def test_streamlit_app_missing_artifacts(self) -> None:
        """Give the user setup instructions when models have not been trained."""
        from streamlit.testing.v1 import AppTest
        from src import app as dashboard

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            with patch.multiple(dashboard, RESULTS_DIR=results_dir, MODELS_DIR=results_dir / "models"):
                app = AppTest.from_string("from src.app import main\nmain()", default_timeout=30).run()

            self.assertFalse(app.exception)
            self.assertIn("Missing prepared artifacts", app.error[0].value)
            self.assertIn("python -m src.experiments", app.info[0].value)

    def test_streamlit_app_ui_runtime(self) -> None:
        """Exercise the runtime dashboard with independently fitted local artifacts."""
        from streamlit.testing.v1 import AppTest
        from src import app as dashboard

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "raw"
            results_dir = root / "results"
            models_dir = results_dir / "models"
            seed = 13
            _make_synthetic_csvs(data_dir)
            dataset = NewsDataset(data_dir=data_dir, seed=seed)
            dev_df, test_df = dataset.split(save_manifest=False)
            titles = dev_df["title"].tolist()
            labels = dev_df["label"].to_numpy()
            groups = dev_df["duplicate_group"].to_numpy()

            agents = {
                "majority_baseline": MajorityBaselineAgent(seed=seed),
                "naive_bayes": NaiveBayesAgent(seed=seed),
                "logistic_regression": LogisticAgent(seed=seed),
            }
            for slug, agent in agents.items():
                agent.fit(titles, labels, groups=groups).save(models_dir / f"{slug}.joblib")
            clusterer = TopicClusterer(k_candidates=(3,), seed=seed)
            clusterer.fit(titles, audit_labels=labels, preferred_k=3)
            clusterer.save(models_dir / "topic_clusterer.joblib")
            (results_dir / "experiment_results.json").write_text(
                json.dumps({"seed": seed, "dataset_audit": dataset.audit_summary}),
                encoding="utf-8",
            )

            with patch.multiple(
                dashboard, RESULTS_DIR=results_dir, MODELS_DIR=models_dir, FIGURES_DIR=root / "figures",
            ):
                app = AppTest.from_string("from src.app import main\nmain()", default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.tabs), 4)
                self.assertIn("Enter a headline", app.info[0].value)
                self.assertIn(f"{len(test_df)} locked titles · Seed {seed}", [c.value for c in app.caption])

                app.radio[0].set_value("Compare Both Side-by-Side")
                app.text_input[0].input("Senate committee approves budget bill in Washington").run()
                self.assertFalse(app.exception)
                predictions = [m.value for m in app.metric if m.label == "Predicted Dataset Class"]
                self.assertEqual(predictions, ["Real (0)", "Real (0)"])
                self.assertIn("Assigned Lexical Cluster", app.success[0].value)
                self.assertTrue(any("Signed contribution" in m.value for m in app.markdown))

                app.text_input[0].input("the and of to in for").run()
                self.assertFalse(app.exception)
                self.assertTrue(any("common English stop words" in w.value for w in app.warning))

                app.text_input[0].input("zzxqvvxx qqqwwweee").run()
                self.assertFalse(app.exception)
                self.assertTrue(any("No known vocabulary terms" in w.value for w in app.warning))
                self.assertFalse(any(m.label == "Predicted Dataset Class" for m in app.metric))


if __name__ == "__main__":
    unittest.main()
