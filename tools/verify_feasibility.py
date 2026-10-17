"""Pre-submission feasibility check, separate from final project experiments.

Uses local ISOT CSVs only. Reserves an outer holdout and never evaluates it.
No final report scores or trained artifacts are published by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import tempfile
from time import perf_counter
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest


SEED = 42
SIMILARITY = 0.90
VECTORIZER = dict(max_features=20_000, ngram_range=(1, 1), min_df=2,
                  stop_words="english", lowercase=True, norm="l2")


def normalized_title(text: str) -> str:
    """Normalize Unicode, case and non-word punctuation for duplicate auditing."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def near_duplicate_groups(titles: list[str]) -> np.ndarray:
    """Group token-set Jaccard >= 0.90 with length and rare-prefix filtering.

    Global token ordering and prefix filtering enumerate candidates without an
    all-pairs dense similarity matrix. Connected components stay in one split.
    Labels are never used to form similarity groups.
    """
    sets = [set(x.split()) for x in titles]
    frequency = Counter(token for tokens in sets for token in tokens)
    ordered = [sorted(tokens, key=lambda t: (frequency[t], t)) for tokens in sets]
    parent = list(range(len(sets)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    index: dict[str, list[int]] = defaultdict(list)
    for i in sorted(range(len(sets)), key=lambda j: (len(sets[j]), j)):
        size = len(sets[i])
        prefix = ordered[i][:size - math.ceil(SIMILARITY * size) + 1]
        candidates = {j for term in prefix for j in index[term]
                      if len(sets[j]) >= math.ceil(SIMILARITY * size)}
        for j in sorted(candidates):
            intersection = len(sets[i] & sets[j])
            union = size + len(sets[j]) - intersection
            if intersection / union >= SIMILARITY:
                parent[find(i)] = find(j)
        for term in prefix:
            index[term].append(i)
    return np.array([find(i) for i in range(len(sets))])


def assert_grouping() -> None:
    """Check prefix grouping against brute force on a small varied corpus."""
    rng = random.Random(SEED)
    words = [f"word{i}" for i in range(30)]
    examples = [" ".join(rng.sample(words, rng.randint(3, 15))) for _ in range(100)]
    examples += [x + " extraword" for x in examples[:20]] + examples[:10]
    groups = near_duplicate_groups(examples)
    sets = [set(x.split()) for x in examples]
    for i in range(len(sets)):
        for j in range(i):
            if len(sets[i] & sets[j]) / len(sets[i] | sets[j]) >= SIMILARITY:
                assert groups[i] == groups[j], "Near-duplicate pair escaped grouping"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("docs/feasibility.json"))
    args = parser.parse_args()
    started = perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    assert_grouping()
    records, files = [], []
    for name, label in [("True.csv", 0), ("Fake.csv", 1)]:
        path = args.data_dir / name
        frame = pd.read_csv(path, keep_default_na=False)
        required = {"title", "text", "subject", "date"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{name}: expected columns {sorted(required)}")
        files.append(dict(name=name, rows=len(frame), columns=list(frame.columns),
                          bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          blank_titles=int(frame.title.str.strip().eq("").sum()),
                          exact_repeated_titles=int(frame.title.duplicated().sum())))
        frame = frame.assign(label=label)
        records.append(frame)
    frame = pd.concat(records, ignore_index=True)
    frame["normalized"] = frame.title.map(normalized_title)
    nonempty = frame[frame.normalized.ne("")].copy()
    conflicts = nonempty.groupby("normalized").label.nunique()
    conflict_keys = conflicts[conflicts > 1].index
    conflict_records = int(nonempty.normalized.isin(conflict_keys).sum())
    clean = nonempty[~nonempty.normalized.isin(conflict_keys)].drop_duplicates("normalized").reset_index(drop=True)
    groups = near_duplicate_groups(clean.normalized.tolist())
    y = clean.label.to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, heldout = next(outer.split(clean.title, y, groups))
    assert not (set(groups[train]) & set(groups[heldout]))
    # The outer heldout labels are used only for split construction/counts.
    # No prediction, threshold selection or scoring uses that partition.
    dev = clean.iloc[train].reset_index(drop=True)
    dev_groups = groups[train]
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    fit, validation = next(inner.split(dev.title, dev.label, dev_groups))
    assert not (set(dev_groups[fit]) & set(dev_groups[validation]))
    texts_fit = dev.title.iloc[fit]
    texts_validation = dev.title.iloc[validation]
    y_fit = dev.label.iloc[fit].to_numpy()
    y_validation = dev.label.iloc[validation].to_numpy()
    checks = {}
    # Cache membership only; no held-out evaluation is performed.
    manifest = pd.DataFrame({"clean_row_id": np.arange(len(clean)), "duplicate_group": groups,
                             "partition": "reserved_test"})
    manifest.loc[train, "partition"] = "development"
    processed = args.data_dir.parent / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(processed / "split_manifest.csv", index=False)
    with tempfile.TemporaryDirectory(prefix="headline-feasibility-") as temporary:
        for name, model in [("naive_bayes", MultinomialNB(alpha=1.0)),
                            ("logistic_regression", LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=SEED))]:
            pipeline = Pipeline([("tfidf", TfidfVectorizer(**VECTORIZER)), ("model", model)])
            pipeline.fit(texts_fit, y_fit)
            predicted = pipeline.predict(texts_validation)
            report = classification_report(y_validation, predicted, output_dict=True, zero_division=0)
            assert np.isfinite(report["macro avg"]["f1-score"])
            assert confusion_matrix(y_validation, predicted, labels=[0, 1]).shape == (2, 2)
            feature = pipeline["tfidf"].transform(texts_validation.iloc[:50])
            if name == "logistic_regression":
                explained = np.asarray(feature @ model.coef_[0]).ravel() + model.intercept_[0]
                actual = model.decision_function(feature)
            else:
                weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
                explained = np.asarray(feature @ weights).ravel() + model.class_log_prior_[1] - model.class_log_prior_[0]
                log_prob = model.predict_log_proba(feature)
                actual = log_prob[:, 1] - log_prob[:, 0]
            assert np.allclose(explained, actual, atol=1e-9)
            model_path = Path(temporary) / f"{name}.joblib"
            joblib.dump(pipeline, model_path)
            restored = joblib.load(model_path)
            assert np.array_equal(restored.predict(texts_validation), predicted)
            # Unknown input must be detected by the final UI before predicting.
            assert pipeline["tfidf"].transform(["", "zzxqvvxx zzxyyqq"]).nnz == 0
            parameter = "model__alpha" if name == "naive_bayes" else "model__C"
            candidates = [0.5, 1.0] if name == "naive_bayes" else [0.5, 1.0, 2.0]
            search = GridSearchCV(pipeline, {parameter: candidates}, scoring="f1_macro",
                                  cv=list(inner.split(dev.title, dev.label, dev_groups)),
                                  n_jobs=1, error_score="raise")
            search.fit(dev.title, dev.label)
            assert np.isfinite(search.cv_results_["mean_test_score"]).all()
            checks[name] = dict(fit="pass", validation_metrics_computable="pass",
                                signed_contribution_identity="pass", offline_reload_parity="pass",
                                empty_unknown_vector_detection="pass", grouped_training_cv_grid="pass",
                                features=len(pipeline["tfidf"].vocabulary_))
        baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(fit), 1)), y_fit)
        assert len(baseline.predict(np.zeros((len(validation), 1)))) == len(validation)
        checks["majority_baseline"] = "pass"
        vectors = TfidfVectorizer(**VECTORIZER).fit_transform(texts_fit)
        clusters = {}
        for k in [3, 5, 8]:
            estimator = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            labels = estimator.fit_predict(vectors)
            score = silhouette_score(vectors, labels, sample_size=min(1000, len(labels)), random_state=SEED)
            assert np.isfinite(score)
            assert len(np.unique(labels)) == k
            assert estimator.cluster_centers_.shape[1] == vectors.shape[1]
            clusters[str(k)] = dict(fit="pass", silhouette_computable="pass",
                                    nonempty_clusters=k, top_terms_available="pass")
        checks["kmeans"] = clusters
        # Exercise Streamlit's real test runtime with a persisted classifier.
        # This is an integration spike, not the final course interface.
        app_code = f'''import joblib
import streamlit as st
pipeline = joblib.load({str(model_path)!r})
title = st.text_input("Headline")
if title.strip():
    vector = pipeline["tfidf"].transform([title])
    if vector.nnz == 0:
        st.warning("No known vocabulary terms. Try a fuller English headline.")
    else:
        st.write("fake" if int(pipeline.predict([title])[0]) == 1 else "real")
else:
    st.info("Enter a headline.")
'''
        app = AppTest.from_string(app_code).run(timeout=30)
        assert not app.exception
        app.text_input[0].input(texts_validation.iloc[0]).run(timeout=30)
        assert not app.exception
        app.text_input[0].input("zzxqvvxx zzxyyqq").run(timeout=30)
        assert not app.exception and len(app.warning) == 1
        checks["streamlit_saved_model_input_and_unknown_terms"] = "pass"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(purpose="Pre-submission feasibility only; no final test results",
                  verified_on="2026-10-17", seed=SEED,
                  files=files, raw_rows=len(frame),
                  empty_normalized_titles=len(frame)-len(nonempty),
                  conflicting_normalized_title_records=conflict_records,
                  cleaned_unique_titles=len(clean),
                  normalized_duplicates_removed=len(nonempty)-conflict_records-len(clean),
                  cleaned_label_counts={str(k):int(v) for k,v in clean.label.value_counts().sort_index().items()},
                  near_duplicate_rule="token-set Jaccard >= 0.90, label-blind connected components",
                  groups=len(set(groups)),
                  split=dict(development_rows=len(train), reserved_heldout_rows=len(heldout),
                             reserved_heldout_evaluated=False,
                             feasibility_fit_rows=len(fit), feasibility_validation_rows=len(validation),
                             duplicate_group_overlap=0),
                  publisher_column_present=False,
                  vectorizer=VECTORIZER, checks=checks,
                  elapsed_seconds=round(perf_counter()-started, 2))
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
