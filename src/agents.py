"""Object-oriented classification agents for headline authenticity prediction.

Satisfies Labs 01-03 OOP and Agent requirements:
- Abstract base class `HeadlineAgent` encapsulating protected `_vectorizer`,
  `_model`, `_pipeline`, `_best_params`, and `_cv_results`.
- Concrete subclasses `NaiveBayesAgent`, `LogisticAgent`, and
  `MajorityBaselineAgent` inheriting from `HeadlineAgent` and overriding
  `fit()` and `explain()`.
- Strict separation between offline learning (`fit()`) and runtime inference
  (`predict()`, `predict_proba()`, `explain()`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src import LABEL_NAMES, SEED
from src.explanations import (
    HeadlineExplanation,
    explain_logistic_prediction,
    explain_naive_bayes_prediction,
)

DEFAULT_VECTORIZER_PARAMS: dict[str, Any] = {
    "max_features": 20_000,
    "ngram_range": (1, 1),
    "min_df": 2,
    "stop_words": "english",
    "lowercase": True,
    "norm": "l2",
}


class HeadlineAgent(ABC):
    """Abstract base learning agent for headline classification and explanation."""

    def __init__(
        self,
        name: str,
        seed: int = SEED,
        vectorizer_params: dict[str, Any] | None = None,
    ) -> None:
        self._name = str(name)
        self._seed = int(seed)
        self._vectorizer_params = dict(vectorizer_params or DEFAULT_VECTORIZER_PARAMS)
        self._pipeline: Pipeline | None = None
        self._best_params: dict[str, Any] = {}
        self._cv_results: list[dict[str, Any]] = []
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        """Human-readable agent name."""
        return self._name

    @property
    def is_fitted(self) -> bool:
        """Whether the agent has been trained on development data."""
        return self._is_fitted

    @property
    def best_params(self) -> dict[str, Any]:
        """Selected hyperparameter configuration from development CV."""
        return dict(self._best_params)

    @property
    def cv_results(self) -> list[dict[str, Any]]:
        """Cross-validation grid results from development training."""
        return list(self._cv_results)

    @property
    def vocabulary_size(self) -> int:
        """Number of fitted TF-IDF vocabulary features (0 if not applicable)."""
        self._require_fitted()
        assert self._pipeline is not None
        if "tfidf" in self._pipeline.named_steps:
            return len(self._pipeline.named_steps["tfidf"].vocabulary_)
        return 0

    def _require_fitted(self) -> None:
        """Raise a descriptive RuntimeError if inference is called before fitting."""
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError(
                f"Agent '{self._name}' is not fitted yet. "
                "Call fit() or load a saved model artifact before predicting."
            )

    @abstractmethod
    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int] | np.ndarray,
        groups: Sequence[int] | np.ndarray | None = None,
    ) -> "HeadlineAgent":
        """Fit the agent on development headlines using grouped cross-validation."""

    @abstractmethod
    def explain(self, headline: str) -> HeadlineExplanation:
        """Return a signed feature contribution explanation for one headline."""

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Predict binary dataset labels (0=Real, 1=Fake) without refitting."""
        self._require_fitted()
        assert self._pipeline is not None
        return np.asarray(self._pipeline.predict(list(texts)), dtype=np.int64)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return uncalibrated class probability estimates [P(Real), P(Fake)]."""
        self._require_fitted()
        assert self._pipeline is not None
        return np.asarray(self._pipeline.predict_proba(list(texts)), dtype=np.float64)

    def save(self, path: str | Path) -> Path:
        """Persist the fitted agent state to disk with joblib."""
        self._require_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent_class": self.__class__.__name__,
            "name": self._name,
            "seed": self._seed,
            "vectorizer_params": self._vectorizer_params,
            "pipeline": self._pipeline,
            "best_params": self._best_params,
            "cv_results": self._cv_results,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "HeadlineAgent":
        """Load a persisted `HeadlineAgent` subclass from disk."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(
                f"Saved model artifact not found at '{source}'. "
                "Run `python -m src.experiments --seed 42 --data-dir data/raw --output-dir outputs` first."
            )
        payload = joblib.load(source)
        registry: dict[str, type[HeadlineAgent]] = {
            "NaiveBayesAgent": NaiveBayesAgent,
            "LogisticAgent": LogisticAgent,
            "MajorityBaselineAgent": MajorityBaselineAgent,
        }
        class_name = payload.get("agent_class", "")
        if class_name not in registry:
            raise ValueError(f"Unknown agent class '{class_name}' in '{source}'.")
        agent_cls = registry[class_name]
        instance = agent_cls(
            seed=payload["seed"],
            vectorizer_params=payload.get("vectorizer_params"),
        )
        instance._name = payload["name"]
        instance._pipeline = payload["pipeline"]
        instance._best_params = payload["best_params"]
        instance._cv_results = payload["cv_results"]
        instance._is_fitted = bool(payload["is_fitted"])
        return instance


class NaiveBayesAgent(HeadlineAgent):
    """Multinomial Naive Bayes headline classifier with TF-IDF features."""

    def __init__(
        self,
        alphas: tuple[float, ...] = (0.5, 1.0),
        seed: int = SEED,
        vectorizer_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name="Multinomial Naive Bayes",
            seed=seed,
            vectorizer_params=vectorizer_params,
        )
        self._alphas = tuple(float(a) for a in alphas)

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int] | np.ndarray,
        groups: Sequence[int] | np.ndarray | None = None,
    ) -> "NaiveBayesAgent":
        """Select smoothing `alpha` via 3-fold grouped CV and refit on development data."""
        y = np.asarray(labels, dtype=np.int64)
        text_list = list(texts)
        if len(text_list) == 0 or len(text_list) != len(y):
            raise ValueError("Texts and labels must be non-empty and of equal length.")

        if groups is None:
            group_arr = np.arange(len(text_list), dtype=np.int64)
        else:
            group_arr = np.asarray(groups, dtype=np.int64)

        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=self._seed)
        splits = list(inner_cv.split(text_list, y, group_arr))

        base_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**self._vectorizer_params)),
                ("model", MultinomialNB()),
            ]
        )
        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid={"model__alpha": list(self._alphas)},
            scoring="f1_macro",
            cv=splits,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(text_list, y)

        self._pipeline = search.best_estimator_
        self._best_params = {"alpha": float(search.best_params_["model__alpha"])}
        self._cv_results = [
            {
                "alpha": float(params["model__alpha"]),
                "mean_cv_f1_macro": float(mean_score),
                "std_cv_f1_macro": float(std_score),
                "rank": int(rank),
            }
            for params, mean_score, std_score, rank in zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"],
                search.cv_results_["rank_test_score"],
            )
        ]
        self._is_fitted = True
        return self

    def explain(self, headline: str) -> HeadlineExplanation:
        """Override explanation using Naive Bayes log-likelihood ratios and class prior."""
        self._require_fitted()
        assert self._pipeline is not None
        vectorizer: TfidfVectorizer = self._pipeline.named_steps["tfidf"]
        model: MultinomialNB = self._pipeline.named_steps["model"]
        return explain_naive_bayes_prediction(
            headline=headline,
            vectorizer=vectorizer,
            model=model,
            model_name=self._name,
        )


class LogisticAgent(HeadlineAgent):
    """Logistic Regression headline classifier with signed log-odds explanations."""

    def __init__(
        self,
        c_values: tuple[float, ...] = (0.5, 1.0, 2.0),
        seed: int = SEED,
        vectorizer_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name="Logistic Regression",
            seed=seed,
            vectorizer_params=vectorizer_params,
        )
        self._c_values = tuple(float(c) for c in c_values)

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int] | np.ndarray,
        groups: Sequence[int] | np.ndarray | None = None,
    ) -> "LogisticAgent":
        """Select regularization `C` via 3-fold grouped CV and refit on development data."""
        y = np.asarray(labels, dtype=np.int64)
        text_list = list(texts)
        if len(text_list) == 0 or len(text_list) != len(y):
            raise ValueError("Texts and labels must be non-empty and of equal length.")

        if groups is None:
            group_arr = np.arange(len(text_list), dtype=np.int64)
        else:
            group_arr = np.asarray(groups, dtype=np.int64)

        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=self._seed)
        splits = list(inner_cv.split(text_list, y, group_arr))

        base_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**self._vectorizer_params)),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1000,
                        random_state=self._seed,
                    ),
                ),
            ]
        )
        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid={"model__C": list(self._c_values)},
            scoring="f1_macro",
            cv=splits,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(text_list, y)

        self._pipeline = search.best_estimator_
        fitted_lr: LogisticRegression = self._pipeline.named_steps["model"]
        n_iter = int(np.max(fitted_lr.n_iter_))
        if n_iter >= 1000:
            raise RuntimeError("LogisticRegression failed to converge within max_iter=1000.")

        self._best_params = {
            "C": float(search.best_params_["model__C"]),
            "solver": "liblinear",
            "max_iter": 1000,
            "iterations_used": n_iter,
        }
        self._cv_results = [
            {
                "C": float(params["model__C"]),
                "mean_cv_f1_macro": float(mean_score),
                "std_cv_f1_macro": float(std_score),
                "rank": int(rank),
            }
            for params, mean_score, std_score, rank in zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"],
                search.cv_results_["rank_test_score"],
            )
        ]
        self._is_fitted = True
        return self

    def explain(self, headline: str) -> HeadlineExplanation:
        """Override explanation using Logistic Regression coefficients and intercept."""
        self._require_fitted()
        assert self._pipeline is not None
        vectorizer: TfidfVectorizer = self._pipeline.named_steps["tfidf"]
        model: LogisticRegression = self._pipeline.named_steps["model"]
        return explain_logistic_prediction(
            headline=headline,
            vectorizer=vectorizer,
            model=model,
            model_name=self._name,
        )


class MajorityBaselineAgent(HeadlineAgent):
    """Majority-class DummyClassifier baseline agent for minimum comparison."""

    def __init__(
        self,
        seed: int = SEED,
        vectorizer_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name="Majority Class Baseline",
            seed=seed,
            vectorizer_params=vectorizer_params,
        )

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int] | np.ndarray,
        groups: Sequence[int] | np.ndarray | None = None,
    ) -> "MajorityBaselineAgent":
        """Fit a `DummyClassifier(strategy='most_frequent')` on development labels."""
        y = np.asarray(labels, dtype=np.int64)
        text_list = list(texts)
        if len(text_list) == 0 or len(text_list) != len(y):
            raise ValueError("Texts and labels must be non-empty and of equal length.")

        self._pipeline = Pipeline(
            [
                ("model", DummyClassifier(strategy="most_frequent", random_state=self._seed)),
            ]
        )
        self._pipeline.fit(text_list, y)
        majority_label = int(self._pipeline.predict([text_list[0]])[0])
        self._best_params = {
            "strategy": "most_frequent",
            "majority_class_label": majority_label,
            "majority_class_name": LABEL_NAMES[majority_label],
        }
        self._cv_results = []
        self._is_fitted = True
        return self

    def explain(self, headline: str) -> HeadlineExplanation:
        """Return a baseline explanation (constant training prior, no word weights)."""
        self._require_fitted()
        assert self._pipeline is not None
        probs = self.predict_proba([headline])[0]
        pred = int(self.predict([headline])[0])
        return HeadlineExplanation(
            headline=headline,
            model_name=self._name,
            is_valid=bool(headline.strip()),
            status_message="Majority baseline always predicts the most frequent training class.",
            predicted_label=pred,
            predicted_class=LABEL_NAMES[pred],
            fake_probability=float(probs[1]),
            real_probability=float(probs[0]),
            base_score=0.0,
            base_score_label="Constant Majority Rule",
            contribution_sum=0.0,
            total_log_odds=0.0,
            known_terms_count=0,
        )
