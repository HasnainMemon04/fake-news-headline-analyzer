"""Unsupervised K-Means topic clustering and lexical cluster interpretation.

Implements `TopicClusterer` following `docs/implementation-plan.md`:
- Fits strictly on development headlines without using truth labels.
- Evaluates k in {3, 5, 8} with `n_init=10` and `random_state=42`.
- Computes sampled silhouette scores (sample_size=1000, random_state=42) and inertia.
- Extracts top centroid terms, cluster sizes, and representative sample headlines
  closest to each centroid.
- Computes a 2D TruncatedSVD projection for cluster visualization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src import SEED
from src.agents import DEFAULT_VECTORIZER_PARAMS


@dataclass(frozen=True)
class ClusterSummary:
    """Interpreted summary of a single K-Means lexical topic cluster."""

    cluster_id: int
    label_name: str
    size: int
    percentage: float
    top_terms: list[str]
    top_term_weights: list[float]
    representative_headlines: list[str]
    real_share_audit: float
    fake_share_audit: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize cluster summary to a dictionary."""
        return asdict(self)


class TopicClusterer:
    """Encapsulated K-Means topic explorer for news headlines."""

    def __init__(
        self,
        k_candidates: tuple[int, ...] = (3, 5, 8),
        seed: int = SEED,
        silhouette_sample_size: int = 1000,
        vectorizer_params: dict[str, Any] | None = None,
    ) -> None:
        self._k_candidates = tuple(int(k) for k in k_candidates)
        self._seed = int(seed)
        self._silhouette_sample_size = int(silhouette_sample_size)
        self._vectorizer_params = dict(vectorizer_params or DEFAULT_VECTORIZER_PARAMS)

        self._vectorizer: TfidfVectorizer | None = None
        self._selected_k: int | None = None
        self._kmeans: KMeans | None = None
        self._k_evaluations: list[dict[str, Any]] = []
        self._cluster_summaries: list[ClusterSummary] = []
        self._projection_2d: dict[str, Any] = {}
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the topic clusterer has been fitted."""
        return self._is_fitted

    @property
    def selected_k(self) -> int:
        """Selected number of clusters k."""
        self._require_fitted()
        assert self._selected_k is not None
        return self._selected_k

    @property
    def k_evaluations(self) -> list[dict[str, Any]]:
        """Evaluation metrics (silhouette score, inertia) for each candidate k."""
        return list(self._k_evaluations)

    @property
    def cluster_summaries(self) -> list[ClusterSummary]:
        """Interpreted summaries of each cluster for the selected k."""
        return list(self._cluster_summaries)

    @property
    def projection_2d(self) -> dict[str, Any]:
        """Sampled 2D SVD coordinates for scatterplot visualization."""
        return dict(self._projection_2d)

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._vectorizer is None or self._kmeans is None:
            raise RuntimeError(
                "TopicClusterer is not fitted yet. Call fit() or load a saved artifact first."
            )

    def fit(
        self,
        texts: Sequence[str],
        audit_labels: Sequence[int] | np.ndarray | None = None,
        preferred_k: int | None = 5,
    ) -> "TopicClusterer":
        """Fit TF-IDF and evaluate K-Means across candidate k values without labels.

        Parameters
        ----------
        texts : Sequence[str]
            Development headlines used for unsupervised fitting.
        audit_labels : Sequence[int] | np.ndarray | None, optional
            Truth labels used ONLY post-hoc to report class composition per cluster
            (never passed to K-Means or TF-IDF fitting).
        preferred_k : int | None, default=5
            If None, chooses k with highest silhouette score; if specified in
            `k_candidates` and within 0.005 of best silhouette or for interpretability,
            uses `preferred_k` while recording all candidate k evaluations.
        """
        text_list = list(texts)
        if len(text_list) < max(self._k_candidates):
            raise ValueError("Not enough texts to fit K-Means across candidate k values.")

        self._vectorizer = TfidfVectorizer(**self._vectorizer_params)
        vectors = self._vectorizer.fit_transform(text_list)
        feature_names = self._vectorizer.get_feature_names_out()

        models_by_k: dict[int, KMeans] = {}
        labels_by_k: dict[int, np.ndarray] = {}
        self._k_evaluations = []

        sample_n = min(self._silhouette_sample_size, vectors.shape[0])
        for k in self._k_candidates:
            estimator = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=self._seed,
            )
            cluster_labels = estimator.fit_predict(vectors)
            sil_score = float(
                silhouette_score(
                    vectors,
                    cluster_labels,
                    sample_size=sample_n,
                    random_state=self._seed,
                )
            )
            inertia = float(estimator.inertia_)
            counts = np.bincount(cluster_labels, minlength=k)
            top_preview = []
            for cid in range(k):
                center = estimator.cluster_centers_[cid]
                top_idx = np.argsort(center)[::-1][:5]
                top_preview.append(", ".join(str(feature_names[i]) for i in top_idx))

            self._k_evaluations.append(
                {
                    "k": int(k),
                    "silhouette_score": sil_score,
                    "inertia": inertia,
                    "min_cluster_size": int(counts.min()),
                    "max_cluster_size": int(counts.max()),
                    "cluster_top_terms_preview": top_preview,
                }
            )
            models_by_k[k] = estimator
            labels_by_k[k] = cluster_labels

        if preferred_k in models_by_k:
            chosen_k = int(preferred_k)
        else:
            best_entry = max(self._k_evaluations, key=lambda item: item["silhouette_score"])
            chosen_k = int(best_entry["k"])

        self._selected_k = chosen_k
        self._kmeans = models_by_k[chosen_k]
        selected_labels = labels_by_k[chosen_k]

        y_audit = (
            np.asarray(audit_labels, dtype=np.int64)
            if audit_labels is not None
            else None
        )

        summaries: list[ClusterSummary] = []
        total_n = len(text_list)
        for cid in range(chosen_k):
            mask = selected_labels == cid
            member_indices = np.where(mask)[0]
            size = int(len(member_indices))
            pct = float((size / total_n) * 100.0) if total_n > 0 else 0.0

            centroid = self._kmeans.cluster_centers_[cid]
            top_idx = np.argsort(centroid)[::-1][:10]
            top_terms = [str(feature_names[i]) for i in top_idx]
            top_weights = [float(centroid[i]) for i in top_idx]

            descriptor_terms = " / ".join(top_terms[:3]).title()
            label_name = f"Cluster {cid + 1}: {descriptor_terms}"

            if size > 0:
                member_vectors = vectors[member_indices]
                sims = np.asarray(member_vectors @ centroid).ravel()
                best_local = np.argsort(sims)[::-1][:3]
                rep_headlines = [str(text_list[int(member_indices[i])]) for i in best_local]
            else:
                rep_headlines = []

            if y_audit is not None and size > 0:
                cluster_y = y_audit[member_indices]
                fake_share = float(np.mean(cluster_y == 1))
                real_share = float(1.0 - fake_share)
            else:
                fake_share = 0.0
                real_share = 0.0

            summaries.append(
                ClusterSummary(
                    cluster_id=int(cid),
                    label_name=label_name,
                    size=size,
                    percentage=pct,
                    top_terms=top_terms,
                    top_term_weights=top_weights,
                    representative_headlines=rep_headlines,
                    real_share_audit=real_share,
                    fake_share_audit=fake_share,
                )
            )

        self._cluster_summaries = summaries

        # Compute 2D TruncatedSVD projection on a deterministic sample for visualization
        rng = np.random.RandomState(self._seed)
        proj_n = min(1500, total_n)
        sample_idx = rng.choice(total_n, size=proj_n, replace=False)
        svd = TruncatedSVD(n_components=2, random_state=self._seed)
        coords = svd.fit_transform(vectors[sample_idx])
        self._projection_2d = {
            "x": coords[:, 0].astype(float).tolist(),
            "y": coords[:, 1].astype(float).tolist(),
            "cluster_id": selected_labels[sample_idx].astype(int).tolist(),
            "audit_label": (
                y_audit[sample_idx].astype(int).tolist()
                if y_audit is not None
                else [0] * proj_n
            ),
            "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        }

        self._is_fitted = True
        return self

    def assign(self, headline: str) -> dict[str, Any]:
        """Assign an unseen headline to its closest fitted K-Means topic cluster."""
        self._require_fitted()
        assert self._vectorizer is not None and self._kmeans is not None

        if not isinstance(headline, str) or not headline.strip():
            return {
                "is_valid": False,
                "status_message": "Enter a non-empty headline to assign a topic cluster.",
                "cluster_id": None,
                "label_name": None,
                "top_terms": [],
                "representative_headlines": [],
            }

        vec = self._vectorizer.transform([headline.strip()])
        if vec.nnz == 0:
            return {
                "is_valid": False,
                "status_message": "Headline contains no known topic vocabulary terms.",
                "cluster_id": None,
                "label_name": None,
                "top_terms": [],
                "representative_headlines": [],
            }

        cluster_id = int(self._kmeans.predict(vec)[0])
        summary = self._cluster_summaries[cluster_id]
        return {
            "is_valid": True,
            "status_message": "Assigned to nearest K-Means lexical topic centroid.",
            "cluster_id": summary.cluster_id,
            "label_name": summary.label_name,
            "top_terms": summary.top_terms[:8],
            "representative_headlines": summary.representative_headlines,
            "cluster_size": summary.size,
            "cluster_percentage": summary.percentage,
        }

    def save(self, path: str | Path) -> Path:
        """Persist fitted TopicClusterer to disk."""
        self._require_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k_candidates": self._k_candidates,
            "seed": self._seed,
            "silhouette_sample_size": self._silhouette_sample_size,
            "vectorizer_params": self._vectorizer_params,
            "vectorizer": self._vectorizer,
            "selected_k": self._selected_k,
            "kmeans": self._kmeans,
            "k_evaluations": self._k_evaluations,
            "cluster_summaries": [s.to_dict() for s in self._cluster_summaries],
            "projection_2d": self._projection_2d,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TopicClusterer":
        """Load persisted TopicClusterer from disk."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(
                f"Saved TopicClusterer artifact not found at '{source}'. "
                "Run `python -m src.experiments` first."
            )
        payload = joblib.load(source)
        instance = cls(
            k_candidates=tuple(payload["k_candidates"]),
            seed=payload["seed"],
            silhouette_sample_size=payload["silhouette_sample_size"],
            vectorizer_params=payload["vectorizer_params"],
        )
        instance._vectorizer = payload["vectorizer"]
        instance._selected_k = payload["selected_k"]
        instance._kmeans = payload["kmeans"]
        instance._k_evaluations = payload["k_evaluations"]
        instance._cluster_summaries = [
            ClusterSummary(**item) for item in payload["cluster_summaries"]
        ]
        instance._projection_2d = payload.get("projection_2d", {})
        instance._is_fitted = bool(payload["is_fitted"])
        return instance
