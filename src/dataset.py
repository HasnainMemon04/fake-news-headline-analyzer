"""Dataset loading, schema validation, deduplication, and group-isolated splitting.

Implements the `NewsDataset` class following the stable data contract in
`docs/implementation-plan.md`:
- Preserves row order (`True.csv` label=0, then `Fake.csv` label=1).
- Normalizes titles with Unicode NFKC, casefolding, and word token extraction.
- Removes empty normalized titles, excludes contradictory exact normalized labels,
  and retains one row per exact normalized title.
- Groups near-duplicates by label-blind token-set Jaccard similarity >= 0.90
  into connected components.
- Uses `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` to
  reserve the first fold (7,744 rows) as the locked held-out test set and the
  remaining 30,976 rows as the development set, with zero duplicate-group leakage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import re
from typing import Any
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import SEED

REQUIRED_COLUMNS: frozenset[str] = frozenset({"title", "text", "subject", "date"})
EXPECTED_FILE_SPECS: tuple[tuple[str, int, str], ...] = (
    ("True.csv", 0, "ba0844414a65dc6ae7402b8eee5306da24b6b56488d6767135af466c7dcb2775"),
    ("Fake.csv", 1, "bebf8bcfe95678bf2c732bf413a2ce5f621af0102c82bf08083b2e5d3c693d0c"),
)


def normalize_title(text: str) -> str:
    """Normalize Unicode (NFKC), casefold, and extract alphanumeric word tokens.

    Parameters
    ----------
    text : str
        Raw headline text.

    Returns
    -------
    str
        Space-joined normalized word tokens.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string headline, got {type(text).__name__}")
    normalized_unicode = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", normalized_unicode))


def compute_near_duplicate_groups(
    normalized_titles: list[str],
    similarity_threshold: float = 0.90,
) -> np.ndarray:
    """Group headlines with token-set Jaccard similarity >= threshold.

    Uses global rare-token ordering and prefix filtering with Union-Find so that
    all near-duplicate connected components are placed into the same group without
    requiring an O(N^2) dense similarity matrix. Labels are never consulted.

    Parameters
    ----------
    normalized_titles : list[str]
        List of normalized space-separated token strings.
    similarity_threshold : float, default=0.90
        Minimum Jaccard similarity between token sets to link two titles.

    Returns
    -------
    np.ndarray
        Integer array of component root IDs of length `len(normalized_titles)`.
    """
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0.0, 1.0].")
    if not normalized_titles:
        return np.array([], dtype=np.int64)

    sets = [set(title.split()) for title in normalized_titles]
    frequency = Counter(token for tokens in sets for token in tokens)
    ordered = [sorted(tokens, key=lambda tok: (frequency[tok], tok)) for tokens in sets]
    parent = list(range(len(sets)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    inverted_index: dict[str, list[int]] = defaultdict(list)
    for i in sorted(range(len(sets)), key=lambda idx: (len(sets[idx]), idx)):
        size = len(sets[i])
        if size == 0:
            continue
        prefix_len = size - math.ceil(similarity_threshold * size) + 1
        prefix = ordered[i][:prefix_len]
        min_candidate_len = math.ceil(similarity_threshold * size)
        candidates = {
            j
            for term in prefix
            for j in inverted_index[term]
            if len(sets[j]) >= min_candidate_len
        }
        for j in sorted(candidates):
            intersection = len(sets[i] & sets[j])
            union = size + len(sets[j]) - intersection
            if union > 0 and (intersection / union) >= similarity_threshold:
                parent[find(i)] = find(j)
        for term in prefix:
            inverted_index[term].append(i)

    return np.array([find(i) for i in range(len(sets))], dtype=np.int64)


class NewsDataset:
    """Encapsulated dataset manager for the ISOT Fake News headline corpus.

    Handles loading, schema and checksum auditing, normalization, conflict/duplicate
    removal, near-duplicate component grouping, and deterministic 5-fold stratified
    group splitting into development and locked test partitions.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/raw",
        processed_dir: str | Path | None = None,
        seed: int = SEED,
        similarity_threshold: float = 0.90,
        verify_checksums: bool = False,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._processed_dir = (
            Path(processed_dir)
            if processed_dir is not None
            else self._data_dir.parent / "processed"
        )
        self._seed = int(seed)
        self._similarity_threshold = float(similarity_threshold)
        self._verify_checksums = bool(verify_checksums)

        self._raw_df: pd.DataFrame | None = None
        self._clean_df: pd.DataFrame | None = None
        self._groups: np.ndarray | None = None
        self._train_indices: np.ndarray | None = None
        self._test_indices: np.ndarray | None = None
        self._audit_summary: dict[str, Any] = {}

    @property
    def data_dir(self) -> Path:
        """Directory containing raw `True.csv` and `Fake.csv` files."""
        return self._data_dir

    @property
    def processed_dir(self) -> Path:
        """Directory where `split_manifest.csv` is persisted."""
        return self._processed_dir

    @property
    def seed(self) -> int:
        """Fixed random seed used for deterministic splitting."""
        return self._seed

    @property
    def audit_summary(self) -> dict[str, Any]:
        """Return a copy of the dataset audit and split metadata."""
        if not self._audit_summary:
            raise RuntimeError("Dataset has not been loaded yet. Call load(), clean(), and split() first.")
        return dict(self._audit_summary)

    def load(self) -> pd.DataFrame:
        """Load and validate raw `True.csv` and `Fake.csv` files.

        Returns
        -------
        pd.DataFrame
            Combined raw dataframe with derived binary `label` (0=Real, 1=Fake).

        Raises
        ------
        FileNotFoundError
            If `True.csv` or `Fake.csv` is missing from `data_dir`.
        ValueError
            If required columns are missing, dataset is empty, or checksum fails.
        """
        frames: list[pd.DataFrame] = []
        file_audits: list[dict[str, Any]] = []

        for filename, label, expected_sha256 in EXPECTED_FILE_SPECS:
            filepath = self._data_dir / filename
            if not filepath.is_file():
                raise FileNotFoundError(
                    f"Missing dataset file: '{filepath}'. "
                    f"Run `python tools/download_data.py --data-dir {self._data_dir}` "
                    "to download and verify the local CSV cache."
                )

            raw_bytes = filepath.read_bytes()
            actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            if self._verify_checksums and actual_sha256 != expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for '{filepath}': expected {expected_sha256}, "
                    f"got {actual_sha256}."
                )

            try:
                df = pd.read_csv(filepath, keep_default_na=False)
            except Exception as exc:
                raise ValueError(f"Failed to parse CSV file '{filepath}': {exc}") from exc

            missing_cols = REQUIRED_COLUMNS - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"File '{filename}' is missing required columns: {sorted(missing_cols)}. "
                    f"Found columns: {list(df.columns)}"
                )
            if df.empty:
                raise ValueError(f"File '{filename}' contains zero rows.")

            file_audits.append(
                {
                    "name": filename,
                    "rows": int(len(df)),
                    "columns": list(df.columns),
                    "bytes": int(filepath.stat().st_size),
                    "sha256": actual_sha256,
                    "checksum_matches_official": bool(actual_sha256 == expected_sha256),
                    "blank_titles": int(df["title"].astype(str).str.strip().eq("").sum()),
                    "exact_repeated_titles": int(df["title"].duplicated().sum()),
                }
            )
            df = df.assign(label=int(label), source_file=filename)
            frames.append(df)

        self._raw_df = pd.concat(frames, ignore_index=True)
        self._audit_summary["files"] = file_audits
        self._audit_summary["raw_rows"] = int(len(self._raw_df))
        self._audit_summary["publisher_column_present"] = False
        return self._raw_df.copy()

    def clean(self) -> pd.DataFrame:
        """Normalize titles, drop blank or conflicting titles, and deduplicate.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe of unique normalized titles with length metadata.
        """
        if self._raw_df is None:
            self.load()
        assert self._raw_df is not None

        df = self._raw_df.copy()
        df["normalized"] = df["title"].astype(str).map(normalize_title)

        nonempty_mask = df["normalized"].ne("")
        empty_count = int((~nonempty_mask).sum())
        nonempty = df[nonempty_mask].copy()

        label_nunique = nonempty.groupby("normalized")["label"].nunique()
        conflict_keys = set(label_nunique[label_nunique > 1].index)
        conflict_mask = nonempty["normalized"].isin(conflict_keys)
        conflict_records = int(conflict_mask.sum())

        consistent = nonempty[~conflict_mask]
        clean_df = consistent.drop_duplicates(subset=["normalized"], keep="first").reset_index(drop=True)
        duplicates_removed = int(len(consistent) - len(clean_df))

        clean_df["clean_row_id"] = np.arange(len(clean_df), dtype=np.int64)
        clean_df["char_length"] = clean_df["title"].astype(str).str.len()
        clean_df["word_count"] = clean_df["normalized"].str.split().map(len)

        q1_words, q3_words = np.percentile(clean_df["word_count"], [25, 75])
        iqr_words = q3_words - q1_words
        upper_word_fence = float(q3_words + 1.5 * iqr_words)
        lower_word_fence = float(max(1.0, q1_words - 1.5 * iqr_words))
        outlier_mask = (clean_df["word_count"] > upper_word_fence) | (
            clean_df["word_count"] < lower_word_fence
        )

        self._clean_df = clean_df
        self._audit_summary.update(
            {
                "empty_normalized_titles": empty_count,
                "conflicting_normalized_title_records": conflict_records,
                "normalized_duplicates_removed": duplicates_removed,
                "cleaned_unique_titles": int(len(clean_df)),
                "cleaned_label_counts": {
                    str(k): int(v)
                    for k, v in clean_df["label"].value_counts().sort_index().items()
                },
                "word_count_iqr_bounds": {
                    "q1": float(q1_words),
                    "q3": float(q3_words),
                    "lower_fence": lower_word_fence,
                    "upper_fence": upper_word_fence,
                    "outlier_count_retained": int(outlier_mask.sum()),
                },
            }
        )
        return self._clean_df.copy()

    def split(self, save_manifest: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Group near-duplicates and create the 80/20 stratified group split.

        Parameters
        ----------
        save_manifest : bool, default=True
            If True, writes `split_manifest.csv` to `processed_dir`.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            `(development_df, reserved_test_df)` with zero group overlap.
        """
        if self._clean_df is None:
            self.clean()
        assert self._clean_df is not None

        groups = compute_near_duplicate_groups(
            self._clean_df["normalized"].tolist(),
            similarity_threshold=self._similarity_threshold,
        )
        y = self._clean_df["label"].to_numpy()

        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=self._seed,
        )
        train_idx, test_idx = next(splitter.split(self._clean_df["title"], y, groups))

        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        overlap = len(train_groups & test_groups)
        if overlap != 0:
            raise RuntimeError(f"Group leakage detected: {overlap} groups span train and test splits.")

        self._clean_df["duplicate_group"] = groups
        self._clean_df["partition"] = "reserved_test"
        self._clean_df.loc[train_idx, "partition"] = "development"

        self._groups = groups
        self._train_indices = train_idx
        self._test_indices = test_idx

        if save_manifest:
            self._processed_dir.mkdir(parents=True, exist_ok=True)
            manifest = pd.DataFrame(
                {
                    "clean_row_id": self._clean_df["clean_row_id"],
                    "duplicate_group": groups,
                    "partition": self._clean_df["partition"],
                }
            )
            manifest.to_csv(self._processed_dir / "split_manifest.csv", index=False)

        self._audit_summary.update(
            {
                "near_duplicate_rule": (
                    f"token-set Jaccard >= {self._similarity_threshold:.2f}, "
                    "label-blind connected components"
                ),
                "groups": int(len(set(groups.tolist()))),
                "split": {
                    "development_rows": int(len(train_idx)),
                    "reserved_heldout_rows": int(len(test_idx)),
                    "duplicate_group_overlap": int(overlap),
                    "development_label_counts": {
                        str(k): int(v)
                        for k, v in self._clean_df.iloc[train_idx]["label"]
                        .value_counts()
                        .sort_index()
                        .items()
                    },
                    "reserved_test_label_counts": {
                        str(k): int(v)
                        for k, v in self._clean_df.iloc[test_idx]["label"]
                        .value_counts()
                        .sort_index()
                        .items()
                    },
                },
            }
        )

        dev_df = self._clean_df.iloc[train_idx].reset_index(drop=True)
        test_df = self._clean_df.iloc[test_idx].reset_index(drop=True)
        return dev_df, test_df

    def get_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return `(development_df, reserved_test_df)`, computing them if needed."""
        if self._clean_df is None or self._train_indices is None or self._test_indices is None:
            return self.split(save_manifest=True)
        dev_df = self._clean_df.iloc[self._train_indices].reset_index(drop=True)
        test_df = self._clean_df.iloc[self._test_indices].reset_index(drop=True)
        return dev_df, test_df
