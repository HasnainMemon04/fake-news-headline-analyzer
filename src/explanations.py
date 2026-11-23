"""Signed feature contribution arithmetic and explainability helpers.

Aligns term-level explanations directly to the fitted TF-IDF vectorizer and
classifier parameters as specified in `docs/implementation-plan.md`:
- Logistic Regression:
    log_odds(Fake) = intercept + sum_j (tfidf_j * coef_j)
- Multinomial Naive Bayes:
    log_odds(Fake) = log(P(y=1)/P(y=0)) + sum_j (tfidf_j * (log P(w_j|y=1) - log P(w_j|y=0)))

Stop words and out-of-vocabulary tokens have zero contribution and are reported
explicitly so the user interface can distinguish unknown vocabulary from empty input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src import LABEL_NAMES

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


@dataclass(frozen=True)
class TermContribution:
    """Single vocabulary term's contribution to the Fake vs. Real log-odds."""

    term: str
    tfidf_weight: float
    model_weight: float
    signed_contribution: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize term contribution to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class HeadlineExplanation:
    """Complete model-specific explanation for a single input headline."""

    headline: str
    model_name: str
    is_valid: bool
    status_message: str
    predicted_label: int | None
    predicted_class: str | None
    fake_probability: float | None
    real_probability: float | None
    base_score: float
    base_score_label: str
    contribution_sum: float
    total_log_odds: float
    known_terms_count: int
    contributions: list[TermContribution] = field(default_factory=list)
    stopwords_removed: list[str] = field(default_factory=list)
    unknown_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize explanation to a JSON-compatible dictionary."""
        data = asdict(self)
        data["contributions"] = [c.to_dict() for c in self.contributions]
        return data


def tokenize_headline_words(headline: str) -> list[str]:
    """Extract lowercase word tokens matching scikit-learn's default token pattern."""
    if not isinstance(headline, str):
        raise TypeError(f"Expected string headline, got {type(headline).__name__}")
    return TOKEN_PATTERN.findall(headline.lower())


def explain_logistic_prediction(
    headline: str,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    model_name: str = "Logistic Regression",
) -> HeadlineExplanation:
    """Compute exact log-odds decomposition for a fitted Logistic Regression model.

    Parameters
    ----------
    headline : str
        Input headline string.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    model : LogisticRegression
        Fitted binary LogisticRegression estimator.
    model_name : str, default="Logistic Regression"
        Human-readable model identifier.

    Returns
    -------
    HeadlineExplanation
        Exact additive decomposition where `base_score + contribution_sum`
        equals `model.decision_function(x)` within floating-point tolerance.
    """
    if not isinstance(headline, str):
        raise TypeError(f"Headline must be a string, got {type(headline).__name__}")

    intercept = float(model.intercept_[0])
    stripped = headline.strip()
    if not stripped:
        return HeadlineExplanation(
            headline=headline,
            model_name=model_name,
            is_valid=False,
            status_message="Please enter a non-empty English news headline.",
            predicted_label=None,
            predicted_class=None,
            fake_probability=None,
            real_probability=None,
            base_score=intercept,
            base_score_label="Intercept (b)",
            contribution_sum=0.0,
            total_log_odds=intercept,
            known_terms_count=0,
        )

    tokens = tokenize_headline_words(stripped)
    vocab: dict[str, int] = vectorizer.vocabulary_
    stopwords_found = sorted({t for t in tokens if t in ENGLISH_STOP_WORDS})
    unknown_found = sorted(
        {t for t in tokens if t not in ENGLISH_STOP_WORDS and t not in vocab}
    )

    vector = vectorizer.transform([stripped])
    if vector.nnz == 0:
        if tokens and all(t in ENGLISH_STOP_WORDS for t in tokens):
            reason = (
                "Headline contains only common English stop words. "
                "Please enter a fuller headline with descriptive terms."
            )
        else:
            reason = (
                "No known vocabulary terms found in this headline. "
                "Try a fuller English news headline."
            )
        return HeadlineExplanation(
            headline=headline,
            model_name=model_name,
            is_valid=False,
            status_message=reason,
            predicted_label=None,
            predicted_class=None,
            fake_probability=None,
            real_probability=None,
            base_score=intercept,
            base_score_label="Intercept (b)",
            contribution_sum=0.0,
            total_log_odds=intercept,
            known_terms_count=0,
            stopwords_removed=stopwords_found,
            unknown_terms=unknown_found,
        )

    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    nonzero_indices = vector.indices
    nonzero_weights = vector.data

    contributions: list[TermContribution] = []
    for idx, tfidf_val in zip(nonzero_indices, nonzero_weights):
        w_j = float(coefs[idx])
        x_j = float(tfidf_val)
        signed = x_j * w_j
        direction = "Fake (+)" if signed >= 0.0 else "Real (-)"
        contributions.append(
            TermContribution(
                term=str(feature_names[idx]),
                tfidf_weight=x_j,
                model_weight=w_j,
                signed_contribution=signed,
                direction=direction,
            )
        )

    contributions.sort(key=lambda c: abs(c.signed_contribution), reverse=True)
    contribution_sum = float(sum(c.signed_contribution for c in contributions))
    total_log_odds = intercept + contribution_sum
    decision_val = float(model.decision_function(vector)[0])
    if not np.isclose(total_log_odds, decision_val, atol=1e-7):
        raise ValueError(
            f"Logistic explanation mismatch: sum={total_log_odds} vs decision={decision_val}"
        )

    probs = model.predict_proba(vector)[0]
    pred_label = int(model.predict(vector)[0])
    return HeadlineExplanation(
        headline=headline,
        model_name=model_name,
        is_valid=True,
        status_message="Prediction explained using signed log-odds feature contributions.",
        predicted_label=pred_label,
        predicted_class=LABEL_NAMES[pred_label],
        fake_probability=float(probs[1]),
        real_probability=float(probs[0]),
        base_score=intercept,
        base_score_label="Intercept (b)",
        contribution_sum=contribution_sum,
        total_log_odds=total_log_odds,
        known_terms_count=len(contributions),
        contributions=contributions,
        stopwords_removed=stopwords_found,
        unknown_terms=unknown_found,
    )


def explain_naive_bayes_prediction(
    headline: str,
    vectorizer: TfidfVectorizer,
    model: MultinomialNB,
    model_name: str = "Multinomial Naive Bayes",
) -> HeadlineExplanation:
    """Compute exact log-posterior odds decomposition for Multinomial Naive Bayes.

    Parameters
    ----------
    headline : str
        Input headline string.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    model : MultinomialNB
        Fitted binary MultinomialNB estimator.
    model_name : str, default="Multinomial Naive Bayes"
        Human-readable model identifier.

    Returns
    -------
    HeadlineExplanation
        Exact additive decomposition where `base_score + contribution_sum`
        equals `log P(y=1|x) - log P(y=0|x)` within floating-point tolerance.
    """
    if not isinstance(headline, str):
        raise TypeError(f"Headline must be a string, got {type(headline).__name__}")

    log_prior_ratio = float(model.class_log_prior_[1] - model.class_log_prior_[0])
    stripped = headline.strip()
    if not stripped:
        return HeadlineExplanation(
            headline=headline,
            model_name=model_name,
            is_valid=False,
            status_message="Please enter a non-empty English news headline.",
            predicted_label=None,
            predicted_class=None,
            fake_probability=None,
            real_probability=None,
            base_score=log_prior_ratio,
            base_score_label="Log Prior Ratio log(P(Fake)/P(Real))",
            contribution_sum=0.0,
            total_log_odds=log_prior_ratio,
            known_terms_count=0,
        )

    tokens = tokenize_headline_words(stripped)
    vocab: dict[str, int] = vectorizer.vocabulary_
    stopwords_found = sorted({t for t in tokens if t in ENGLISH_STOP_WORDS})
    unknown_found = sorted(
        {t for t in tokens if t not in ENGLISH_STOP_WORDS and t not in vocab}
    )

    vector = vectorizer.transform([stripped])
    if vector.nnz == 0:
        if tokens and all(t in ENGLISH_STOP_WORDS for t in tokens):
            reason = (
                "Headline contains only common English stop words. "
                "Please enter a fuller headline with descriptive terms."
            )
        else:
            reason = (
                "No known vocabulary terms found in this headline. "
                "Try a fuller English news headline."
            )
        return HeadlineExplanation(
            headline=headline,
            model_name=model_name,
            is_valid=False,
            status_message=reason,
            predicted_label=None,
            predicted_class=None,
            fake_probability=None,
            real_probability=None,
            base_score=log_prior_ratio,
            base_score_label="Log Prior Ratio log(P(Fake)/P(Real))",
            contribution_sum=0.0,
            total_log_odds=log_prior_ratio,
            known_terms_count=0,
            stopwords_removed=stopwords_found,
            unknown_terms=unknown_found,
        )

    feature_names = vectorizer.get_feature_names_out()
    log_likelihood_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    nonzero_indices = vector.indices
    nonzero_weights = vector.data

    contributions: list[TermContribution] = []
    for idx, tfidf_val in zip(nonzero_indices, nonzero_weights):
        w_j = float(log_likelihood_diff[idx])
        x_j = float(tfidf_val)
        signed = x_j * w_j
        direction = "Fake (+)" if signed >= 0.0 else "Real (-)"
        contributions.append(
            TermContribution(
                term=str(feature_names[idx]),
                tfidf_weight=x_j,
                model_weight=w_j,
                signed_contribution=signed,
                direction=direction,
            )
        )

    contributions.sort(key=lambda c: abs(c.signed_contribution), reverse=True)
    contribution_sum = float(sum(c.signed_contribution for c in contributions))
    total_log_odds = log_prior_ratio + contribution_sum

    log_proba = model.predict_log_proba(vector)[0]
    actual_log_odds = float(log_proba[1] - log_proba[0])
    if not np.isclose(total_log_odds, actual_log_odds, atol=1e-7):
        raise ValueError(
            f"Naive Bayes explanation mismatch: sum={total_log_odds} vs actual={actual_log_odds}"
        )

    probs = model.predict_proba(vector)[0]
    pred_label = int(model.predict(vector)[0])
    return HeadlineExplanation(
        headline=headline,
        model_name=model_name,
        is_valid=True,
        status_message="Prediction explained using log-likelihood ratios and class prior.",
        predicted_label=pred_label,
        predicted_class=LABEL_NAMES[pred_label],
        fake_probability=float(probs[1]),
        real_probability=float(probs[0]),
        base_score=log_prior_ratio,
        base_score_label="Log Prior Ratio log(P(Fake)/P(Real))",
        contribution_sum=contribution_sum,
        total_log_odds=total_log_odds,
        known_terms_count=len(contributions),
        contributions=contributions,
        stopwords_removed=stopwords_found,
        unknown_terms=unknown_found,
    )
