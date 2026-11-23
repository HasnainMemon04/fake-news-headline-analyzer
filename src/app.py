"""Local offline Streamlit dashboard for Explainable Fake News Headline Analyzer.

Run from the repository root:
    python -m streamlit run src/app.py

Loads saved artifacts from `outputs/results/` without fitting or network calls.
Provides interactive headline classification, signed word-level highlighting,
K-Means topic assignment, held-out evaluation tables, and EDA visualizations.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
import streamlit as st

from src.agents import HeadlineAgent
from src.explanations import HeadlineExplanation
from src.topics import TopicClusterer

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "outputs" / "results"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
MODELS_DIR = RESULTS_DIR / "models"

SAMPLE_HEADLINES: dict[str, str] = {
    "Custom input (type below)": "",
    "Sample 1 (Wire-style policy report)": (
        "U.S. Senate passes bipartisan border security and foreign aid bill after late vote"
    ),
    "Sample 2 (Sensational political claim)": (
        "BREAKING: Shocking secret video exposes corrupt FBI plot to rig the election!"
    ),
    "Sample 3 (International diplomacy headline)": (
        "South Korea and Japan agree to share intelligence on North Korean missile tests"
    ),
    "Sample 4 (Clickbait / partisan headline)": (
        "WATCH: Hillary Clinton caught lying to reporters about classified emails scandal"
    ),
}


@st.cache_resource(show_spinner=False)
def load_saved_artifacts(
    models_dir_str: str,
    results_dir_str: str,
) -> dict[str, Any]:
    """Load persisted classifiers, clusterer, and experiment summary from disk."""
    models_dir = Path(models_dir_str)
    results_dir = Path(results_dir_str)

    lr_path = models_dir / "logistic_regression.joblib"
    nb_path = models_dir / "naive_bayes.joblib"
    base_path = models_dir / "majority_baseline.joblib"
    cluster_path = models_dir / "topic_clusterer.joblib"
    summary_path = results_dir / "experiment_results.json"

    missing = [
        str(p)
        for p in (lr_path, nb_path, base_path, cluster_path, summary_path)
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing prepared artifacts: "
            + ", ".join(missing)
            + ". Run `python tools/download_data.py` followed by "
            "`python -m src.experiments --seed 42 --data-dir data/raw --output-dir outputs`."
        )

    return {
        "logistic_regression": HeadlineAgent.load(lr_path),
        "naive_bayes": HeadlineAgent.load(nb_path),
        "majority_baseline": HeadlineAgent.load(base_path),
        "topic_clusterer": TopicClusterer.load(cluster_path),
        "experiment_results": json.loads(summary_path.read_text(encoding="utf8")),
    }


def render_highlighted_headline_html(explanation: HeadlineExplanation) -> str:
    """Format headline words with color-coded badges based on signed contributions."""
    contrib_map = {c.term.lower(): c.signed_contribution for c in explanation.contributions}
    stop_set = set(explanation.stopwords_removed)
    tokens = re.findall(r"\w+|\W+", explanation.headline)

    parts: list[str] = []
    for tok in tokens:
        if not tok.strip() or not re.match(r"^\w+$", tok):
            parts.append(html.escape(tok))
            continue
        low = tok.lower()
        if low in contrib_map:
            val = contrib_map[low]
            if val >= 0:
                bg = "rgba(217, 83, 79, 0.22)"
                border = "#d9534f"
                tag = f"{val:+.3f} Fake"
            else:
                bg = "rgba(27, 108, 168, 0.22)"
                border = "#1b6ca8"
                tag = f"{val:+.3f} Real"
            parts.append(
                f'<span style="background:{bg}; border-bottom:2px solid {border}; '
                f'padding:2px 6px; border-radius:4px; margin:0 2px; display:inline-block;" '
                f'title="Signed contribution: {tag}">'
                f"<strong>{html.escape(tok)}</strong> "
                f'<small style="font-size:0.75em; opacity:0.85;">({val:+.2f})</small></span>'
            )
        elif low in stop_set:
            parts.append(
                f'<span style="color:#6c757d; padding:1px 3px;" title="English stop word (0.00)">'
                f"{html.escape(tok)}</span>"
            )
        else:
            parts.append(
                f'<span style="color:#868e96; border-bottom:1px dashed #adb5bd; padding:1px 3px;" '
                f'title="Out-of-vocabulary token (0.00)">{html.escape(tok)}</span>'
            )

    return (
        '<div style="font-size:1.1rem; line-height:2.0; padding:12px 16px; '
        'background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; color:#212529;">'
        + "".join(parts)
        + "</div>"
    )


def render_single_explanation(explanation: HeadlineExplanation) -> None:
    """Display prediction metrics, highlighted terms, and contribution table."""
    if not explanation.is_valid:
        st.warning(explanation.status_message)
        if explanation.stopwords_removed:
            st.caption(
                "Stop words ignored: " + ", ".join(explanation.stopwords_removed)
            )
        if explanation.unknown_terms:
            st.caption(
                "Out-of-vocabulary terms ignored: " + ", ".join(explanation.unknown_terms)
            )
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Predicted Dataset Class",
        f"{explanation.predicted_class} ({explanation.predicted_label})",
    )
    col2.metric(
        "Uncalibrated P(Fake)",
        f"{(explanation.fake_probability or 0.0) * 100:.1f}%",
    )
    col3.metric(
        explanation.base_score_label,
        f"{explanation.base_score:+.4f}",
    )
    col4.metric(
        "Total Log-Odds (Fake vs Real)",
        f"{explanation.total_log_odds:+.4f}",
        delta=f"Terms sum: {explanation.contribution_sum:+.4f}",
    )

    st.markdown("#### Highlighted Term Contributions in Headline")
    st.caption(
        "Red badges push log-odds toward **Fake (1)**; Blue badges push toward **Real (0)**; "
        "greyed words are stop words or unseen tokens with zero weight."
    )
    st.markdown(render_highlighted_headline_html(explanation), unsafe_allow_html=True)

    rows = [
        {
            "Term": c.term,
            "TF-IDF Weight (x_j)": round(c.tfidf_weight, 4),
            "Model Weight (w_j)": round(c.model_weight, 4),
            "Signed Contribution (x_j * w_j)": round(c.signed_contribution, 4),
            "Direction": c.direction,
        }
        for c in explanation.contributions
    ]
    df_contrib = pd.DataFrame(rows)
    st.markdown("#### Signed Feature Contributions Table")
    st.dataframe(df_contrib, width="stretch", hide_index=True)

    if explanation.stopwords_removed or explanation.unknown_terms:
        st.caption(
            f"Ignored stop words: {explanation.stopwords_removed or 'None'} | "
            f"Out-of-vocabulary tokens: {explanation.unknown_terms or 'None'}"
        )


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Explainable Fake News Headline Analyzer",
        layout="wide",
    )
    st.title("Explainable Fake News Headline Analyzer")
    st.caption(
        "FAST-NUCES Karachi | AL-2002 Artificial Intelligence Lab (Fall 2026, Section BCY-5A) — Idea 19 | "
        "Note: Predictions reflect statistical patterns in the ISOT dataset and do not establish factual truth."
    )

    try:
        artifacts = load_saved_artifacts(str(MODELS_DIR), str(RESULTS_DIR))
    except FileNotFoundError as err:
        st.error(str(err))
        st.info(
            "To prepare the local dataset and train the models offline, run:\n"
            "```powershell\n"
            "python tools/download_data.py\n"
            "python -m src.experiments --seed 42 --data-dir data/raw --output-dir outputs\n"
            "```"
        )
        return

    lr_agent: HeadlineAgent = artifacts["logistic_regression"]
    nb_agent: HeadlineAgent = artifacts["naive_bayes"]
    clusterer: TopicClusterer = artifacts["topic_clusterer"]
    exp_results: dict[str, Any] = artifacts["experiment_results"]
    audit = exp_results.get("dataset_audit", {})

    tab_live, tab_eval, tab_topics, tab_eda = st.tabs(
        [
            "1. Live Headline Analyzer",
            "2. Model Comparison & Evaluation",
            "3. K-Means Topic Clusters",
            "4. Exploratory Data Analysis & Provenance",
        ]
    )

    with tab_live:
        st.subheader("Classify & Explain a News Headline")
        col_preset, col_model = st.columns([2, 1])
        with col_preset:
            preset_choice = st.selectbox(
                "Choose a sample headline or type your own below:",
                options=list(SAMPLE_HEADLINES.keys()),
                index=0,
            )
        with col_model:
            model_choice = st.radio(
                "Select Classifier:",
                options=[
                    "Logistic Regression",
                    "Multinomial Naive Bayes",
                    "Compare Both Side-by-Side",
                ],
                horizontal=False,
            )

        default_text = SAMPLE_HEADLINES[preset_choice]
        headline_input = st.text_input(
            "Headline",
            value=default_text,
            placeholder="Paste an English news headline here...",
        )

        if not headline_input.strip():
            st.info("Enter a headline above or choose a sample to inspect predictions and signed word weights.")
        else:
            if model_choice == "Logistic Regression":
                render_single_explanation(lr_agent.explain(headline_input))
            elif model_choice == "Multinomial Naive Bayes":
                render_single_explanation(nb_agent.explain(headline_input))
            else:
                left_col, right_col = st.columns(2)
                with left_col:
                    st.markdown("### Logistic Regression")
                    render_single_explanation(lr_agent.explain(headline_input))
                with right_col:
                    st.markdown("### Multinomial Naive Bayes")
                    render_single_explanation(nb_agent.explain(headline_input))

            st.divider()
            st.markdown("#### Unsupervised K-Means Topic Assignment")
            topic_info = clusterer.assign(headline_input)
            if not topic_info["is_valid"]:
                st.warning(topic_info["status_message"])
            else:
                st.success(
                    f"**Assigned Lexical Cluster:** {topic_info['label_name']} "
                    f"({topic_info['cluster_size']:,} dev headlines, {topic_info['cluster_percentage']:.1f}% of corpus)"
                )
                st.write("**Top Cluster Centroid Terms:** " + ", ".join(topic_info["top_terms"]))
                with st.expander("View representative development headlines from this topic cluster"):
                    for rep in topic_info["representative_headlines"]:
                        st.markdown(f"- {rep}")

    with tab_eval:
        st.subheader("Held-Out Test Evaluation")
        heldout_count = audit.get("split", {}).get("reserved_heldout_rows")
        run_seed = exp_results.get("seed")
        if isinstance(heldout_count, int) and isinstance(run_seed, int):
            st.caption(f"{heldout_count:,} locked titles · Seed {run_seed}")
        metrics_csv = RESULTS_DIR / "metrics_summary.csv"
        if metrics_csv.is_file():
            st.dataframe(pd.read_csv(metrics_csv), width="stretch", hide_index=True)

        fig_comp = FIGURES_DIR / "model_comparison_metrics.png"
        fig_cm = FIGURES_DIR / "confusion_matrices.png"
        if fig_comp.is_file():
            st.image(str(fig_comp), caption="Held-Out Test Metrics Comparison")
        if fig_cm.is_file():
            st.image(str(fig_cm), caption="Held-Out Confusion Matrices")

        st.markdown("#### 3-Fold Stratified Group CV Grid Search Results (Development Set Only)")
        cv_csv = RESULTS_DIR / "cv_grid_results.csv"
        if cv_csv.is_file():
            st.dataframe(pd.read_csv(cv_csv), width="stretch", hide_index=True)

    with tab_topics:
        st.subheader("Unsupervised K-Means Topic Exploration (Development Corpus)")
        cluster_csv = RESULTS_DIR / "cluster_summary.csv"
        if cluster_csv.is_file():
            st.dataframe(pd.read_csv(cluster_csv), width="stretch", hide_index=True)

        fig_k = FIGURES_DIR / "kmeans_clusters_analysis.png"
        fig_proj = FIGURES_DIR / "kmeans_2d_projection.png"
        if fig_k.is_file():
            st.image(str(fig_k), caption="K-Means Silhouette Selection & Cluster Sizes")
        if fig_proj.is_file():
            st.image(str(fig_proj), caption="2D TruncatedSVD Topic Projection")

    with tab_eda:
        st.subheader("Exploratory Data Analysis & Collection Provenance Audit")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows", f"{audit.get('raw_rows', 0):,}")
        c2.metric("Unique Cleaned Titles", f"{audit.get('cleaned_unique_titles', 0):,}")
        c3.metric("Duplicates Removed", f"{audit.get('normalized_duplicates_removed', 0):,}")
        c4.metric("Near-Duplicate Groups", f"{audit.get('groups', 0):,}")

        fig_len = FIGURES_DIR / "eda_length_distributions.png"
        fig_words = FIGURES_DIR / "eda_top_words_and_provenance.png"
        if fig_len.is_file():
            st.image(str(fig_len), caption="Headline Word & Character Length Distributions")
        if fig_words.is_file():
            st.image(str(fig_words), caption="Top Frequent Words & Source Provenance Audit")


if __name__ == "__main__":
    main()
