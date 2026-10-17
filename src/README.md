# Planned source design

No implementation is supplied at proposal stage. Planned modules:

| Module | Responsibility | Owner |
|---|---|---|
| dataset.py | NewsDataset: schema validation, audit, duplicate grouping, immutable split IDs | Akbar acquisition; Hasnain integration |
| agents.py | HeadlineAgent base, NaiveBayesAgent and LogisticAgent, encapsulated fitted state | Hasnain |
| explanations.py | Signed contributions aligned to the selected fitted model and tokenizer | Ateeb; Hasnain coefficient review |
| topics.py | TopicClusterer, train-only fitting, top terms and cluster interpretation | Akbar |
| analysis.py | EDA, grouped statistics and labelled plots | Akbar |
| experiments.py | ExperimentRunner, fixed-seed CV/test workflow and result exports | Hasnain |
| app.py | Streamlit input and analysis views; actionable error messages | Ateeb |

Subclasses override `fit()` and `explain()`. Internal records, vectorizers, fitted estimators and centroids are changed through controlled methods. `predict()` must not fit or learn from new user input. Persist vocabulary, estimator, label mapping, config, dataset hash and split manifest together.

Use docstrings, type annotations, modular names and specific exception handling (`FileNotFoundError`, `ValueError`, etc.). Do not use bare except blocks. Tests should cover empty/unknown-token input, missing artifacts, schema errors, duplicate groups spanning splits, contribution sums versus decision scores, consistent label ordering, and offline reload parity. No tests or model results are claimed complete here.
