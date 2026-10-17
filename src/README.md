# Planned source design

Final course modules remain planned. The separate tools/verify_feasibility.py spike validates feasibility and is not a final project implementation. Planned modules:

| Module | Responsibility | Owner |
|---|---|---|
| dataset.py | NewsDataset: schema validation, audit, duplicate grouping, immutable split IDs | Akbar acquisition; Hasnain integration |
| agents.py | HeadlineAgent base, NaiveBayesAgent and LogisticAgent, encapsulated fitted state | Hasnain |
| explanations.py | Signed contributions aligned to the selected fitted model and tokenizer | Ateeb integration; Hasnain coefficient review |
| topics.py | TopicClusterer, train-only fitting, top terms and cluster interpretation | Akbar |
| analysis.py | EDA, grouped statistics and labelled plots | Akbar |
| experiments.py | ExperimentRunner, fixed-seed CV/test workflow and result exports | Hasnain |
| app.py | Streamlit input and analysis views; actionable error messages | Ateeb |

Subclasses override `fit()` and `explain()`. Internal records, vectorizers, fitted estimators and centroids are changed through controlled methods. `predict()` must not fit or learn from new user input. Persist vocabulary, estimator, label mapping, config, dataset hash and split manifest together.

Use docstrings, type annotations, modular names and specific exception handling (`FileNotFoundError`, `ValueError`, etc.). Do not use bare except blocks. Tests should cover empty/unknown-token input, missing artifacts, schema errors, duplicate groups spanning splits, contribution sums versus decision scores, consistent label ordering, and offline reload parity. All three members share integration, experiments, testing and final viva preparation, subject to instructor approval of the proposed group size. The separate feasibility record lists completed pre-submission checks; final held-out evaluation and course application tests remain future work.

Team: Muhammad Hasnain Memon (24K-2001), Akbar Hussain (24K-2026), and Ateeb Ahmed Qureshi (24K-2036).
