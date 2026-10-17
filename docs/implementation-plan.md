# Implementation decisions fixed for the proposal

This plan makes the scope reproducible and addresses data and library compatibility before submission. It does not promise an accuracy threshold or replace final implementation/evaluation.

## Team

| Member | Primary ownership | Planned share |
|---|---|---:|
| Muhammad Hasnain Memon, 24K-2001, Group Leader | Architecture, preprocessing integration, NB/LR, signed word explanations, evaluation and repository coordination | 50% |
| Akbar Hussain, 24K-2026, Member 2 | Data acquisition, EDA, K-Means, plots, Streamlit and user documentation | 50% |

Both collaborate on integration, experiments, testing and final viva preparation. Each makes genuine commits and records actual contributions in the final report.

## Stable data contract

Use the verified ISOT mirror/version and hashes in dataset-access.json and feasibility.json. Input is English title text only. Labels are 0 for True.csv and 1 for Fake.csv. No external API, account, GPU or deep-learning model is needed for inference. Download is a one-time preparation step.

Duplicate normalization uses Unicode NFKC, case folding and word tokens. Remove empty normalized titles, exclude contradictory exact normalized labels, and retain one row per exact normalized title. Group token-set Jaccard similarity >= 0.90 as connected components, without labels. The feasibility script validates its efficient prefix-filter grouping against brute force on small examples.

Preserve row order (True.csv then Fake.csv). Use StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), reserving the first fold for final testing. Preserve its IDs. Use three-fold stratified grouped CV within development. Seed Python random, NumPy and applicable estimators with 42. Record hardware, versions, rules and class proportions.

## Stable feature and model contract

- TF-IDF: lowercase word unigrams, max_features=20000, min_df=2, English stop words and L2 normalization. 20,000 is a cap, not a promised vocabulary size. Fit within each CV training fold.
- MultinomialNB: alpha in {0.5, 1.0}; identical CV folds for comparison.
- LogisticRegression: solver="liblinear", max_iter=1000, random_state=42; C in {0.5, 1.0, 2.0}. Check convergence and record settings.
- DummyClassifier: strategy="most_frequent", using training labels only.
- Select settings by development CV macro-F1, refit on development and evaluate once on the locked test. Selecting values within declared grids is part of the experiment, not a new feature/model family.
- LR explanation: fake-class log odds = intercept + sum(TF-IDF weight times coefficient). NB: log prior ratio + sum(TF-IDF times class log-likelihood differences). Highlight signed terms for the selected model and show intercept/prior separately. Stop words and unknown words have no feature contribution.
- K-Means: development features, without labels; k in {3,5,8}, n_init=10, random_state=42. Sample up to 1,000 for silhouette; inspect centroid terms and representative headlines. Human names describe lexical topic groups, not verified semantics.

## Application and OOP contract

Implement NewsDataset, abstract HeadlineAgent with NB/LR subclasses overriding methods, TopicClusterer and ExperimentRunner. Encapsulate state with controlled methods. Streamlit loads saved pipelines; input never triggers fitting. Include input, model selection, score, term highlighting and topic/EDA views. Scores are uncalibrated. Empty/stopword-only/unknown input gets an explanatory message; missing data/artifacts gets setup instructions.

Exclude deployment, paid APIs, article-body features, live fact verification and deep learning. Source EDA reports documented provenance and missing publisher coverage. It cannot invent unavailable publisher identities or establish source-independent performance.

## Final evidence

The separate tools/verify_feasibility.py spike records pre-submission checks. Final work still needs the complete class design and interface, meaningful error/offline tests, accuracy/precision/recall/F1/runtime tables, confusion matrices and at least three labelled charts. Explain clusters and mistakes. Complete PDF/MD report and fresh-clone verification by Week 15. Both members must explain their own modules in the individual viva.
