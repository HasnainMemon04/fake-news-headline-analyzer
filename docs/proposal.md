# PROJECT PROPOSAL
FAST - National University of Computer and Emerging Sciences
Karachi Campus | FAST School of Computing | Department of AI and Data Science
AL-2002 Artificial Intelligence Laboratory | Fall 2026 | Section BCY-5A
Instructor: Miss Amma Aslam | Proposal weightage: 1 mark

## Group Information
| Position | Name | Roll number | Section | Planned role |
|---|---|---|---|---|
| Leader | Muhammad Hasnain Memon | 24K-2001 | BCY-5A | Architecture and models |
| Member 2 | Akbar Hussain | 24K-2026 | BCY-5A | Data, EDA and clustering |
| Member 3 | Ateeb Ahmed Qureshi | 24K-2036 | BCY-5A | Interface and explanations |

| Field | Value |
|---|---|
| Project title | Explainable Fake News Headline Analyzer |
| Project idea | Idea No. 19 selected from the official list; not an own idea |
| Category | Part B - Applied and Data Projects |
| GitHub repository URL | https://github.com/HasnainMemon04/fake-news-headline-analyzer |
| Date of submission | Pending instructor announcement; prepared 17 October 2026 |

**Group approval required:** Section 1 limits groups to two, although the guidelines heading and template allow three. All three members are proposed; instructor approval has not been granted. Idea allocation on the BCY-5A group sheet must also be confirmed.

## 1. Project Title and Summary
Explainable Fake News Headline Analyzer will classify English headlines as likely real or likely fake based on a labelled dataset. Students and readers will enter a headline and inspect the prediction and influential words. Two supervised classifiers will learn language patterns that fixed keyword rules cannot reliably describe. Topic clustering will organize similar headlines for exploration. The application will support learning and critical reading, while making clear that a headline classification does not independently establish factual truth.

## 2. Problem Statement and Motivation
Readers encounter headlines without the evidence needed to verify their claims. Short text can omit context, use irony, or repeat the wording of a genuine report, so suspicious words alone cannot determine whether a claim is true. Manually checking every headline requires opening articles, locating original sources, and comparing accounts, which is difficult when many items arrive together. Students and online readers often rely on familiar publishers, search engines, or fact-checking services, but these approaches require time and judgement. This project addresses a narrower, measurable task: predicting the real or fake label assigned to a headline in a public dataset. Vocabulary combinations and uncertain labels make fixed rules inadequate, while sparse text features allow manageable statistical learning. Explanations will reveal the words affecting predictions, and topic exploration will help expose patterns and bias in the data. Building this system is worthwhile because it connects supervised learning, clustering, and honest evaluation to a familiar information problem.

<!-- pagebreak -->

## 3. Problem Formulation
### 3.1 PEAS Description
| Component | Description for this project |
|---|---|
| Performance measure | Primary macro-F1; accuracy, per-class precision, recall and F1, confusion matrices, fit and prediction time. |
| Environment | Labelled English headlines for offline learning; unseen headlines entered into a local interface for inference. |
| Actuators | Display predicted dataset class, model score, signed influential terms and topic assignment. |
| Sensors | Headline text; reference labels available only during training and evaluation, never as inference inputs. |

### 3.2 Environment and Agent Type
| Property | Classification and interpretation |
|---|---|
| Observability | Partially observable: headline visible, supporting evidence and factual truth hidden. |
| Determinism | Stochastic task: text does not uniquely determine truth; a fitted inference pipeline is deterministic. |
| Change over time | Static during a prediction; language and news distributions can drift between datasets. |
| State and actions | Discrete text inputs and label actions; internal TF-IDF weights are continuous. |
| Agents | Single-agent; no strategic opponent. |
| Agent type | Learning agent: offline fit updates parameters using labels; frozen models predict at runtime. |

### 3.3 Problem Representation
| Element | Representation |
|---|---|
| State representation | Validated headline transformed to a sparse, non-negative TF-IDF vector x. |
| Actions / moves | Select y in {0: real, 1: fake}; return explanation and topic. |
| Initial state | Unclassified headline with a saved vocabulary and trained model. |
| Goal or terminal test | Produce a label; during evaluation compare it with the held-out dataset label. |
| Cost, utility or evaluation | Logistic log loss or Naive Bayes likelihood during fit; macro-F1 on unseen records for comparison. |

## 4. Proposed AI Approach
| # | Technique | Role | Own code or library |
|---|---|---|---|
| 1 | TF-IDF + Multinomial Naive Bayes | Probabilistic classifier | scikit-learn; team pipeline |
| 2 | TF-IDF + Logistic Regression | Linear classifier and explanations | scikit-learn; team explanation code |
| 3 | K-Means | Unlabelled topic exploration | scikit-learn; team interpretation |
| B | Majority-class baseline | Minimum comparison | DummyClassifier, most_frequent |

TF-IDF represents short headlines with sparse word weights, keeping training practical on a laptop. Multinomial Naive Bayes combines class priors with word likelihoods under a conditional independence assumption. Logistic Regression learns a weighted sum of features and maps it to a class probability; this provides a contrasting decision rule. For class fake, each term contributes x_j times w_j to its log odds, with the intercept reported separately. Naive Bayes explanations use x_j times the difference between the two class log likelihoods, plus the log prior ratio. Highlighted terms therefore explain the selected model, rather than arbitrary keywords. K-Means minimizes within-cluster squared distance to explore lexical topics without using truth labels. A majority-class predictor tests whether learned patterns improve on simply choosing the common training label.

### 4.1 Planned Class Design
**NewsDataset:** private records/split IDs; load(), clean(), split(). **HeadlineAgent:** abstract base; protected vectorizer/model; fit(), predict(), explain(). **NaiveBayesAgent** and **LogisticAgent** inherit it and override fit() and explain(). **TopicClusterer:** private centroids; fit(), assign(), top_terms(). **ExperimentRunner:** manifest/seed; compare(), export(). Controlled methods encapsulate state; the UI uses the shared agent API.

<!-- pagebreak -->

## 5. Scope
| # | In-scope feature | Why it is essential |
|---|---|---|
| 1 | EDA of lengths, common words and available source metadata | Meets Idea 19; documents missing data, outliers and bias. |
| 2 | Naive Bayes and Logistic Regression on identical test records | Measures two supervised approaches against the baseline. |
| 3 | Highlight signed terms behind each model prediction | Makes decisions inspectable; handles no known terms explicitly. |
| 4 | K-Means topics with top terms and sample headlines | Demonstrates and interprets unsupervised learning. |
| 5 | Local Streamlit headline box and analysis views | Connects predictions, explanations, EDA and topic exploration. |

**Out of scope:** Article-body classification, automated fact verification, live scraping, multilingual models, deep learning, paid APIs and deployment. Model probabilities will be labelled uncalibrated scores, not certainty of truth.

## 6. Data or Environment Source
| Field | Verified source and planned use |
|---|---|
| Source name and URL | ISOT Fake News Dataset, University of Victoria [1]; direct archive [2]. |
| Type | Public dataset; locally loaded CSV files. |
| Access requirements | Public university archive; no API key. Manual Kaggle mirror [3] is a fallback and may require a free account. |
| Approximate size | 44,898 articles before cleaning: 21,417 real and 23,481 fake, per the official description [4]. |
| Required columns | title for input; binary label derived from True.csv/Fake.csv. subject, date and text support auditing only. |
| Licence / terms | Official description requests attribution but states no explicit reuse licence. Do not infer redistribution rights; keep raw text local and confirm terms before sharing data. |
| Offline fallback | Cache manually downloaded CSVs in data/raw; processed splits in data/processed; trained artifacts locally in outputs. Record SHA-256 and source. Once prepared, training and inference require no network. |

**Preparation:** Validate schema and labels; remove empty titles and exact normalized duplicates; exclude conflicting-label duplicates. Group near-duplicate titles before splitting. Inspect unusually short/long records and document retention or removal rules without tuning to test performance. Use title text only as features. Source analysis will distinguish verified publisher metadata from unknown values; subject and input filename are not publishers. Where source IDs exist, add a grouped source holdout; otherwise report the lack of source-independent evidence and Reuters-related bias. Never infer individual fake publishers from class labels.

## 7. Evaluation Plan
| Requirement | Planned experiment |
|---|---|
| What will be compared | Multinomial Naive Bayes, Logistic Regression and majority baseline with the same representation and test IDs. |
| Metric(s) | Accuracy; per-class precision, recall and F1; macro-F1; confusion matrices; fit and batch prediction seconds on the same hardware. |
| Baseline | Most frequent training class; tie rule and label order recorded. |
| Test instances / test set | Approx. 80/20 stratified group split after cleaning; lock 20% away. Duplicate groups cannot cross splits; record final counts and class proportions. |
| Number of runs and seed | random.seed(42), NumPy seed 42, estimator/split random_state=42. Three-fold grouped stratified CV within training only; one final test evaluation per selected model. |
| Planned charts | Labelled model metric bars; confusion matrices with class names; length distributions; frequent terms and source coverage; cluster sizes and top terms. Include a numeric results table. |

Fit TF-IDF inside each CV fold, then refit on training only. Compare small grids for NB alpha and LR C; fix vocabulary cap and document all parameters. Fit K-Means on training features without labels; select k from {3,5,8} using sampled silhouette and readable terms, with seed 42 and n_init=10. Report silhouette, cluster sizes, representative headlines and cautious human topic names. No target accuracy or results are claimed.

<!-- pagebreak -->

## 8. Division of Work
| Member | Modules / components owned | Approximate share |
|---|---|---|
| Muhammad Hasnain Memon - 24K-2001 | Architecture, preprocessing integration, classifiers, evaluation and repository coordination | 34% |
| Akbar Hussain - 24K-2026 | Dataset acquisition, EDA, K-Means clustering and visualizations | 33% |
| Ateeb Ahmed Qureshi - 24K-2036 | Streamlit interface, explanation integration, testing and user documentation | 33% |

These are planned responsibilities, subject to three-member approval. Each member will write and explain their modules, make genuine commits from their own account and review one integration boundary. The final report will record actual contributions and cite external code, datasets and AI assistance.

## 9. Milestones and Timeline
| # | Demonstrable milestone | Target | Owner |
|---|---|---|---|
| 1 | Validated dataset, source audit, split manifest and EDA charts | Week 9 | Akbar; Hasnain integration |
| 2 | Two classifiers and baseline reproducibly trained; CV table exported | Week 10 | Hasnain; Akbar review |
| 3 | Interpreted topic clusters and verified signed word contributions | Weeks 11-12 | Akbar + Ateeb; Hasnain review |
| 4 | Streamlit input, explanations and analysis views run offline; error cases checked | Weeks 13-14 | Ateeb; team integration |
| 5 | Final held-out experiments, three or more charts, report PDF/MD, fresh-clone verification and clean repository | Week 15 | All; leader submits |

Week targets are provisional planning slots; the current teaching week and proposal deadline need instructor confirmation. If fewer weeks remain, compress the first four milestones while retaining the Week 15 submission. Individual viva takes place in submission week; evaluation finishes before the Week 16 Lab Final Exam.

## 10. Expected Outcome and Risks
A working submission will open a local dashboard where users classify English headlines, inspect signed word contributions, and explore topic groups. The report will present measured classifier and baseline scores, confusion matrices, runtime, and interpreted clusters without promising an accuracy threshold. The first risk is learning publisher style or duplicated wording instead of useful headline patterns; duplicate grouping, source auditing, and explicit generalization limits will address it. The second risk is data access or unclear reuse terms; early acquisition, local caching, and a verified mirror will support continuity, while data redistribution remains excluded until permission is clear. Empty input and missing artifacts will produce actionable messages.

### References and access record
[1] University of Victoria ISOT dataset page: https://onlineacademiccommunity.uvic.ca/isot/datasets/
[2] University archive: https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip
[3] Kaggle mirror: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
[4] ISOT official description: https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf
Sources checked 17 October 2026. Cite the dataset papers requested in [4] in the final report. University source documents: AL-2002 Project Guidelines, Project Ideas and Project Proposal Template, Fall 2026, supplied as DOCX.

### For office use - Lab Instructor
| Field | Reserved for instructor |
|---|---|
| Proposal status | |
| Comments | |
| Signature and date | |
