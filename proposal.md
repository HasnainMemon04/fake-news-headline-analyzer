# FAST National University of Computer & Emerging Sciences
**FAST School of Computing, Karachi Campus - Department of AI & Data Science**  
**AL-2002 Artificial Intelligence (Laboratory) - Fall 2026**

# Project Proposal
**Project:** Explainable Fake News Headline Analyzer  
**Instructor:** Miss Amma Aslam  
**Section:** BCY-5A  
**Submission date:** 17 October 2026

## Group Information
| Position | Name | Roll number | Section | Primary responsibility |
|---|---|---|---|---|
| Leader | Muhammad Hasnain Memon | 24K-2001 | BCY-5A | Data pipeline & integration |
| Member 2 | Akbar Hussain | 24K-2026 | BCY-5A | Classifiers & explanations |
| Member 3 | Ateeb Ahmed Qureshi | 24K-2036 | BCY-5A | Clustering, EDA & UI |

| Field | Value |
|---|---|
| Project title | Explainable Fake News Headline Analyzer |
| Project idea | **[x] Idea No. 19** from Project Ideas list; [ ] Own idea |
| Category | [x] Part B - Applied / Data; [ ] Part A - Game |
| GitHub repository URL | **Pending - insert public repository URL before submission** |

> **Administrative approval needed:** The proposal template has a Member 3 row, while the separate project guidelines allow at most two members. Approval from Miss Amma Aslam is required for a three-member team.

## 1. Project Title and Summary
This project proposes an interactive application that classifies English news headlines as likely real or likely fake using machine learning. Users will enter a headline in a simple web interface and receive a prediction, an estimated confidence score and the words that most influenced the result. The system will compare Naive Bayes and Logistic Regression using TF-IDF text features, while K-Means clustering will explore common headline topics. Unlike fixed keyword rules, trained models learn patterns from labelled examples and can evaluate new combinations of words. The application is intended as an educational decision-support demonstration, not a substitute for independent fact-checking.

## 2. Problem Statement and Motivation
Misleading headlines can circulate quickly, while readers usually have limited time to check each claim against trusted reporting. A headline is short and often ambiguous: the same words can appear in genuine reports, satire, opinion or fabricated content. A fixed list of suspicious phrases therefore fails to capture the range of possible writing styles. Even a moderate vocabulary of 10,000 terms creates an enormous number of possible word combinations, making manual rule-writing or brute-force checking impractical. Students, online readers and content reviewers face this uncertainty and often rely on intuition or time-consuming external searches. We will frame the task as supervised binary classification, learning statistical associations between headline text and labels in an existing dataset. We will also inspect clusters of related headlines and expose influential words, so results are interpretable. Because a headline alone cannot establish truth, predictions will be presented as model estimates rather than verified facts. The project is worthwhile because it demonstrates explainable AI and careful evaluation on a socially relevant problem.

## 3. Problem Formulation
### 3.1 PEAS Description
| Component | Description |
|---|---|
| Performance measure | Macro-F1, class-wise precision and recall, accuracy, false positives/negatives, and reproducibility. |
| Environment | Offline English-headline dataset during training; user-submitted headline and saved model in the web application. |
| Actuators | Display a label, estimated class probabilities, influential words, topic grouping and charts. |
| Sensors | Headline text entered by the user; headline and reference label in training/test records. |

### 3.2 Environment and Agent Type
| Property | Classification | Reason |
|---|---|---|
| Observability | Partially observable | Headline visible; underlying factual truth not directly observed. |
| Determinism | Stochastic | Observed labels and language patterns are uncertain. |
| Change over time | Static | Text unchanged during a prediction. |
| State and actions | Discrete | Tokens, classes and displayed actions are discrete. |
| Agents | Single-agent | One classifier makes each prediction. |
| Agent type | Learning | Learns parameters from labelled examples. |

### 3.3 Problem Representation
| Element | Representation |
|---|---|
| State representation | Cleaned headline and sparse TF-IDF feature vector. |
| Actions / moves | Predict real or fake; return estimated confidence and feature contributions. |
| Initial state | Unclassified headline, or labelled input during training. |
| Goal or terminal test | Output binary prediction; measure against held-out reference labels. |
| Cost / evaluation | Primary macro-F1; secondary precision, recall and confusion matrix. |

## 4. Proposed AI Approach
| # | Technique | Role | Own code or library |
|---|---|---|---|
| 1 | TF-IDF + Multinomial Naive Bayes | First supervised classifier | scikit-learn; own pipeline |
| 2 | TF-IDF + Logistic Regression | Second classifier + word contributions | scikit-learn; own pipeline |
| 3 | K-Means clustering | Find topic groups without class labels | scikit-learn; own interpretation |
| B | Majority-class dummy baseline | Lower benchmark | scikit-learn DummyClassifier |

TF-IDF converts the observed headline into a numerical representation of informative words, which suits short, sparse text. Multinomial Naive Bayes gives an easy-to-explain probabilistic classifier using word evidence, while Logistic Regression provides a contrasting linear decision model. Both will be trained and tested under identical splits so that their results are comparable. For explanations, we will display the highest contributing TF-IDF word weights from the fitted linear model, identifying the direction of the prediction rather than claiming causation. K-Means will group similar headlines without using class labels, and top cluster terms will help interpret possible topics. A majority-class classifier provides a transparent lower benchmark. The main evaluation criteria will be macro-F1, precision, recall and confusion matrices, with accuracy as a supplementary measure. All preprocessing and model settings will be recorded for repeatable experiments.

### 4.1 Planned Class Design
- **NewsDataset:** loads, validates, de-duplicates, and partitions headlines.
- **HeadlineAgent (abstract):** common `fit()` and `predict()` interface.
- **NaiveBayesAgent** and **LogisticAgent:** inherit and override `fit()`/`predict()`.
- **TopicClusterer:** fits K-Means and returns interpretable topic terms.
- **ExperimentRunner:** manages consistent comparisons, results and charts.

## 5. Scope
| # | In-scope feature | Why it is essential |
|---|---|---|
| 1 | Cleaned labelled headlines + EDA | Quantifies distributions, lengths, words and sources. |
| 2 | TF-IDF + Naive Bayes vs Logistic Regression | Implements and compares two learning models. |
| 3 | Explainable predictions | Highlights model-influential words alongside predictions. |
| 4 | K-Means topic discovery | Groups unlabelled headline representations. |
| 5 | Streamlit application + results dashboard | Interactive input, metrics, clusters and visual evidence. |

**Out of scope:** Real-time news scraping, live online fact-checking, article-body verification, non-English classification, transformer/deep-learning systems, paid APIs and guaranteed truth detection.

## 6. Data or Environment Source
| Field | Plan |
|---|---|
| Source name and URL | ISOT Fake News Dataset, University of Victoria; public Kaggle mirror: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset |
| Type | [x] Public dataset; [ ] Web API; [ ] Self-collected; [ ] Generated / simulated |
| Access requirements | Free Kaggle account may be required for manual download; a free API token is needed for scripted downloads. No paid key. |
| Approximate size | About 45,000 English news articles in `Fake.csv` and `True.csv`; title field only. |
| Licence / terms of use | Kaggle listing: CC BY-NC-SA 4.0. Attribute sources; non-commercial use and share-alike requirements; verify before publishing a subset. |
| Offline fallback | Licence-compliant cached labelled subset (target: 1,000 headlines) and fitted demo models locally; no network in ordinary operation; full raw dataset ignored by Git. |

## 7. Evaluation Plan
| Requirement | Plan |
|---|---|
| What will be compared | TF-IDF + Naive Bayes vs TF-IDF + Logistic Regression vs majority-class dummy. |
| Metric(s) | Primary macro-F1; secondary accuracy, per-class precision/recall, confusion matrix and training time. |
| Baseline | `DummyClassifier(strategy="most_frequent")` on the same training split. |
| Test instances / test set | De-duplicate before stratified 80/20 train/test split; untouched test data. |
| Number of runs and seed | Stratified 5-fold CV within train data, shuffle seed `42`; final held-out test once. |
| Planned charts | (1) headline length distributions; (2) CV macro-F1 model comparison; (3) test confusion matrices; (4) cluster size/top terms. |

**Safeguards:** Fit TF-IDF inside every training fold; inspect publisher/class imbalance; report domain-generalization limits. Do not invent evaluation results.

## 8. Division of Work
| Member | Modules / components owned | Approximate share |
|---|---|---|
| Muhammad Hasnain Memon (leader) | NewsDataset, cleaning, TF-IDF pipeline, repository integration, reproducible tests, README | 34% |
| Akbar Hussain | NaiveBayesAgent, LogisticAgent, majority baseline, evaluation, keyword explanations | 33% |
| Ateeb Ahmed Qureshi | EDA, TopicClusterer, Streamlit views, figures and UI tests | 33% |

Each member will make individual commits and explain their own contribution during viva.

## 9. Milestones and Timeline
| # | Demonstrable milestone | Target | Owner |
|---|---|---|---|
| 1 | Repository, validated dataset and duplicate-free EDA | Week 9 | Hasnain + Ateeb |
| 2 | TF-IDF, two classifiers and majority baseline running | Week 10 | Akbar + Hasnain |
| 3 | CV, held-out metrics and explainable contributions verified | Weeks 11-12 | Akbar |
| 4 | K-Means topics and Streamlit app running offline | Weeks 13-14 | Ateeb |
| 5 | Three+ labelled charts; fresh-clone check; PDF/MD report; clean final submission | Week 15 | All members |

Calendar dates will be aligned with the instructor's announced lab schedule. The final repository submission is due in Week 15.

## 10. Expected Outcome and Risks
The final submission will launch a local Streamlit dashboard where a user types a headline, views a likely real/fake prediction, and sees the words contributing to that prediction. A second view will present exploratory statistics, K-Means topics and comparison charts for the two supervised classifiers and baseline. We will report measured performance on an untouched test set rather than promise a particular accuracy. Risk 1: labels and publisher-specific wording may bias a headline-only model; we will remove duplicates, examine class/source patterns and explain limits of generalization. Risk 2: downloading or processing the full dataset may be inconvenient; a documented, licence-compliant local sample and saved models will keep the application usable offline.

## For Office Use - Lab Instructor
| Field | Value |
|---|---|
| Instructor | Miss Amma Aslam |
| Proposal status | [ ] Approved  [ ] Approved with changes  [ ] Revision required |
| Comments | ____________________________ |
| Signature and date | ____________________________ |

**Academic integrity:** The team will acknowledge data, references and AI assistance, and be prepared to explain its own submitted code.
