# Explainable Fake News Headline Analyzer

AL-2002 Artificial Intelligence Laboratory semester project proposal, FAST-NUCES Karachi, Fall 2026, BCY-5A. Instructor: Miss Amma Aslam. Idea 19, Part B. The proposed application will classify English headlines using TF-IDF with Naive Bayes and Logistic Regression, explain each prediction using signed feature contributions, and explore K-Means topics in a local Streamlit interface. Predictions describe dataset patterns and cannot establish factual truth.

**Status: proposal and repository preparation complete; application implementation has not started.** No trained models, benchmark scores or final report are supplied. The three-member team requires instructor approval because Section 1 of the guidelines says at most two despite the three-member heading and template. Idea allocation must be confirmed on the section sheet.

## Team and planned ownership

| Member | Roll number | Planned work |
|---|---|---|
| Muhammad Hasnain Memon, leader | 24K-2001 | Architecture, preprocessing integration, classification, evaluation, repository coordination |
| Akbar Hussain | 24K-2026 | Data acquisition, EDA, K-Means, visualizations |
| Ateeb Ahmed Qureshi | 24K-2036 | Streamlit, explanation integration, testing, user documentation |

Responsibilities are plans, not claims of completed software contributions. Each student must commit their own actual work from their own account. Document preparation used OpenAI Codex assistance; final report acknowledgements must record external code, datasets and AI assistance.

## Proposal and requirements

- [Proposal PDF](docs/proposal.pdf) and [matching Markdown](docs/proposal.md)
- [Official requirement mapping and remaining actions](docs/compliance.md)
- [Planned module interfaces](src/README.md)
- [Dataset access and preparation](docs/data-source.md)
- [Environment validation](docs/environment.md)

The four-page proposal follows all ten official sections and retains their tables. Planned features are EDA of headline length, frequent words and sources; two classifier comparison plus a majority baseline; influential word highlighting; interpreted K-Means clusters; and interactive headline input. Scope excludes deep learning, paid APIs, deployment and live fact verification.

## Setup

Python 3.12 is the validated dependency interpreter. From a terminal:

```powershell
git clone https://github.com/HasnainMemon04/fake-news-headline-analyzer.git
cd fake-news-headline-analyzer
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

On Linux/macOS use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. The dependencies support the planned stack: Pandas, NumPy, scikit-learn, Matplotlib and Streamlit. `requirements.txt` records the actual resolved environment, including transitive dependencies; these are dependency checks, not application tests.

## Planned execution workflow

**The following commands define future interfaces and will work only after the modules are implemented.**

```powershell
python -m src.experiments --seed 42 --data-dir data/raw --output-dir outputs
python -m streamlit run src/app.py
```

First acquire local CSVs as documented in `docs/data-source.md`. The experiment command will validate/clean data, save split IDs, perform training-only cross-validation, fit selected models, evaluate once on the same held-out test IDs and write metrics/charts. The app will load saved artifacts without retraining or requiring a network connection. Invalid input, missing data and missing model files will produce specific instructions. No application command is represented as runnable today.

## Repository layout

```text
docs/                 proposal PDF/MD and supporting documentation
src/                  planned source modules; currently documentation only
data/raw/             local source CSVs, ignored except .gitkeep
data/processed/       local cleaned data and split manifests
outputs/figures/      future labelled charts
outputs/results/      future results and local model artifacts
requirements.txt      installed and verified dependency versions
```

The previous root proposal files were superseded by `docs/proposal.*`; Git history preserves the earlier draft. Raw data, local models, secrets, virtual environments and caches must not be committed.

## Evaluation and limitations

Seed Python `random`, NumPy and applicable estimators with 42. Group duplicate headlines before an approximately 80/20 stratified split, fit TF-IDF within training CV folds, and use identical locked test records for both models and the baseline. Report accuracy, class-wise precision/recall/F1, macro-F1, confusion matrices, timing, at least three labelled charts and interpreted clusters. Source overlap, unknown fake publishers, dated political language and headline ambiguity limit generalization. Probabilities are uncalibrated model scores. Highlighting explains model associations, not causality or truth.

Primary data: [University of Victoria ISOT](https://onlineacademiccommunity.uvic.ca/isot/datasets/); [official description](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf). Keep dataset contents local while reuse/redistribution terms remain unresolved. See the access record for details.

## Final project submission, later

Complete implementation and an 8-10-page report in PDF and Markdown under `docs/` by Week 15. The leader must verify a fresh clone on another machine, confirm genuine individual commits, and submit the public repository link to Google Classroom with member names and roll numbers. The proposal deadline and calendar dates are instructor-announced; this repository has not been submitted to Classroom.
