# Dataset and implementation feasibility

This is a pre-submission engineering check, not the final project report or final test evaluation. Checks were executed against the real local ISOT CSVs on 17 October 2026. [Machine-readable evidence](feasibility.json) records observed counts, hashes, split sizes and outcomes.

| Check | Observed outcome |
|---|---|
| Actual download and integrity | Complete anonymous mirror download; ZIP CRC and file SHA-256 recorded; download helper also passed in an empty directory and reused the cache without network |
| Schema and binary labels | title/text/subject/date confirmed; 44,898 raw records; labels derived from the two files |
| Duplicate handling | 6,178 normalized repeats removed; 38,720 unique titles; zero conflicting exact normalized labels |
| Near-duplicate grouping | Label-blind token-set Jaccard >= 0.90; 38,695 groups; grouping checked against brute force on a small corpus |
| Held-out isolation | 30,976 development / 7,744 reserved test titles; zero duplicate-group overlap; no reserved-test predictions or scores |
| Naive Bayes and Logistic Regression | Fitting, finite development-validation metrics and declared three-fold grouped CV parameter grids pass |
| Explainability | Signed term sums plus LR intercept or NB prior match fitted decision/log-odds values within numerical tolerance |
| Majority baseline | Fits and predicts matching validation instances |
| K-Means | k=3,5,8 all fit with nonempty clusters; silhouette computation and centroid features available |
| Offline model reuse | Saved/reloaded local pipelines produce identical validation predictions without network calls |
| Invalid text | Empty and unknown-token vectors detected for an explanatory UI response |
| Streamlit integration spike | Actual AppTest runtime loads a saved model, accepts text and handles unknown vocabulary without exceptions |
| Dependency compatibility | Required packages import and pip check passes in the pinned Python 3.12 environment |

The checks support implementation feasibility on the current machine. They do not establish a target accuracy, real-world truth detection, semantic topic quality or final project completion. Final classes, full highlighting UI, complete EDA views, documentation, report, experiments and another-machine clean-clone verification still need to be built and assessed.

The material dataset boundary is the absence of publisher identifiers. The proposal therefore explicitly defines source EDA as collection-provenance and missing-source analysis; no per-publisher statistics or publisher-disjoint evaluation is promised. No other mandatory Idea 19 feature requires unavailable data or infrastructure.

Reproduce from the repository root:

```powershell
python tools/download_data.py
python tools/verify_feasibility.py
```

The feasibility script writes only an audit/check record and a local split manifest; temporary trained models are deleted. It does not publish article text, final test metrics, best hyperparameters or a final report. [Implementation decisions](implementation-plan.md) fix the data, preprocessing, parameter grids and responsibilities before proposal submission.
