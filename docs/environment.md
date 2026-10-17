# Dependency validation

Validated on Windows with Python 3.12.14 on 17 October 2026 in an isolated local `.venv`. The environment is ignored by Git. `requirements.txt` is the actual output of `python -m pip freeze`, including resolved transitive packages.

| Direct dependency | Installed version |
|---|---|
| Pandas | 3.0.6 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| Streamlit | 1.66.0 |

`python -m pip check` returned **No broken requirements found**. Importing all five direct packages succeeded. The separate feasibility script additionally passed real-data NB/LR fitting and grouped CV, contribution identities, saved-model reload parity, K-Means for k=3/5/8, and Streamlit input/unknown-vocabulary checks. These checks exercise the proposed methods; the full course application and a clean-clone run on a different machine remain future work. No final test predictions or scores were produced.

Document QA: all four A4 PDF pages were rendered and visually inspected. All ten numbered sections and required member metadata were checked. After stripping Markdown syntax and recurring PDF page furniture, the PDF and Markdown matched in ordered alphanumeric text. Word counts are recorded in compliance.md. Dataset auditing and development-only feasibility checks are separate from final held-out evaluation. Proposal authoring uses ReportLab 4.4.9 from the established document runtime, declared separately in requirements-docs.txt; it is not needed for classification.
