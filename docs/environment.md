# Dependency validation

Validated on Windows with Python 3.12.14 on 17 October 2026 in an isolated local `.venv`. The environment is ignored by Git. `requirements.txt` is the actual output of `python -m pip freeze`, including resolved transitive packages.

| Direct dependency | Installed version |
|---|---|
| Pandas | 3.0.6 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| Streamlit | 1.66.0 |

`python -m pip check` returned **No broken requirements found**. Importing all five direct packages succeeded. These checks establish an installed dependency environment only. No classifier, app, dataset experiment or clean-clone application run has been completed. Verify compatibility again as implementation develops, and test the final project from a fresh clone on a different machine as the course requires.

Document QA: all four A4 PDF pages were rendered and visually inspected. All ten numbered sections and required member metadata were checked. After stripping Markdown syntax and recurring PDF page furniture, the PDF and Markdown matched in ordered alphanumeric text. Word counts are 158 for the problem statement, 125 for the technique justification and 106 for expected outcomes/risks. No measured model results were generated.
