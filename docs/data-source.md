# Dataset access and preparation

Checked 17 October 2026.

Access verification: the university archive returned HTTP 200, Content-Type `application/zip`, and Content-Length 43,106,824 bytes without authentication. A complete bulk download was not retained or validated during proposal preparation; CSV schema and row counts still require validation at acquisition. The source page and official PDF description were readable. This distinction avoids treating a reachable archive as a completed local dataset.

- Primary: [ISOT, University of Victoria](https://onlineacademiccommunity.uvic.ca/isot/datasets/).
- [Public ZIP](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip).
- [Official description and requested citations](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf).
- Practical mirror if the university host is unavailable: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). Its page is reachable; manual download may require a free account. Scripted access uses Kaggle credentials; do not store credentials in Git. Mirror download was not independently exercised.

The official description reports 21,417 real and 23,481 fake records, totalling 44,898 before cleaning. It describes title, article text, subject/type and date fields. Expected CSV headers are `title,text,subject,date`; validate them against the actual archive. Use `title` only for classification; construct labels from `True.csv` (0) and `Fake.csv` (1), and retain audit metadata separately.

The official description asks users to cite Ahmed, Traore and Saad (2018), *Detecting opinion spams and fake news using text classification*, and Ahmed, Traore and Saad (2017), *Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques*. It does not state an explicit dataset licence. No Creative Commons licence or redistribution permission is asserted here. Confirm applicable terms before redistributing articles or derived text samples. Neither original nor mirror data is committed.

Download manually, extract CSVs to `data/raw/`, record source URL, acquisition date, byte hashes and observed counts. Cache cleaned records and split/group IDs in `data/processed/`. Save trained pipelines locally in `outputs/results/`; model use must respect applicable source terms. The cache is a local prerequisite: no dataset or trained artifact is bundled with the proposal. Once acquired and trained, no network will be needed for experiments or inference. A fresh offline machine must receive the permitted local files first.

Clean empty titles and exact normalized duplicates; quarantine conflicting labels. Use label-blind near-duplicate groups, then approximately stratified 80/20 grouped splitting and grouped CV inside training. Fit vocabularies, IDF, model parameters and cluster centroids only on relevant training partitions. Audit outliers before choosing documented rules. Keep the final test sealed until settings are selected.

Reuters provenance is described for real news; individual publisher identifiers may be absent for fake articles. Record missing source coverage explicitly. `subject`, CSV origin and labels are not publisher IDs. Use verified metadata where present, never fabricated publishers; add a source-group holdout only if both classes and adequate groups permit meaningful evaluation. Report residual source confounding rather than claiming publisher-independent performance.
