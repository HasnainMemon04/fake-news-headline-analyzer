# Verified dataset and offline preparation

Verified on 17 October 2026 using the complete downloaded CSVs. Access evidence is in [dataset-access.json](dataset-access.json); file hashes, counts and feasibility checks are in [feasibility.json](feasibility.json).

## Sources and access

- Original: [University of Victoria ISOT](https://onlineacademiccommunity.uvic.ca/isot/datasets/) and [official description](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf).
- Verified mirror: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), version 1. Its [public download endpoint](https://www.kaggle.com/api/v1/datasets/download/clmentbisaillon/fake-and-real-news-dataset) completed without an account or API key. The ZIP is 42,975,911 bytes and passes CRC integrity checks.
- Alternative [university ZIP](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip): HTTP 200, but slow transfers; its complete archive was not validated. Use the verified mirror/cache. The two archives are not claimed to be byte-identical.

Archive SHA-256: `38bf7e046c3a59b2c746c38f2fb74f4603493c6ed9c997f2f2ddb3969705fb42`. Access works today; local caching removes the external host requirement for subsequent work.

## Observed schema and counts

| File | Raw rows | Fields | Derived label |
|---|---:|---|---|
| True.csv | 21,417 | title, text, subject, date | 0 = real |
| Fake.csv | 23,481 | title, text, subject, date | 1 = fake |

There are 44,898 raw records, no empty normalized titles, and no exact normalized title assigned both labels. Literal repeated titles within the raw files are 591 and 5,578 respectively. Unicode, case and punctuation normalization identifies 6,178 repeated records in total, leaving 38,720 unique titles: 20,819 real and 17,901 fake. These are dataset audit observations, not classifier results.

Only title is a model input. File origin supplies labels; text, subject and date are audit fields only. Inspect length outliers and retain valid unusual headlines; document invalid-record exclusions without consulting test scores.

## Source analysis boundary

The CSVs have **no publisher/source/URL column**. The official description identifies Reuters for the real-news collection and multiple unreliable sites for fake news; individual fake publishers are not provided. EDA will report documented collection provenance and missing publisher coverage, alongside lengths, common words and subject distributions. Subject and filename must never be presented as publisher identity. Publisher-disjoint testing and individual-publisher accuracy cannot be claimed. Source confounding remains a documented limitation.

## Published usage terms

The mirror's [official Kaggle metadata API](https://www.kaggle.com/api/v1/datasets/list?search=fake-and-real-news-dataset) lists **CC BY-NC-SA 4.0** for this entry. The returned licence field, reference and version are recorded in dataset-access.json. This records the uploader's published terms; it is not a claim that the university separately issued that licence for every underlying article.

The university description requests citations to Ahmed, Traore and Saad (2018), *Detecting opinion spams and fake news using text classification*, and Ahmed, Traore and Saad (2017), *Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques*. Preserve attribution and the mirror's stated terms. Article text stays local and is not republished in this repository.

## Reproduce acquisition and verification

From the repository root, after installing requirements.txt:

```powershell
python tools/download_data.py
python tools/verify_feasibility.py
```

The downloader checks cached CSV hashes and returns without network access when they match. Otherwise it downloads and verifies the exact archive and file hashes before writing. It fails explicitly on an unexpected dataset change rather than silently replacing the data. The current workspace already contains the matching files in data/raw/, which Git ignores. Another machine must acquire them once or receive the same permitted files locally.

The verifier groups near duplicates by token-set Jaccard similarity >= 0.90 and keeps connected components in one partition. Its five-fold stratified group splitter reserves the first fold: 30,976 development and 7,744 held-out titles. Membership is saved to ignored data/processed/split_manifest.csv. Three-fold grouped CV and all feasibility checks use development data only; no predictions or metrics use the reserved final test.

Cache future fitted pipelines, label mapping, settings and hashes locally under outputs/results/. Once prepared, the final application and experiments require no network. Only load model artifacts created by this project from a trusted local location.
