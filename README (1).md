# A Deep Autoencoder and Quantum-Inspired Feature Encoding Framework for DNS Anomaly Detection

Official code, data-preparation scripts and reproducibility artifacts for the paper:

> **A Deep Autoencoder and Quantum-Inspired Feature Encoding Framework for DNS Anomaly Detection**
> N. M. Hussien, M. T. Gaata, H. B. Taher. *International Journal of Intelligent Engineering and Systems*, [year].

The framework combines a deep autoencoder, a quantum-inspired angular feature encoding (a cosine–sine projection onto the unit circle) and a supervised classifier (SVM-RBF / Random Forest), evaluated under a leakage-controlled protocol on two public benchmarks. The repository also contains the data-quality audit of CIC-Bell-DNS-EXF-2021 that quantifies its duplicate (56.8%) and label-conflict (65.3%) rates.

The angular encoding is a **classical periodic coordinate transformation** motivated by angle-encoding concepts from variational quantum circuits. It does not use superposition or entanglement and is not executed on quantum hardware.

---

## 1. Repository contents

| File | Purpose |
|---|---|
| `prepare_data.py` | Downloads both datasets (Kaggle), merges the CSVs, assigns labels, removes leakage-prone identifier columns, runs the duplicate/conflict audit |
| `experiments_main.py` | Main pipeline: preprocessing → autoencoder → angular encoding → classifier. Produces Tables 4, 4b, 5, 6, 7 and Fig. 4 |
| `tune_pipeline.py` | Architecture/hyperparameter search (latent dim × activation × SVM grid) |
| `model_tournament.py` | Classifier comparison on the full BCCC-2024 corpus (Table 8, Fig. 5) |
| `optuna_search.py` | 20-trial TPE Bayesian optimization over the joint AE + Random Forest space (Section 4.6, Appendix A) |
| `statistical_tests.py` | Paired t-test / Wilcoxon over identical five-fold splits (Table 10) |
| `complexity_explainability.py` | Training/inference timing, memory, parameters (Table 11) and feature-importance analysis (Table 12) |
| `verify_kmeans_on_real_data.py` | Latent-space K-Means diagnostic (Section 3.4) |
| `requirements.txt` | Pinned package versions of the environment used for all reported results |

Note on `model_tournament.py`: the BCCC-2024 comparison covers **five classifier families in seven configurations**. Random Forest and XGBoost are run under both representations (raw and proposed); LightGBM and HistGradientBoosting are run on raw features only, as supplementary baselines; SVM (RBF) is run on the proposed representation only. It is not a full paired grid, and representation-effect conclusions in the paper are restricted to Random Forest and XGBoost.

---

## 2. Datasets

Both datasets are public and are **not redistributed here**, in accordance with their licences.

| Dataset | Source | Mirror used | Portion used |
|---|---|---|---|
| CIC-Bell-DNS-EXF-2021 | Canadian Institute for Cybersecurity / Bell Canada — https://www.unb.ca/cic/datasets/dns-exf-2021.html | Kaggle `humera11/cicbelldnsexf2021` | Heavy-exfiltration stateful subset |
| BCCC-CIC-Bell-DNS-2024 | Behaviour-Centric Cybersecurity Center, York University | Kaggle `bcccdatasets/malicious-dns-and-attacks-bccc-cic-bell-dns-2024` | Full Mal portion, leakage-prone columns removed |

### 2.1 Exact source files and checksums

`prepare_data.py --verify` prints the file name, byte size and SHA-256 of every source CSV it consumes and compares them against `data/checksums.csv`. The table below records the files used for the reported results.

| Dataset | File name | Rows | SHA-256 |
|---|---|---|---|
| EXF-2021 | *[TO FILL]* | *[TO FILL]* | *[TO FILL]* |
| BCCC-2024 | *[TO FILL]* | *[TO FILL]* | *[TO FILL]* |

### 2.2 Label mapping

*[TO FILL — state the exact source column and the mapping to {0 = benign, 1 = attack} for each dataset.]*

### 2.3 Excluded columns

Identifier and leakage-prone columns removed before modelling are listed in `config/excluded_columns.json`. *[TO FILL — list them explicitly here as well, so a reader can check without running the code.]*

### 2.4 Retained feature list

The final feature sets (20 features for EXF-2021, 100 for BCCC-2024) are written to `config/features_exf2021.json` and `config/features_bccc2024.json` by `prepare_data.py`.

### 2.5 Sampling procedure

Where a subsample is used, the fraction, the seed and the stage at which it is applied are recorded in `config/sampling.json`. *[TO FILL — confirm the fractions actually used, e.g. the 10% chunk sampling in preprocessing and the autoencoder training subsample size reported in Section 4.9.]*

---

## 3. Protocol, seeds and splits

All experiments fix `random_state = 42`. The scaler, the autoencoder and the angular-encoding min–max bounds are fitted on the training partition only.

* **Split:** stratified 80/20 train/test. The exact row indices are released as `splits/test_indices.npy` and `splits/train_indices.npy` so that the partition can be reproduced without re-running the split.
* **Cross-validation:** stratified five-fold with shared folds across all compared configurations (the paired protocol of Section 4.8). Fold indices are released as `splits/cv_folds.npz`. This is the **single authoritative cross-validation experiment**; the manuscript reports no other.
* **Per-fold predictions:** `artifacts/cv_predictions/` contains one file per fold and configuration.
* **Optuna:** the full study is released as `artifacts/optuna_study.db` and `artifacts/optuna_trials.csv`, giving the configuration, resulting representation dimensionality and objective value of every trial (Appendix A of the paper).
* **Trained-model metadata:** `artifacts/model_metadata.json` records, for each model, the hyperparameters, the number of trees and nodes, the serialized size and the peak resident memory.

**Known limitation.** The split is sample-level and stratified; it is not grouped by capture session, domain or source entity. Because CIC-Bell-DNS-EXF-2021 contains extensive exact duplication, identical feature vectors can fall on both sides of the boundary. `prepare_data.py --overlap` reports the overlap statistics before and after splitting. Group-disjoint evaluation is identified as future work in the paper.

---

## 4. Environment

`requirements.txt` pins the exact versions used. The reported results were produced on Google Colab (CPU runtime).

```
python  [TO FILL]
numpy   [TO FILL]
pandas  [TO FILL]
scikit-learn [TO FILL]
tensorflow   [TO FILL]
xgboost      [TO FILL]
lightgbm     [TO FILL]
optuna       [TO FILL]
```

---

## 5. Quick start

```bash
pip install -r requirements.txt

python prepare_data.py                                # download, merge, audit, write configs and checksums
python experiments_main.py merged_heavy.csv label     # EXF-2021: Tables 4, 4b, 5, 6, 7; Fig. 4
python model_tournament.py                            # BCCC-2024: Table 8; Fig. 5
python optuna_search.py                               # Section 4.6 and Appendix A
python statistical_tests.py                           # Table 10
python complexity_explainability.py                   # Tables 11 and 12
python make_figures.py                                # regenerates every figure in the paper
```

Each script writes its numerical output to `results/` in CSV form, so that every number in the paper can be traced to a file.

---

## 6. Key results

| Dataset | Setting | Accuracy | Notes |
|---|---|---|---|
| CIC-Bell-DNS-EXF-2021 | AE(16) + QIE + SVM-RBF | 79.25% | attack recall 98.82%; five-fold CV 79.65% ± 0.35% |
| BCCC-CIC-Bell-DNS-2024 | AE(32, tanh) + QIE + RF | 88.00% | 0.68 points below the best raw-feature model in accuracy, 0.0099 below in AUC-ROC and 0.0178 below in attack F1 |

The 88.00% result uses a 64-dimensional encoded representation against 100 raw features. On EXF-2021 the encoded representation is **not** a dimensional reduction: 20 raw features are compressed to 16 latent dimensions and then expanded to 32 cosine–sine components.

The data-quality audit shows that accuracies of 99%+ previously reported on CIC-Bell-DNS-EXF-2021 arise from duplicate leakage under random splits; see Section 4.2 of the paper.

---

## 7. Citation

```bibtex
@article{dnsqie2026,
  title   = {A Deep Autoencoder and Quantum-Inspired Feature Encoding
             Framework for DNS Anomaly Detection},
  author  = {Hussien, Nadia Mahmood and Gaata, Methaq Talib and Taher, Hazeem B.},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {[year]},
  note    = {Code: https://github.com/Nadi-989/DNS_QUA}
}
```

## 8. License

MIT — see `LICENSE`. The licence covers the code in this repository only; the datasets remain under the terms of their respective providers.
