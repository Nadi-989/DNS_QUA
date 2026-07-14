# A Deep Autoencoder and Quantum-Inspired Feature Encoding Framework for DNS Anomaly Detection

Official code and data-preparation scripts for the paper:

> *A Deep Autoencoder and Quantum-Inspired Feature Encoding Framework for DNS Anomaly Detection* — [authors], [journal], [year].

The framework combines a deep autoencoder, a quantum-inspired angular feature
encoding (cosine–sine projection onto the unit circle), and a supervised
classifier (SVM-RBF / Random Forest), evaluated under a strict
**leakage-controlled protocol** on two public benchmarks. The repository also
contains the **data-quality audit** of CIC-Bell-DNS-EXF-2021 that quantifies
its duplicate (56.8%) and label-conflict (65.3%) rates.

## Repository contents

| File | Purpose |
|---|---|
| `prepare_data.py` | Downloads both datasets (Kaggle), merges the CSVs, assigns labels, removes leakage-prone identifier columns, and runs the duplicate/conflict audit |
| `experiments_main.py` | Main pipeline: preprocessing → autoencoder → angular encoding → classifier. Produces every table in the paper (metrics, per-class, confusion matrix, ablation, baselines, 5-fold CV) |
| `tune_pipeline.py` | Architecture/hyperparameter search (latent dim × activation × SVM grid) |
| `model_tournament.py` | 4 classifier families × {raw, proposed} features on the full BCCC-2024 corpus |
| `optuna_search.py` | 20-trial TPE Bayesian optimization over the joint AE + Random Forest space |
| `statistical_tests.py` | Paired t-test / Wilcoxon over identical 5-fold splits (Table 10) |
| `complexity_explainability.py` | Training/inference timing, memory, parameters (Table 11) and feature-importance analysis (Table 12) |

## Datasets

Both datasets are public; they are **not** redistributed in this repository.

1. **CIC-Bell-DNS-EXF-2021** — Canadian Institute for Cybersecurity / Bell Canada.
   Official page: https://www.unb.ca/cic/datasets/dns-exf-2021.html
   Kaggle mirror used in the paper: `humera11/cicbelldnsexf2021`
2. **BCCC-CIC-Bell-DNS-2024** — Behaviour-Centric Cybersecurity Center (York University).
   Kaggle: `bcccdatasets/malicious-dns-and-attacks-bccc-cic-bell-dns-2024`

`prepare_data.py` downloads both via `kagglehub` and reproduces the exact
corpora used in the paper (heavy-exfiltration stateful subset for EXF-2021;
the full Mal portion with leakage-prone columns removed for BCCC-2024).

## Quick start (Google Colab or local)

```bash
pip install numpy pandas scikit-learn tensorflow kagglehub optuna
python prepare_data.py                 # downloads + merges + audits
python experiments_main.py merged_heavy.csv label        # EXF-2021 results
python model_tournament.py                               # BCCC-2024 results
python statistical_tests.py                              # Table 10
python complexity_explainability.py                      # Tables 11-12
```

All experiments fix `random_state = 42`; the stratified 80/20 split, the
scaler, the autoencoder, and the angular-encoding bounds are fitted on the
training partition only (no information leakage).

## Key results

| Dataset | Setting | Accuracy | Notes |
|---|---|---|---|
| CIC-Bell-DNS-EXF-2021 | AE(16) + QIE + SVM-RBF | 79.25% | attack recall **98.82%**, CV 79.69 ± 0.28% |
| BCCC-CIC-Bell-DNS-2024 | AE(32, tanh) + QIE + RF | 88.00% | within 0.7 pts of the best 100-dim raw model |

The data-quality audit shows that accuracies of 99%+ previously reported on
CIC-Bell-DNS-EXF-2021 arise from duplicate leakage under random splits; see
Section 4.2 of the paper.

## Citation

```bibtex
@article{dnsqie2026,
  title   = {A Deep Autoencoder and Quantum-Inspired Feature Encoding
             Framework for DNS Anomaly Detection},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  note    = {Code: https://github.com/[USERNAME]/dns-qie-anomaly-detection}
}
```

## License

MIT — see `LICENSE`.
