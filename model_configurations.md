# Complete model and stage configurations

Every configuration reported in the paper, in one place. Parameters not listed are at the library defaults of
**Python 3.12.13, TensorFlow 2.20.0, scikit-learn 1.6.1**. Every experiment uses **random seed 42** for the
split, the initializers and the estimators. Every partition is a single stratified 80/20 split unless a
five-fold protocol is stated. **No decision threshold was tuned for any model** — the default of 0.5 on the
predicted probability, or the sign of the decision function for the SVM, is used throughout.

| Model / stage | Configuration | Class weighting | Reported in |
|---|---|---|---|
| Autoencoder, EXF-2021 | 20-64-32-16-32-64-20; ReLU on hidden and latent layers, linear output; Adam, learning rate 0.001; batch 256; up to 100 epochs; early stopping, patience 10 on a 10% validation split | — | Tables 4-7 |
| Autoencoder, BCCC-2024 | 100-64-32-32-32-64-100; ReLU on hidden layers, tanh latent, linear output; Adam, learning rate 0.001; batch 512; up to 40 epochs; early stopping, patience 6; fitted on a 100,000-row subsample of the training partition | — | Tables 8, 11 |
| Angular encoding (QIE) | Per-dimension min-max bounds from the training partition; scaling to [-1, 1] with explicit clipping; angle θ = π z; output [cos θ, sin θ]. No free parameters | — | All |
| K-Means (diagnostic) | k = 2; n_init = 10; run on the latent vectors; not part of the decision function | — | Sec. 3.4 |
| SVM (RBF), proposed pipeline | C = 10; gamma = "scale"; cache 1000 MB; decision by the sign of the decision function, no threshold tuning | None | Tables 4-7 |
| Random Forest, ablation baseline | 200 trees; all other parameters at scikit-learn defaults | None | Table 7 |
| Gradient Boosting, ablation baseline | scikit-learn defaults: 100 stages, learning rate 0.1, maximum depth 3 | None | Table 7 |
| MLP, ablation baseline | Hidden layers (64, 32); maximum 300 iterations; Adam, learning rate 0.001, ReLU; other parameters at defaults | None | Table 7 |
| Random Forest, BCCC-2024 | 300 trees; n_jobs = -1 | balanced | Tables 8, 11, 13 |
| XGBoost, BCCC-2024 | 400 estimators; maximum depth 8; learning rate 0.1; tree_method "hist"; eval_metric "logloss" | scale_pos_weight = n_negative / n_positive | Table 8 |
| LightGBM, BCCC-2024 | 400 estimators; all other parameters at library defaults | balanced | Table 8 |
| HistGradientBoosting, BCCC-2024 | max_iter = 400; all other parameters at library defaults | balanced | Table 8 |

## Optimization budgets are not equal

The proposed pipeline received the 20-trial TPE Bayesian search of Section 4.6 (`optuna_search.py`, released
as `optuna_trials.csv` and `optuna_study.db`). Every baseline above was run at the configuration shown, which
is either a small manual setting or the library default. The baseline figures are therefore **untuned
reference points**, and Tables 7 and 8 are not comparisons at matched tuning effort. The search did not
improve on the manually selected configuration, so this asymmetry does not favour the proposed method in the
reported results.

## Search space of the Bayesian optimization

| Hyperparameter | Range |
|---|---|
| First hidden layer | {64, 128, 256} |
| Second hidden layer | {32, 64} |
| Latent dimensionality | {24, 32, 48} |
| Dropout | [0.0, 0.3] |
| Trees | 150-400, step 50 |
| Maximum depth | 10-30 |

Sampler: TPE, seed 42. Trials: 20. Objective: attack-class F1 on a held-out validation subset drawn from the
training partition. Full per-trial results: `optuna_trials.csv`.

## Variability

Variability beyond a single fixed split is reported through the paired five-fold protocol of Section 4.8, in
which all compared configurations share identical folds and per-fold means are given with 95% confidence
intervals (Table 10 of the paper). Fold indices: `cv_folds_<dataset>.npz`.
