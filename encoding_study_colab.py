"""Encoding study: mapping baselines, boundary behaviour, and operational metrics.

NOTEBOOK VERSION - paste this whole cell into Colab and run. No command-line arguments.

Covers three reviewer requests in a single pass:
  * Comparison of the proposed angular (cosine-sine) mapping against normalization-only,
    tanh, sine-only, cosine-only, polynomial features and random Fourier features,
    under an identical classifier and tuning budget.
  * Boundary analysis of the angular mapping: out-of-range rate on test data, endpoint
    collisions at -1 and +1, aliasing without clipping, and numerical inversion error,
    which together test the "near-lossless" and "numerically stable" claims.
  * Operational metrics: false-positive rate, PR-AUC, alerts per 1,000 flows, a threshold
    sweep and a calibration score.

Everything is written to encoding_study_results.json.
"""

# ============================== CONFIG ==============================
CSV_PATH   = "merged_heavy.csv"     # "merged_mal_clean.csv" for BCCC-2024
LABEL_COL  = "label"
TEST_SIZE  = 0.2
SEED       = 42
LATENT     = 16                     # 16 for EXF-2021, 32 for BCCC-2024
N_TREES    = 300                    # identical budget for every representation
SUBSAMPLE  = None                   # e.g. 60000 to speed things up; None = use all rows
# ====================================================================

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             recall_score, precision_score, f1_score, brier_score_loss)

rng = np.random.RandomState(SEED)
results = {"file": CSV_PATH, "seed": SEED, "latent": LATENT, "n_trees": N_TREES}


def show(title, d):
    print("\n" + title)
    print("-" * len(title))
    for k, v in d.items():
        print(f"  {k:<44s} {v}")


# ----------------------------------------------------------------- data
df = pd.read_csv(CSV_PATH, low_memory=False)
y = df[LABEL_COL].astype(int).to_numpy()
Xdf = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(Xdf.median(numeric_only=True))
X = Xdf.to_numpy(dtype=np.float32)
if SUBSAMPLE and SUBSAMPLE < len(y):
    keep = rng.choice(len(y), size=SUBSAMPLE, replace=False)
    X, y = X[keep], y[keep]
print(f"Loaded {CSV_PATH}: {X.shape[0]:,} rows x {X.shape[1]} features; attack rate {y.mean():.4f}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
sc = StandardScaler().fit(Xtr)
Atr, Ate = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)

# ----------------------------------------------------------------- autoencoder
HAVE_TF = True
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    n = Atr.shape[1]
    inp = tf.keras.Input(shape=(n,))
    e = tf.keras.layers.Dense(64, activation='relu')(inp)
    e = tf.keras.layers.Dense(32, activation='relu')(e)
    z = tf.keras.layers.Dense(LATENT, activation='tanh')(e)
    d = tf.keras.layers.Dense(32, activation='relu')(z)
    d = tf.keras.layers.Dense(64, activation='relu')(d)
    out = tf.keras.layers.Dense(n, activation='linear')(d)
    ae = tf.keras.Model(inp, out)
    ae.compile(optimizer='adam', loss='mse')
    es = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    sub = rng.choice(len(Atr), size=min(100_000, len(Atr)), replace=False)
    ae.fit(Atr[sub], Atr[sub], validation_split=0.1, epochs=40, batch_size=512,
           callbacks=[es], verbose=0)
    enc = tf.keras.Model(inp, z)
    Ztr = enc.predict(Atr, batch_size=4096, verbose=0).astype(np.float64)
    Zte = enc.predict(Ate, batch_size=4096, verbose=0).astype(np.float64)
except ImportError:
    HAVE_TF = False
    print("[tensorflow unavailable] falling back to a PCA latent space of the same width")
    from sklearn.decomposition import PCA
    p = PCA(n_components=LATENT, random_state=SEED).fit(Atr)
    Ztr, Zte = p.transform(Atr), p.transform(Ate)
results["latent_source"] = "autoencoder" if HAVE_TF else "PCA fallback"

# min-max bounds from TRAINING data only
lo, hi = Ztr.min(0), Ztr.max(0)
span = np.where(hi - lo == 0, 1, hi - lo)


def norm(Z):
    return 2 * (Z - lo) / span - 1


Ntr_raw, Nte_raw = norm(Ztr), norm(Zte)     # may leave [-1, 1] on test data
Ntr, Nte = np.clip(Ntr_raw, -1, 1), np.clip(Nte_raw, -1, 1)


# ----------------------------------------------------------------- Part A: mappings
def m_identity(Ntr_, Nte_):
    return Ztr, Zte


def m_norm(Ntr_, Nte_):
    return Ntr_, Nte_


def m_tanh(Ntr_, Nte_):
    return np.tanh(Ztr), np.tanh(Zte)


def m_cos(Ntr_, Nte_):
    return np.cos(np.pi * Ntr_), np.cos(np.pi * Nte_)


def m_sin(Ntr_, Nte_):
    return np.sin(np.pi * Ntr_), np.sin(np.pi * Nte_)


def m_qie(Ntr_, Nte_):
    f = lambda N: np.hstack([np.cos(np.pi * N), np.sin(np.pi * N)])
    return f(Ntr_), f(Nte_)


def m_poly(Ntr_, Nte_):
    pf = PolynomialFeatures(degree=2, include_bias=False).fit(Ntr_)
    return pf.transform(Ntr_), pf.transform(Nte_)


def m_rff(Ntr_, Nte_):
    rf = RBFSampler(n_components=2 * LATENT, gamma='scale' if False else 1.0 / LATENT,
                    random_state=SEED).fit(Ntr_)
    return rf.transform(Ntr_), rf.transform(Nte_)


MAPPINGS = [
    ("raw standardized features (no AE)", None),
    ("latent, no mapping", m_identity),
    ("latent, min-max normalization only", m_norm),
    ("latent, tanh", m_tanh),
    ("latent, cosine only", m_cos),
    ("latent, sine only", m_sin),
    ("latent, cosine-sine (proposed QIE)", m_qie),
    ("latent, polynomial degree 2", m_poly),
    ("latent, random Fourier features", m_rff),
]


def fit_eval(Ftr, Fte):
    clf = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1,
                                 class_weight='balanced', random_state=SEED).fit(Ftr, ytr)
    proba = clf.predict_proba(Fte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return clf, proba, dict(
        dims=int(Ftr.shape[1]),
        accuracy=round(accuracy_score(yte, pred), 4),
        auc_roc=round(roc_auc_score(yte, proba), 4),
        pr_auc=round(average_precision_score(yte, proba), 4),
        attack_recall=round(recall_score(yte, pred), 4),
        attack_precision=round(precision_score(yte, pred, zero_division=0), 4),
        attack_f1=round(f1_score(yte, pred, zero_division=0), 4),
        fpr=round(float(((pred == 1) & (yte == 0)).sum() / max((yte == 0).sum(), 1)), 4),
    )


print("\n" + "=" * 96)
print("PART A - Representation comparison (identical classifier and budget: "
      f"RandomForest, {N_TREES} trees, seed {SEED})")
print("=" * 96)
hdr = f"{'Representation':<38s}{'dims':>6s}{'acc':>9s}{'AUC':>9s}{'PR-AUC':>9s}{'recall':>9s}{'F1':>9s}{'FPR':>8s}"
print(hdr)
partA, qie_proba = {}, None
for name, fn in MAPPINGS:
    Ftr, Fte = (Atr, Ate) if fn is None else fn(Ntr, Nte)
    clf, proba, m = fit_eval(Ftr, Fte)
    partA[name] = m
    if 'proposed' in name:
        qie_proba = proba
    print(f"{name:<38s}{m['dims']:>6d}{m['accuracy']:>9.4f}{m['auc_roc']:>9.4f}"
          f"{m['pr_auc']:>9.4f}{m['attack_recall']:>9.4f}{m['attack_f1']:>9.4f}{m['fpr']:>8.4f}")
results["part_A_representations"] = partA

# ----------------------------------------------------------------- Part B: boundary
print("\n" + "=" * 96)
print("PART B - Boundary behaviour of the angular mapping")
print("=" * 96)
out_lo = (Nte_raw < -1).sum()
out_hi = (Nte_raw > 1).sum()
tot = Nte_raw.size
rows_out = int((np.abs(Nte_raw) > 1).any(axis=1).sum())

# Without clipping, any value outside [-1, 1] is scaled to an angle outside [-pi, pi] and
# therefore wraps onto a point already occupied by an in-range value: that is the aliasing
# the clipping step prevents. The count is exactly the number of out-of-range values.
alias_pairs = int(out_lo + out_hi)
max_abs_norm = float(np.abs(Nte_raw).max())

at_neg1 = int((Nte == -1).sum())
at_pos1 = int((Nte == 1).sum())
# theta = -pi and theta = +pi map to the same point on the unit circle
collisions = min(at_neg1, at_pos1)

# numerical inversion: recover the normalized value from the (cos, sin) pair
theta = np.pi * Nte
C, S = np.cos(theta), np.sin(theta)
rec = np.arctan2(S, C) / np.pi
err = np.abs(rec - Nte)
err_interior = err[np.abs(Nte) < 1]

boundary = dict(
    test_values_total=int(tot),
    values_below_minus1_before_clipping=int(out_lo),
    values_above_plus1_before_clipping=int(out_hi),
    out_of_range_rate_pct=round(100 * (out_lo + out_hi) / tot, 4),
    test_rows_with_at_least_one_out_of_range_value=rows_out,
    test_rows_with_out_of_range_pct=round(100 * rows_out / len(Nte), 3),
    values_that_would_alias_without_clipping=alias_pairs,
    max_absolute_normalized_value_before_clipping=round(max_abs_norm, 4),
    values_clipped_to_minus1=at_neg1,
    values_clipped_to_plus1=at_pos1,
    endpoint_collision_pairs=collisions,
    inversion_error_mean_all=float(f"{err.mean():.3e}"),
    inversion_error_max_all=float(f"{err.max():.3e}"),
    inversion_error_max_interior_only=float(f"{err_interior.max():.3e}") if err_interior.size else None,
)
boundary["note_max_error"] = ("a maximum error of 2.0 can occur only at an endpoint, where -1 and +1 "
                              "map to the same point and the inversion returns the other endpoint")
show("Boundary and invertibility statistics (test partition)", boundary)
print("  Note: the mapping is injective on the open interval (-1, 1); the two endpoints -1 and +1")
print("  are mapped to the same point on the unit circle, so any collisions are confined to values")
print("  clipped to an endpoint. The counts above quantify how often that occurs in practice.")
results["part_B_boundary"] = boundary

# ----------------------------------------------------------------- Part C: operational
print("\n" + "=" * 96)
print("PART C - Operational metrics for the proposed representation")
print("=" * 96)
n_neg = int((yte == 0).sum())
sweep = []
print(f"{'threshold':>10s}{'precision':>11s}{'recall':>9s}{'FPR':>9s}{'alerts/1k flows':>17s}")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = (qie_proba >= t).astype(int)
    prec = precision_score(yte, pred, zero_division=0)
    rec_ = recall_score(yte, pred, zero_division=0)
    fpr = ((pred == 1) & (yte == 0)).sum() / max(n_neg, 1)
    alerts = 1000 * pred.mean()
    sweep.append(dict(threshold=t, precision=round(float(prec), 4), recall=round(float(rec_), 4),
                      fpr=round(float(fpr), 4), alerts_per_1000_flows=round(float(alerts), 1)))
    print(f"{t:>10.1f}{prec:>11.4f}{rec_:>9.4f}{fpr:>9.4f}{alerts:>17.1f}")
calib = dict(brier_score=round(float(brier_score_loss(yte, qie_proba)), 4),
             pr_auc=partA["latent, cosine-sine (proposed QIE)"]["pr_auc"],
             positive_rate_in_test=round(float(yte.mean()), 4))
show("Calibration and prevalence", calib)
results["part_C_threshold_sweep"] = sweep
results["part_C_calibration"] = calib

with open("encoding_study_results.json", "w") as fh:
    json.dump(results, fh, indent=2)
print("\nWritten to encoding_study_results.json")
