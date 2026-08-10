"""Computational complexity (Table 11) and explainability (Table 12).

Usage: python complexity_explainability.py   (expects merged_mal_clean.csv)

Revised for the reviewer's comment on Table 11: the single "Parameters / Memory"
column is replaced by separate, explicitly-unitted quantities —
trainable parameters, number of trees, number of nodes, serialized model size (MB)
and peak resident memory (MB). Every measurement is written to
artifacts/model_metadata.json so the table can be regenerated without re-running.
"""
import warnings, time, gc, os, json, threading
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import joblib

try:
    import psutil
    proc = psutil.Process(os.getpid())
    def mem_mb(): return proc.memory_info().rss / 1e6
    HAVE_PSUTIL = True
except ImportError:
    def mem_mb(): return float('nan')
    HAVE_PSUTIL = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

SEED = 42
AE_SUBSAMPLE = 100_000          # reported in Section 4.9 — keep manuscript and code identical
ARTIFACTS = "artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)

df = pd.read_csv("merged_mal_clean.csv", low_memory=False)
y = df['label'].astype(int).values
Xdf = df.drop(columns=['label']).select_dtypes(include=[np.number])
Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(Xdf.median(numeric_only=True))
feat_names = list(Xdf.columns)
X = Xdf.values.astype(np.float32)
del df, Xdf; gc.collect()

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                      stratify=y, random_state=SEED)
del X; gc.collect()
sc = StandardScaler().fit(Xtr)
Xs_tr = sc.transform(Xtr).astype(np.float32)
Xs_te = sc.transform(Xte).astype(np.float32)
del Xtr, Xte; gc.collect()

# ---------------- AE timing / params ----------------
import tensorflow as tf
tf.random.set_seed(SEED)
n = Xs_tr.shape[1]
inp = tf.keras.Input(shape=(n,))
e = tf.keras.layers.Dense(64, activation='relu')(inp)
e = tf.keras.layers.Dense(32, activation='relu')(e)
z = tf.keras.layers.Dense(32, activation='tanh')(e)
d = tf.keras.layers.Dense(32, activation='relu')(z)
d = tf.keras.layers.Dense(64, activation='relu')(d)
out = tf.keras.layers.Dense(n, activation='linear')(d)
ae = tf.keras.Model(inp, out); ae.compile(optimizer='adam', loss='mse')

rs = np.random.RandomState(SEED)
sub = rs.choice(len(Xs_tr), size=AE_SUBSAMPLE, replace=False)
t0 = time.time()
es = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
ae.fit(Xs_tr[sub], Xs_tr[sub], validation_split=0.1, epochs=40,
       batch_size=512, callbacks=[es], verbose=0)
ae_train_s = time.time() - t0
enc = tf.keras.Model(inp, z)

ae_params = ae.count_params()
enc_params = enc.count_params()
ae_path = os.path.join(ARTIFACTS, "autoencoder.keras")
ae.save(ae_path)
ae_size_mb = os.path.getsize(ae_path) / 1e6

t0 = time.time()
Z_tr = enc.predict(Xs_tr, batch_size=4096, verbose=0)
Z_te = enc.predict(Xs_te, batch_size=4096, verbose=0)
enc_ms = (time.time() - t0) / (len(Xs_tr) + len(Xs_te)) * 1000

lo, hi = Z_tr.min(0), Z_tr.max(0)
def qie(Z):
    Zn = 2 * (Z - lo) / np.where(hi - lo == 0, 1, hi - lo) - 1
    th = np.pi * np.clip(Zn, -1, 1)
    return np.hstack([np.cos(th), np.sin(th)]).astype(np.float32)

t0 = time.time()
P_tr, P_te = qie(Z_tr), qie(Z_te)
qie_ms = (time.time() - t0) / (len(Z_tr) + len(Z_te)) * 1000
del Z_tr, Z_te; gc.collect()


# ---------------- peak-RSS sampler ----------------
class PeakRSS:
    """Samples RSS in a background thread so the reported memory is a true peak
    during fitting, not a start/end difference."""
    def __init__(self, interval=0.2):
        self.interval, self.peak, self._stop = interval, 0.0, threading.Event()
    def __enter__(self):
        self.base = mem_mb()
        self.peak = self.base
        if HAVE_PSUTIL:
            self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
        return self
    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, mem_mb())
    def __exit__(self, *a):
        self._stop.set()
        if HAVE_PSUTIL:
            self._t.join(timeout=2)
        self.peak = max(self.peak, mem_mb())
    @property
    def delta(self):
        return self.peak - self.base


# ---------------- RF timing / size / memory ----------------
def rf_profile(tag, Atr, Ate):
    m_before = mem_mb()
    with PeakRSS() as pk:
        t0 = time.time()
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                   class_weight='balanced', random_state=SEED).fit(Atr, ytr)
        tr_s = time.time() - t0
    rss_delta = mem_mb() - m_before        # same method as the originally published 162 / 254
    t0 = time.time(); pred = m.predict(Ate)
    inf_ms = (time.time() - t0) / len(Ate) * 1000
    path = os.path.join(ARTIFACTS, f"{tag}.joblib")
    joblib.dump(m, path, compress=0)
    rec = dict(
        model=tag,
        n_features=int(Atr.shape[1]),
        train_s=round(tr_s, 1),
        infer_ms_per_sample=round(inf_ms, 4),
        trainable_parameters=None,          # not applicable to a forest
        n_trees=len(m.estimators_),
        n_nodes=int(sum(t.tree_.node_count for t in m.estimators_)),
        serialized_mb=round(os.path.getsize(path) / 1e6, 1),
        rss_delta_mb=round(rss_delta, 0),
        peak_rss_mb=round(pk.delta, 0),
        accuracy=round(accuracy_score(yte, pred), 4),
    )
    return m, rec


rf_q, rec_q = rf_profile("RF_proposed_QIE_64dim", P_tr, P_te)
rf_r, rec_r = rf_profile("RF_raw_100dim", Xs_tr, Xs_te)

rec_ae = dict(
    model="Autoencoder (one-time training)",
    n_features=int(n),
    train_s=round(ae_train_s, 1),
    infer_ms_per_sample=round(enc_ms, 4),
    trainable_parameters=int(ae_params),
    encoder_only_parameters=int(enc_params),
    n_trees=None, n_nodes=None,
    serialized_mb=round(ae_size_mb, 1),
    peak_rss_mb=None,
    ae_training_subsample=AE_SUBSAMPLE,
)
rec_qie = dict(
    model="Angular encoding (QIE)",
    train_s=None,
    infer_ms_per_sample=round(qie_ms, 4),
    trainable_parameters=0,
    n_trees=None, n_nodes=None, serialized_mb=None, peak_rss_mb=None,
)

meta = dict(seed=SEED, records=[rec_ae, rec_qie, rec_q, rec_r])
with open(os.path.join(ARTIFACTS, "model_metadata.json"), "w") as fh:
    json.dump(meta, fh, indent=2)

# ---------------- Table 11 ----------------
H = "{:<32s}{:>10s}{:>16s}{:>13s}{:>8s}{:>12s}{:>11s}{:>12s}{:>13s}"
R = H
print("=" * 118)
print("TABLE 11 - Computational cost on BCCC-CIC-Bell-DNS-2024 (CPU)")
print("=" * 118)
print(H.format("Component / Model", "Train (s)", "Infer (ms/samp)", "Params",
               "Trees", "Nodes", "File (MB)", "RSS-d (MB)", "PeakRSS (MB)"))


def fmt(v, kind="n"):
    if v is None:
        return "-"
    if kind == "i":
        return f"{int(v):,}"
    return str(v)


for r in (rec_ae, rec_qie, rec_q, rec_r):
    print(R.format(
        r["model"][:32],
        fmt(r.get("train_s")),
        fmt(r.get("infer_ms_per_sample")),
        fmt(r.get("trainable_parameters"), "i"),
        fmt(r.get("n_trees"), "i"),
        fmt(r.get("n_nodes"), "i"),
        fmt(r.get("serialized_mb")),
        fmt(r.get("rss_delta_mb")),
        fmt(r.get("peak_rss_mb")),
    ))
print()
print(f"Encoder-only trainable parameters: {enc_params:,}   "
      f"(full autoencoder, encoder+decoder: {ae_params:,})")
print(f"Autoencoder trained on a {AE_SUBSAMPLE:,}-sample subset of the training partition.")
print(f"acc: QIE={rec_q['accuracy']:.4f}  raw={rec_r['accuracy']:.4f}")
print()
print("Units: Params = trainable parameters (count). Trees / Nodes = counts over the whole forest.")
print("File (MB) = size of the joblib-serialized model on disk, uncompressed, decimal MB (bytes/1e6).")
print("RSS-d (MB)   = resident set size after fitting minus before fitting (the method used for the")
print("               originally reported 162 / 254 values), decimal MB.")
print("PeakRSS (MB) = peak resident set size during fitting minus the baseline, sampled every 200 ms")
print("               via psutil, decimal MB. This is the stricter of the two measures.")

# ---------------- Table 12 ----------------
print("\n" + "=" * 74)
print("TABLE 12 - Explainability (RF on raw features)")
print("=" * 74)
imp = sorted(zip(feat_names, rf_r.feature_importances_), key=lambda x: -x[1])[:10]
print("Impurity importance (top-10):")
for f, v in imp:
    print(f"  {f:<45s} {v:.4f}")

sub_te = rs.choice(len(Xs_te), size=10_000, replace=False)
pi = permutation_importance(rf_r, Xs_te[sub_te], yte[sub_te], n_repeats=5,
                            random_state=SEED, n_jobs=-1)
pim = sorted(zip(feat_names, pi.importances_mean), key=lambda x: -x[1])[:10]
print("Permutation importance (top-10):")
for f, v in pim:
    print(f"  {f:<45s} {v:.4f}")

print(f"\nArtifacts written to ./{ARTIFACTS}/ (model_metadata.json + serialized models).")
