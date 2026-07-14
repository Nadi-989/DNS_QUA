"""Computational complexity (Table 11) and explainability (Table 12).
Usage: python complexity_explainability.py   (expects merged_mal_clean.csv)"""
import warnings, time, gc, os
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
try:
    import psutil
    proc = psutil.Process(os.getpid())
    def mem_mb(): return proc.memory_info().rss / 1e6
except ImportError:
    def mem_mb(): return float('nan')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

SEED = 42
df = pd.read_csv("merged_mal_clean.csv", low_memory=False)
y = df['label'].astype(int).values
Xdf = df.drop(columns=['label']).select_dtypes(include=[np.number])
Xdf = Xdf.replace([np.inf,-np.inf], np.nan).fillna(Xdf.median(numeric_only=True))
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
sub = rs.choice(len(Xs_tr), size=100_000, replace=False)
t0 = time.time()
es = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
ae.fit(Xs_tr[sub], Xs_tr[sub], validation_split=0.1, epochs=40,
       batch_size=512, callbacks=[es], verbose=0)
ae_train_s = time.time() - t0
enc = tf.keras.Model(inp, z)

t0 = time.time()
Z_tr = enc.predict(Xs_tr, batch_size=4096, verbose=0)
Z_te = enc.predict(Xs_te, batch_size=4096, verbose=0)
enc_ms = (time.time()-t0) / (len(Xs_tr)+len(Xs_te)) * 1000

lo, hi = Z_tr.min(0), Z_tr.max(0)
def qie(Z):
    Zn = 2*(Z-lo)/np.where(hi-lo==0, 1, hi-lo) - 1
    th = np.pi*np.clip(Zn, -1, 1)
    return np.hstack([np.cos(th), np.sin(th)]).astype(np.float32)
t0 = time.time()
P_tr, P_te = qie(Z_tr), qie(Z_te)
qie_ms = (time.time()-t0) / (len(Z_tr)+len(Z_te)) * 1000
del Z_tr, Z_te; gc.collect()

# ---------------- RF timing: proposed vs raw ----------------
def rf_timing(Atr, Ate):
    m0 = mem_mb(); t0 = time.time()
    m = RandomForestClassifier(n_estimators=300, n_jobs=-1,
            class_weight='balanced', random_state=SEED).fit(Atr, ytr)
    tr_s = time.time()-t0; dm = mem_mb() - m0
    t0 = time.time(); pred = m.predict(Ate)
    inf_ms = (time.time()-t0)/len(Ate)*1000
    return m, tr_s, inf_ms, dm, accuracy_score(yte, pred)

rf_q, q_tr, q_inf, q_mem, q_acc = rf_timing(P_tr, P_te)
rf_r, r_tr, r_inf, r_mem, r_acc = rf_timing(Xs_tr, Xs_te)

print("="*74)
print("TABLE 11 — Computational cost (CPU)")
print("="*74)
print(f"{'Component':<36s}{'Train(s)':>9s}{'Infer(ms/sample)':>18s}{'Params/Mem':>11s}")
print(f"{'Autoencoder (one-time)':<36s}{ae_train_s:>9.1f}{enc_ms:>18.4f}{ae.count_params():>11,d}")
print(f"{'Angular encoding (QIE)':<36s}{'—':>9s}{qie_ms:>18.4f}{'—':>11s}")
print(f"{'RF on proposed QIE (64-dim)':<36s}{q_tr:>9.1f}{q_inf:>18.4f}{q_mem:>9.0f}MB")
print(f"{'RF on raw features (100-dim)':<36s}{r_tr:>9.1f}{r_inf:>18.4f}{r_mem:>9.0f}MB")
print(f"acc: QIE={q_acc:.4f}  raw={r_acc:.4f}")

print("\n" + "="*74)
print("TABLE 12 — Explainability (RF on raw features)")
print("="*74)
imp = sorted(zip(feat_names, rf_r.feature_importances_), key=lambda x: -x[1])[:10]
print("Impurity importance (top-10):")
for f, v in imp: print(f"  {f:<45s} {v:.4f}")
sub_te = rs.choice(len(Xs_te), size=10_000, replace=False)
pi = permutation_importance(rf_r, Xs_te[sub_te], yte[sub_te], n_repeats=5,
                            random_state=SEED, n_jobs=-1)
pim = sorted(zip(feat_names, pi.importances_mean), key=lambda x: -x[1])[:10]
print("Permutation importance (top-10):")
for f, v in pim: print(f"  {f:<45s} {v:.4f}")
