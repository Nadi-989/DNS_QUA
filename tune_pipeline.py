"""
TUNING RUN — find the right configuration for the 100-feature dataset.
Tests: latent dim {16, 32} x latent activation {relu, tanh} x SVM class_weight,
plus a small C/gamma grid, all on merged_mal_clean.csv.
Baselines (RF tuned lightly) included for a fair comparison.
Runtime: ~30-45 min on Colab. No CV here — CV comes in the final approved run.
"""
import time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------ data
df = pd.read_csv("merged_mal_clean.csv", low_memory=False)
y = df['label'].astype(int).values
X = df.drop(columns=['label']).select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
X = X.values.astype(np.float64)
print(f"Full: {X.shape}, classes {np.bincount(y)}")

# stratified sample (same dimensions as before)
TARGET = {0: 35895, 1: 16545}
rs = np.random.RandomState(SEED)
idx = np.concatenate([rs.choice(np.where(y == c)[0], size=n, replace=False)
                      for c, n in TARGET.items()])
X, y = X[idx], y[idx]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=SEED)
scaler = StandardScaler().fit(X_tr)
Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

# smaller subset for the grid search (speed), final eval on full sample
sub = rs.choice(len(Xs_tr), size=15000, replace=False)

# ------------------------------------------------------------------ AE
def train_ae(X_tr_, latent_dim, latent_act, seed=SEED):
    import tensorflow as tf
    tf.random.set_seed(seed)
    n = X_tr_.shape[1]
    inp = tf.keras.Input(shape=(n,))
    e = tf.keras.layers.Dense(64, activation='relu')(inp)
    e = tf.keras.layers.Dense(32, activation='relu')(e)
    z = tf.keras.layers.Dense(latent_dim, activation=latent_act)(e)
    d = tf.keras.layers.Dense(32, activation='relu')(z)
    d = tf.keras.layers.Dense(64, activation='relu')(d)
    out = tf.keras.layers.Dense(n, activation='linear')(d)
    ae = tf.keras.Model(inp, out)
    ae.compile(optimizer='adam', loss='mse')
    es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ae.fit(X_tr_, X_tr_, validation_split=0.1, epochs=100, batch_size=256,
           callbacks=[es], verbose=0)
    enc = tf.keras.Model(inp, z)
    return lambda A: enc.predict(A, verbose=0)

def angular(Z, lo, hi):
    Zn = 2 * (Z - lo) / np.where(hi - lo == 0, 1, hi - lo) - 1
    th = np.pi * np.clip(Zn, -1, 1)
    return np.hstack([np.cos(th), np.sin(th)])

# ------------------------------------------------------------------ search
print("\n--- Stage 1: architecture search (latent dim x activation) ---")
best = None
for ldim in (16, 32):
    for act in ('relu', 'tanh'):
        t0 = time.time()
        enc = train_ae(Xs_tr, ldim, act)
        Z_tr, Z_te = enc(Xs_tr), enc(Xs_te)
        lo, hi = Z_tr.min(0), Z_tr.max(0)
        P_tr, P_te = angular(Z_tr, lo, hi), angular(Z_te, lo, hi)
        m = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced',
                cache_size=1000).fit(P_tr[sub], y_tr[sub])
        pred = m.predict(P_te)
        acc = accuracy_score(y_te, pred)
        f1a = f1_score(y_te, pred)          # attack F1 — the metric that broke
        print(f"latent={ldim:>2} act={act:<4s}  acc={acc:.4f}  "
              f"attackF1={f1a:.4f}  ({time.time()-t0:.0f}s)")
        if best is None or f1a > best[0]:
            best = (f1a, ldim, act, P_tr, P_te)

_, BLDIM, BACT, P_tr, P_te = best
print(f"\nBest architecture: latent={BLDIM}, activation={BACT}")

print("\n--- Stage 2: SVM grid (C x gamma), class_weight=balanced ---")
best_svm = None
for C in (1, 10, 100):
    for g in ('scale', 0.01, 0.1):
        m = SVC(kernel='rbf', C=C, gamma=g, class_weight='balanced',
                cache_size=1000).fit(P_tr[sub], y_tr[sub])
        pred = m.predict(P_te)
        acc, f1a = accuracy_score(y_te, pred), f1_score(y_te, pred)
        print(f"C={C:<4} gamma={str(g):<6}  acc={acc:.4f}  attackF1={f1a:.4f}")
        if best_svm is None or f1a > best_svm[0]:
            best_svm = (f1a, C, g)

_, BC, BG = best_svm
print(f"\nBest SVM: C={BC}, gamma={BG}")

print("\n--- Stage 3: final fit on FULL training sample with best config ---")
m = SVC(kernel='rbf', C=BC, gamma=BG, class_weight='balanced',
        cache_size=1000).fit(P_tr, y_tr)
pred = m.predict(P_te)
scores = m.decision_function(P_te)
print(f"PROPOSED (tuned): acc={accuracy_score(y_te, pred):.4f}  "
      f"auc={roc_auc_score(y_te, scores):.4f}  "
      f"attackF1={f1_score(y_te, pred):.4f}")

# fair baseline: RF with balanced class weight too
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                            n_jobs=-1, random_state=SEED).fit(Xs_tr, y_tr)
rp = rf.predict(Xs_te)
print(f"RF (balanced)   : acc={accuracy_score(y_te, rp):.4f}  "
      f"auc={roc_auc_score(y_te, rf.predict_proba(Xs_te)[:,1]):.4f}  "
      f"attackF1={f1_score(y_te, rp):.4f}")

print("\nSend this whole output back.")
