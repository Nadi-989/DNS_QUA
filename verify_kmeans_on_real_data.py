"""
Verification script — run this on your real CIC-Bell-DNS-EXF-2021 data.

It reproduces the paper's pipeline and tests the three K-Means scenarios:
  A) Phi(z) only                     -> the "exploratory K-Means" reading
  B) Phi(z) + centroid distances     -> K-Means feeds the SVM (distances)
  C) Phi(z) + one-hot cluster id     -> K-Means feeds the SVM (labels)

Whichever variant reproduces YOUR reported numbers (99.52 / 99.69) tells
you what your original experiment actually did.

USAGE:
    python verify_kmeans_on_real_data.py <preprocessed_data.csv> <label_column>

Assumes the CSV is the preprocessed numeric feature table (after the
Section-3.2 steps), with a binary label column (0 = normal, 1 = attack).
Requires: numpy, pandas, scikit-learn, tensorflow (or edit build_ae()).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------- data
csv_path, label_col = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path)
y = df[label_col].astype(int).values
X = df.drop(columns=[label_col]).values.astype(np.float64)
print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, "
      f"class balance: {np.bincount(y)}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

scaler = StandardScaler().fit(X_tr)            # train only — no leakage
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

# ---------------------------------------------------------------- autoencoder
def build_and_encode(X_tr, X_te):
    """n -> 64 -> 32 -> 16 -> 32 -> 64 -> n, Adam + MSE + early stopping."""
    import tensorflow as tf
    tf.random.set_seed(SEED)
    n = X_tr.shape[1]
    inp = tf.keras.Input(shape=(n,))
    e = tf.keras.layers.Dense(64, activation='relu')(inp)
    e = tf.keras.layers.Dense(32, activation='relu')(e)
    z = tf.keras.layers.Dense(16, activation='relu', name='latent')(e)
    d = tf.keras.layers.Dense(32, activation='relu')(z)
    d = tf.keras.layers.Dense(64, activation='relu')(d)
    out = tf.keras.layers.Dense(n, activation='linear')(d)
    ae = tf.keras.Model(inp, out)
    ae.compile(optimizer='adam', loss='mse')
    es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ae.fit(X_tr, X_tr, validation_split=0.1, epochs=100, batch_size=256,
           callbacks=[es], verbose=0)
    enc = tf.keras.Model(inp, z)
    return enc.predict(X_tr, verbose=0), enc.predict(X_te, verbose=0)

Z_tr, Z_te = build_and_encode(X_tr, X_te)

# ---------------------------------------------------------------- angular map
lo, hi = Z_tr.min(0), Z_tr.max(0)              # normalization from train only
def angular(Z):
    Zn = 2 * (Z - lo) / np.where(hi - lo == 0, 1, hi - lo) - 1
    th = np.pi * np.clip(Zn, -1, 1)
    return np.hstack([np.cos(th), np.sin(th)])
P_tr, P_te = angular(Z_tr), angular(Z_te)

# ---------------------------------------------------------------- K-Means
km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(Z_tr)
dsc = StandardScaler().fit(km.transform(Z_tr))
D_tr = dsc.transform(km.transform(Z_tr)); D_te = dsc.transform(km.transform(Z_te))
H_tr = np.eye(2)[km.labels_];             H_te = np.eye(2)[km.predict(Z_te)]

# ---------------------------------------------------------------- variants
def evaluate(name, Xtr, Xte):
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
              random_state=SEED).fit(Xtr, y_tr)
    pred = svm.predict(Xte)
    acc = accuracy_score(y_te, pred)
    auc = roc_auc_score(y_te, svm.predict_proba(Xte)[:, 1])
    cm = confusion_matrix(y_te, pred)
    print(f"\n{name}\n  accuracy = {acc:.4f}   AUC-ROC = {auc:.4f}")
    print(f"  confusion matrix:\n{cm}")
    return acc

print("=" * 70)
a = evaluate("A) Phi(z) -> SVM   [K-Means exploratory only]", P_tr, P_te)
b = evaluate("B) [Phi(z); centroid distances] -> SVM",
             np.hstack([P_tr, D_tr]), np.hstack([P_te, D_te]))
c = evaluate("C) [Phi(z); one-hot cluster id] -> SVM",
             np.hstack([P_tr, H_tr]), np.hstack([P_te, H_te]))
print("=" * 70)
print(f"\nDeltas vs A:   B {b-a:+.4f}   C {c-a:+.4f}")
print("Compare each variant's accuracy with your reported 99.52% / 99.69%")
print("to identify what your original experiment actually computed.")
