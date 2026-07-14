"""Classifier tournament on BCCC-CIC-Bell-DNS-2024 (paper Table 8 / Figure 5):
4 classifier families x {raw features, proposed AE+QIE features}.
Usage: python model_tournament.py   (expects merged_mal_clean.csv)"""
import warnings, time, gc
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

SEED = 42
np.random.seed(SEED)
df = pd.read_csv("merged_mal_clean.csv", low_memory=False)
y = df['label'].astype(int).values
X = df.drop(columns=['label']).select_dtypes(include=[np.number])
X = X.replace([np.inf,-np.inf], np.nan).fillna(X.median(numeric_only=True)).values.astype(np.float32)
del df; gc.collect()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                      stratify=y, random_state=SEED)
del X; gc.collect()
sc = StandardScaler().fit(Xtr)
Xs_tr = sc.transform(Xtr).astype(np.float32)
Xs_te = sc.transform(Xte).astype(np.float32)
del Xtr, Xte; gc.collect()

import tensorflow as tf
tf.random.set_seed(SEED)
rs = np.random.RandomState(SEED)
ae_sub = rs.choice(len(Xs_tr), size=100_000, replace=False)
n = Xs_tr.shape[1]
inp = tf.keras.Input(shape=(n,))
e = tf.keras.layers.Dense(64, activation='relu')(inp)
e = tf.keras.layers.Dense(32, activation='relu')(e)
z = tf.keras.layers.Dense(32, activation='tanh')(e)
d = tf.keras.layers.Dense(32, activation='relu')(z)
d = tf.keras.layers.Dense(64, activation='relu')(d)
out = tf.keras.layers.Dense(n, activation='linear')(d)
ae = tf.keras.Model(inp, out); ae.compile(optimizer='adam', loss='mse')
es = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
ae.fit(Xs_tr[ae_sub], Xs_tr[ae_sub], validation_split=0.1, epochs=40,
       batch_size=512, callbacks=[es], verbose=0)
enc = tf.keras.Model(inp, z)
Z_tr = enc.predict(Xs_tr, batch_size=4096, verbose=0)
Z_te = enc.predict(Xs_te, batch_size=4096, verbose=0)
lo, hi = Z_tr.min(0), Z_tr.max(0)
def qie(Z):
    Zn = 2*(Z-lo)/np.where(hi-lo==0, 1, hi-lo) - 1
    th = np.pi*np.clip(Zn, -1, 1)
    return np.hstack([np.cos(th), np.sin(th)]).astype(np.float32)
P_tr, P_te = qie(Z_tr), qie(Z_te)
del Z_tr, Z_te; gc.collect()

def make(name):
    if name == 'RandomForest':
        return RandomForestClassifier(n_estimators=300, n_jobs=-1,
                 class_weight='balanced', random_state=SEED)
    if name == 'HistGradBoost':
        return HistGradientBoostingClassifier(max_iter=400,
                 class_weight='balanced', random_state=SEED)
    if name == 'XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1,
                 n_jobs=-1, random_state=SEED, eval_metric='logloss',
                 scale_pos_weight=(ytr==0).sum()/(ytr==1).sum(),
                 tree_method='hist')
    if name == 'LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, n_jobs=-1,
                 class_weight='balanced', random_state=SEED, verbose=-1)

results = {}
for name in ['RandomForest', 'HistGradBoost', 'XGBoost', 'LightGBM']:
    for feats, Atr, Ate in [('RAW', Xs_tr, Xs_te), ('+QIE', P_tr, P_te)]:
        try:
            t0 = time.time()
            m = make(name); m.fit(Atr, ytr)
            pred = m.predict(Ate); proba = m.predict_proba(Ate)[:, 1]
            key = f"{name} {feats}"
            results[key] = (accuracy_score(yte, pred),
                            roc_auc_score(yte, proba), f1_score(yte, pred))
            a, u, f = results[key]
            print(f"{key:<22s} acc={a:.4f} auc={u:.4f} attackF1={f:.4f} "
                  f"({time.time()-t0:.0f}s)")
            del m; gc.collect()
        except Exception as ex:
            print(f"{name} {feats}: error — {str(ex)[:60]}")

print("\n=== FINAL RANKING (by attack F1) ===")
for k, (a, u, f) in sorted(results.items(), key=lambda kv: -kv[1][2]):
    print(f"{k:<22s} acc={a:.4f} auc={u:.4f} attackF1={f:.4f}")
