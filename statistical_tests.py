"""Paired statistical comparison (Table 10): Raw vs AE vs Proposed over
identical stratified 5-fold splits on CIC-Bell-DNS-EXF-2021.
Usage: python statistical_tests.py   (expects merged_heavy.csv)"""
import warnings, gc
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

SEED = 42
df = pd.read_csv("merged_heavy.csv", low_memory=False)
y = df['label'].astype(int).values
X = df.drop(columns=['label']).select_dtypes(include=[np.number])
X = X.replace([np.inf,-np.inf], np.nan).fillna(X.median(numeric_only=True)).values.astype(np.float32)
del df; gc.collect()
rs = np.random.RandomState(SEED)
TARGET = {0: 35895, 1: 16545}
idx = np.concatenate([rs.choice(np.where(y==c)[0], size=n, replace=False)
                      for c, n in TARGET.items()])
X, y = X[idx], y[idx]

import tensorflow as tf
def train_enc(Xtr, seed):
    tf.keras.backend.clear_session(); tf.random.set_seed(seed)
    n = Xtr.shape[1]
    inp = tf.keras.Input(shape=(n,))
    e = tf.keras.layers.Dense(64, activation='relu')(inp)
    e = tf.keras.layers.Dense(32, activation='relu')(e)
    z = tf.keras.layers.Dense(16, activation='relu')(e)
    d = tf.keras.layers.Dense(32, activation='relu')(z)
    d = tf.keras.layers.Dense(64, activation='relu')(d)
    out = tf.keras.layers.Dense(n, activation='linear')(d)
    ae = tf.keras.Model(inp, out); ae.compile(optimizer='adam', loss='mse')
    es = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    ae.fit(Xtr, Xtr, validation_split=0.1, epochs=60, batch_size=256,
           callbacks=[es], verbose=0)
    return tf.keras.Model(inp, z)

def svm_metrics(Atr, ytr_, Ate, yte_):
    m = SVC(kernel='rbf', C=10, gamma='scale', cache_size=1000).fit(Atr, ytr_)
    p = m.predict(Ate); s = m.decision_function(Ate)
    return (accuracy_score(yte_, p), roc_auc_score(yte_, s), recall_score(yte_, p))

res = {'Raw': [], 'AE': [], 'QIE': []}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for k, (tr, te) in enumerate(skf.split(X, y)):
    sc = StandardScaler().fit(X[tr])
    Xtr = sc.transform(X[tr]).astype(np.float32)
    Xte = sc.transform(X[te]).astype(np.float32)
    res['Raw'].append(svm_metrics(Xtr, y[tr], Xte, y[te]))
    enc = train_enc(Xtr, SEED + k)
    Ztr = enc.predict(Xtr, batch_size=4096, verbose=0)
    Zte = enc.predict(Xte, batch_size=4096, verbose=0)
    res['AE'].append(svm_metrics(Ztr, y[tr], Zte, y[te]))
    lo, hi = Ztr.min(0), Ztr.max(0)
    def q(Z):
        Zn = 2*(Z-lo)/np.where(hi-lo==0, 1, hi-lo) - 1
        th = np.pi*np.clip(Zn, -1, 1)
        return np.hstack([np.cos(th), np.sin(th)]).astype(np.float32)
    res['QIE'].append(svm_metrics(q(Ztr), y[tr], q(Zte), y[te]))
    print(f"fold {k+1}: Raw={res['Raw'][-1][0]:.4f} "
          f"AE={res['AE'][-1][0]:.4f} QIE={res['QIE'][-1][0]:.4f}")
    del enc, Ztr, Zte; gc.collect()

print("\n" + "="*76)
print("PAIRED STATISTICAL ANALYSIS (identical 5 folds) — paper Table 10")
print("="*76)
for mi, mn in enumerate(['Accuracy', 'AUC-ROC', 'Attack Recall']):
    print(f"\n--- {mn} ---")
    for cfg in res:
        v = np.array([r[mi] for r in res[cfg]])
        ci = stats.t.interval(0.95, len(v)-1, loc=v.mean(), scale=stats.sem(v))
        print(f"{cfg:<5s} mean={v.mean():.4f} ± {v.std():.4f}  "
              f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    a = np.array([r[mi] for r in res['Raw']])
    b = np.array([r[mi] for r in res['QIE']])
    t, pt = stats.ttest_rel(b, a)
    try: w, pw = stats.wilcoxon(b, a)
    except Exception: pw = float('nan')
    print(f"QIE vs Raw: paired t-test p={pt:.4f} | Wilcoxon p={pw:.4f}")
