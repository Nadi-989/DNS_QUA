"""20-trial TPE Bayesian optimization over the joint AE + RF space
(paper Section 4.6). Usage: python optuna_search.py  (expects merged_mal_clean.csv)"""
import warnings, gc
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
df = pd.read_csv("merged_mal_clean.csv", low_memory=False)
y = df['label'].astype(int).values
X = df.drop(columns=['label']).select_dtypes(include=[np.number])
X = X.replace([np.inf,-np.inf], np.nan).fillna(X.median(numeric_only=True)).values.astype(np.float32)
del df; gc.collect()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
del X; gc.collect()
sc = StandardScaler().fit(Xtr)
Xs_tr = sc.transform(Xtr).astype(np.float32); Xs_te = sc.transform(Xte).astype(np.float32)
del Xtr, Xte; gc.collect()
rs = np.random.RandomState(SEED)
dev = rs.choice(len(Xs_tr), size=110_000, replace=False)
Xd, yd = Xs_tr[dev], ytr[dev]
Xd_tr, Xd_va, yd_tr, yd_va = train_test_split(Xd, yd, test_size=0.25, stratify=yd, random_state=SEED)

import tensorflow as tf
def build_encode(h1, h2, ldim, drop, act, epochs=25):
    tf.keras.backend.clear_session(); tf.random.set_seed(SEED)
    n = Xd_tr.shape[1]
    inp = tf.keras.Input(shape=(n,))
    e = tf.keras.layers.Dense(h1, activation='relu')(inp)
    e = tf.keras.layers.Dropout(drop)(e)
    e = tf.keras.layers.Dense(h2, activation='relu')(e)
    z = tf.keras.layers.Dense(ldim, activation=act)(e)
    d = tf.keras.layers.Dense(h2, activation='relu')(z)
    d = tf.keras.layers.Dense(h1, activation='relu')(d)
    out = tf.keras.layers.Dense(n, activation='linear')(d)
    ae = tf.keras.Model(inp, out); ae.compile(optimizer='adam', loss='mse')
    es = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
    ae.fit(Xd_tr, Xd_tr, validation_split=0.1, epochs=epochs, batch_size=512,
           callbacks=[es], verbose=0)
    return tf.keras.Model(inp, z)

def objective(trial):
    h1 = trial.suggest_categorical('h1', [64, 128, 256])
    h2 = trial.suggest_categorical('h2', [32, 64])
    ldim = trial.suggest_categorical('ldim', [24, 32, 48])
    drop = trial.suggest_float('drop', 0.0, 0.3)
    n_est = trial.suggest_int('n_est', 150, 400, step=50)
    max_d = trial.suggest_int('max_d', 10, 30)
    enc = build_encode(h1, h2, ldim, drop, 'tanh')
    Z1 = enc.predict(Xd_tr, batch_size=4096, verbose=0)
    Z2 = enc.predict(Xd_va, batch_size=4096, verbose=0)
    lo, hi = Z1.min(0), Z1.max(0)
    def q(Z):
        Zn = 2*(Z-lo)/np.where(hi-lo==0, 1, hi-lo) - 1
        th = np.pi*np.clip(Zn, -1, 1)
        return np.hstack([np.cos(th), np.sin(th)]).astype(np.float32)
    m = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, n_jobs=-1,
                               class_weight='balanced', random_state=SEED)
    m.fit(q(Z1), yd_tr)
    f1 = f1_score(yd_va, m.predict(q(Z2)))
    del enc, Z1, Z2, m; gc.collect()
    return f1

study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=20, show_progress_bar=True)
print("Best params:", study.best_params, "| val F1:", round(study.best_value, 4))

bp = study.best_params
enc = build_encode(bp['h1'], bp['h2'], bp['ldim'], bp['drop'], 'tanh', epochs=40)
Z_tr = enc.predict(Xs_tr, batch_size=4096, verbose=0)
Z_te = enc.predict(Xs_te, batch_size=4096, verbose=0)
lo, hi = Z_tr.min(0), Z_tr.max(0)
def q(Z):
    Zn = 2*(Z-lo)/np.where(hi-lo==0, 1, hi-lo) - 1
    th = np.pi*np.clip(Zn, -1, 1)
    return np.hstack([np.cos(th), np.sin(th)]).astype(np.float32)
m = RandomForestClassifier(n_estimators=bp['n_est'], max_depth=bp['max_d'],
        n_jobs=-1, class_weight='balanced', random_state=SEED)
m.fit(q(Z_tr), ytr)
pred = m.predict(q(Z_te)); proba = m.predict_proba(q(Z_te))[:, 1]
print(f"OPTIMIZED PROPOSED: acc={accuracy_score(yte,pred):.4f} "
      f"auc={roc_auc_score(yte,proba):.4f} attackF1={f1_score(yte,pred):.4f}")
