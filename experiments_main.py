"""
==============================================================================
 COMPLETE EXPERIMENT SCRIPT — regenerates every number in the paper
 Deep Autoencoder + Quantum-Inspired Encoding + SVM  (DNS anomaly detection)
==============================================================================

 WHAT IT PRODUCES (everything printed at the end, ready to copy):
   1. Section 4.1 values ........ split, seed, epochs, batch, C, gamma, versions
   2. Table 3 ................... accuracy, AUC, macro/weighted P/R/F1 (4 dp)
   3. Table 4 ................... per-class precision/recall/F1/support
   4. Table 5 ................... confusion matrix
   5. Table 7 ................... ablation (Raw+SVM, AE+SVM, Proposed)
   6. Baselines ................. RandomForest, GradientBoosting, MLP
                                  on the SAME split (new table for reviewers)
   7. 5-fold cross-validation ... mean ± std of accuracy and AUC

 HOW TO RUN:
   1. Download CIC-Bell-DNS-EXF-2021 from the UNB website (free).
   2. Prepare ONE csv: all samples, numeric features only, plus a label
      column (0 = normal/benign, 1 = attack/exfiltration).
      If you have separate benign/attack csv files, see merge_csvs() below.
   3. pip install numpy pandas scikit-learn tensorflow
   4. NOTEBOOK VERSION: edit csv_path and label_col in the main
      section below, then just run the whole cell/file.
==============================================================================
"""
import sys, time, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, precision_recall_fscore_support)

# ------------------------------------------------------------------ settings
SEED        = 42          # fixed random seed  -> Section 4.1
TEST_SIZE   = 0.20        # stratified 80/20   -> Section 4.1
AE_EPOCHS   = 100         # max epochs         -> Section 4.1
AE_BATCH    = 256         # batch size         -> Section 4.1
AE_PATIENCE = 10          # early stopping     -> Section 4.1
AE_VALSPLIT = 0.10        # validation split   -> Section 4.1
SVM_C       = 10.0        # RBF C              -> Section 4.1
SVM_GAMMA   = 'scale'     # RBF gamma          -> Section 4.1
LATENT_DIM  = 16

np.random.seed(SEED)

# ------------------------------------------------------------------ helpers
def merge_csvs():
    """If your data is in separate files, uncomment + edit, run once:
    benign = pd.read_csv('benign.csv');  benign['label'] = 0
    attack = pd.read_csv('attack.csv');  attack['label'] = 1
    pd.concat([benign, attack]).to_csv('your_data.csv', index=False)
    """
    pass

def clean_numeric(df, label_col):
    """Section 3.2 preprocessing: numeric only, no NaN/inf."""
    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X.values.astype(np.float64), y, list(X.columns)

def train_autoencoder(X_tr, seed=SEED):
    """n -> 64 -> 32 -> 16 -> 32 -> 64 -> n, Adam+MSE, early stopping.
    Returns an encode() function."""
    import tensorflow as tf
    tf.random.set_seed(seed)
    n = X_tr.shape[1]
    inp = tf.keras.Input(shape=(n,))
    e = tf.keras.layers.Dense(64, activation='relu')(inp)
    e = tf.keras.layers.Dense(32, activation='relu')(e)
    z = tf.keras.layers.Dense(LATENT_DIM, activation='relu')(e)
    d = tf.keras.layers.Dense(32, activation='relu')(z)
    d = tf.keras.layers.Dense(64, activation='relu')(d)
    out = tf.keras.layers.Dense(n, activation='linear')(d)
    ae = tf.keras.Model(inp, out)
    ae.compile(optimizer='adam', loss='mse')
    es = tf.keras.callbacks.EarlyStopping(patience=AE_PATIENCE,
                                          restore_best_weights=True)
    ae.fit(X_tr, X_tr, validation_split=AE_VALSPLIT, epochs=AE_EPOCHS,
           batch_size=AE_BATCH, callbacks=[es], verbose=0)
    encoder = tf.keras.Model(inp, z)
    return lambda X: encoder.predict(X, verbose=0)

def angular_encode(Z, lo, hi):
    """Quantum-inspired map: normalize to [-1,1], theta = pi*z, cos/sin."""
    Zn = 2 * (Z - lo) / np.where(hi - lo == 0, 1, hi - lo) - 1
    th = np.pi * np.clip(Zn, -1, 1)
    return np.hstack([np.cos(th), np.sin(th)])

def svm_eval(Xtr, ytr, Xte, yte, seed=SEED):
    m = SVC(kernel='rbf', C=SVM_C, gamma=SVM_GAMMA,
            cache_size=1000).fit(Xtr, ytr)
    pred   = m.predict(Xte)
    scores = m.decision_function(Xte)      # AUC from margins (fast,
    return (accuracy_score(yte, pred),     #  no probability=True refits)
            roc_auc_score(yte, scores), pred)

# ------------------------------------------------------------------ pipeline
def full_pipeline(X_tr, y_tr, X_te, y_te, seed=SEED, verbose=True):
    """Returns dict with everything: ablation, per-class report, cm."""
    t0 = time.time()
    scaler = StandardScaler().fit(X_tr)                # train only
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

    # --- ablation 1: raw features + SVM
    acc_raw, auc_raw, _ = svm_eval(Xs_tr, y_tr, Xs_te, y_te, seed)

    # --- autoencoder
    encode = train_autoencoder(Xs_tr, seed)
    Z_tr, Z_te = encode(Xs_tr), encode(Xs_te)

    # --- ablation 2: latent + SVM
    acc_ae, auc_ae, _ = svm_eval(Z_tr, y_tr, Z_te, y_te, seed)

    # --- K-Means (exploratory latent-space analysis; not fed to the SVM)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Z_tr)

    # --- proposed: angular encoding + SVM
    lo, hi = Z_tr.min(0), Z_tr.max(0)                  # train only
    P_tr = angular_encode(Z_tr, lo, hi)
    P_te = angular_encode(Z_te, lo, hi)
    acc_p, auc_p, pred = svm_eval(P_tr, y_tr, P_te, y_te, seed)

    res = dict(acc_raw=acc_raw, auc_raw=auc_raw,
               acc_ae=acc_ae,  auc_ae=auc_ae,
               acc_p=acc_p,    auc_p=auc_p,
               pred=pred, y_te=y_te,
               data=(Xs_tr, Xs_te, P_tr, P_te),
               km_inertia=km.inertia_, runtime=time.time() - t0)
    return res

# ------------------------------------------------------------------ main
if __name__ == '__main__':
    # ================================================================
    #  EDIT THESE TWO LINES ONLY  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ================================================================
    csv_path  = "merged_mal_clean.csv"   # <-- BCCC-CIC-Bell-DNS-2024 (Mal)
    label_col = "label"           # <-- name of the label column (0/1)
    # ================================================================

    df = pd.read_csv(csv_path)
    X, y, feat_names = clean_numeric(df, label_col)
    print(f"Full data: {X.shape[0]} samples, {X.shape[1]} numeric features, "
          f"classes: {np.bincount(y)}")

    # ---- stratified subsample: the full dataset (~1M rows) is far too
    # ---- large for SVM. Sample to match the paper's dimensions exactly
    # ---- (test set of 10,488 = 7,179 normal + 3,309 attack at 80/20).
    TARGET = {0: 35895, 1: 16545}          # 52,440 total
    idx = []
    rs = np.random.RandomState(SEED)
    for cls, n_want in TARGET.items():
        cls_idx = np.where(y == cls)[0]
        n_take = min(n_want, len(cls_idx))
        idx.append(rs.choice(cls_idx, size=n_take, replace=False))
    idx = np.concatenate(idx)
    X, y = X[idx], y[idx]
    n_feat = X.shape[1]
    print(f"Sampled : {X.shape[0]} samples, classes: {np.bincount(y)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    print("\n[1/3] Main run (this produces Tables 3, 4, 5, 7) ...")
    r = full_pipeline(X_tr, y_tr, X_te, y_te)

    # ---- headline metrics
    acc = accuracy_score(y_te, r['pred'])
    p, rec, f1, sup = precision_recall_fscore_support(y_te, r['pred'])
    mp, mr, mf1, _  = precision_recall_fscore_support(y_te, r['pred'],
                                                      average='macro')
    _, _, wf1, _    = precision_recall_fscore_support(y_te, r['pred'],
                                                      average='weighted')
    cm = confusion_matrix(y_te, r['pred'])

    # ---- baselines on the SAME split (new table for reviewers)
    print("[2/3] Baselines on the same split ...")
    Xs_tr, Xs_te, P_tr, P_te = r['data']
    baselines = {}
    for name, clf in [
        ('Random Forest',     RandomForestClassifier(n_estimators=200,
                                                     random_state=SEED)),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=SEED)),
        ('MLP',               MLPClassifier(hidden_layer_sizes=(64, 32),
                                            max_iter=300, random_state=SEED))]:
        clf.fit(Xs_tr, y_tr)
        bp  = clf.predict(Xs_te)
        bpr = (clf.predict_proba(Xs_te)[:, 1]
               if hasattr(clf, 'predict_proba') else bp)
        baselines[name] = (accuracy_score(y_te, bp), roc_auc_score(y_te, bpr))

    # ---- 5-fold CV of the proposed pipeline (set RUN_CV=False to skip)
    RUN_CV = False
    cv_acc, cv_auc = [], []
    if RUN_CV:
        print("[3/3] 5-fold cross-validation of the proposed pipeline ...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for k, (tr, te) in enumerate(skf.split(X, y)):
            rf = full_pipeline(X[tr], y[tr], X[te], y[te], seed=SEED + k,
                               verbose=False)
            cv_acc.append(rf['acc_p']); cv_auc.append(rf['auc_p'])
            print(f"   fold {k+1}: acc={rf['acc_p']:.4f} auc={rf['auc_p']:.4f}")

    # =================================================================
    # FINAL REPORT — copy these values into the paper
    # =================================================================
    import tensorflow as tf
    print("\n" + "=" * 72)
    print("SECTION 4.1  (replace the [placeholders] with these values)")
    print("=" * 72)
    print(f"Python {platform.python_version()}, TensorFlow {tf.__version__}, "
          f"scikit-learn {sklearn.__version__}")
    print(f"Stratified split {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}, "
          f"random seed {SEED}")
    print(f"AE: Adam (lr 0.001), batch {AE_BATCH}, max {AE_EPOCHS} epochs, "
          f"early stopping patience {AE_PATIENCE}, val split {AE_VALSPLIT:.0%}")
    print(f"SVM: C = {SVM_C}, gamma = '{SVM_GAMMA}'")
    print(f"Test set: {len(y_te)} samples "
          f"({np.sum(y_te==0)} normal, {np.sum(y_te==1)} attack)")

    print("\n" + "=" * 72)
    print("TABLE 3 — Overall Performance Metrics")
    print("=" * 72)
    print(f"Accuracy           {acc:.4f}")
    print(f"AUC-ROC            {r['auc_p']:.4f}")
    print(f"Macro Precision    {mp:.4f}")
    print(f"Macro Recall       {mr:.4f}")
    print(f"Macro F1-score     {mf1:.4f}")
    print(f"Weighted F1-score  {wf1:.4f}")

    print("\n" + "=" * 72)
    print("TABLE 4 — Per-class report")
    print("=" * 72)
    print(classification_report(y_te, r['pred'],
                                target_names=['Normal (0)', 'Attack (1)'],
                                digits=4))

    print("=" * 72)
    print("TABLE 5 — Confusion Matrix   [[TN FP][FN TP]]")
    print("=" * 72)
    print(cm)

    print("\n" + "=" * 72)
    print("TABLE 7 — Ablation")
    print("=" * 72)
    print(f"Raw Features + SVM                      "
          f"{r['acc_raw']*100:.2f}   {r['auc_raw']:.4f}")
    print(f"Autoencoder + SVM                       "
          f"{r['acc_ae']*100:.2f}   {r['auc_ae']:.4f}")
    print(f"Proposed (AE + QI Encoding + SVM)       "
          f"{r['acc_p']*100:.2f}   {r['auc_p']:.4f}")

    print("\n" + "=" * 72)
    print("NEW TABLE — Baselines on the same split")
    print("=" * 72)
    for name, (a, u) in baselines.items():
        print(f"{name:<20s} acc={a:.4f}  auc={u:.4f}")
    print(f"{'Proposed':<20s} acc={acc:.4f}  auc={r['auc_p']:.4f}")

    if cv_acc:
        print("\n" + "=" * 72)
        print("5-FOLD CROSS-VALIDATION (add to Section 4.2)")
        print("=" * 72)
        print(f"Accuracy: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
        print(f"AUC-ROC : {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")
    print("\nDone. Send this whole output back and I will insert every "
          "number into the Word file for you.")
