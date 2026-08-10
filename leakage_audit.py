"""Leakage audit, deterministic label ceiling, and group-disjoint re-evaluation.

Addresses:
  * Reviewer 1, comment 10  - the deterministic majority-label upper bound, computed by
                              grouping identical feature vectors, reported separately from
                              any classifier's empirical accuracy.
  * Reviewer 2, comment 2   - overlap statistics before and after splitting, a group-disjoint
                              split in which no group of identical vectors crosses the
                              train/test boundary, and an explicit separation of
                              preprocessing leakage, duplicate leakage and entity-level leakage.

Usage
-----
    # fast: audit + ceiling + overlap statistics only (no model training)
    python leakage_audit.py merged_heavy.csv label

    # additionally re-evaluate the pipeline under a group-disjoint split
    python leakage_audit.py merged_heavy.csv label --grouped-eval

    # treat vectors as identical after rounding to 6 decimals (near-duplicate analysis)
    python leakage_audit.py merged_heavy.csv label --round 6

    # also report entity-level overlap for a categorical column
    python leakage_audit.py merged_heavy.csv label --entity-col domain

Every number printed here is written to leakage_audit_results.json.
"""
import argparse, json, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

SEED = 42


# ----------------------------------------------------------------------------- helpers
def load(path, label_col):
    df = pd.read_csv(path, low_memory=False)
    if label_col not in df.columns:
        sys.exit(f"label column '{label_col}' not found; available: {list(df.columns)[:20]} ...")
    y = df[label_col].astype(int).to_numpy()
    Xdf = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(Xdf.median(numeric_only=True))
    return df, Xdf, y


def group_ids(X, decimals=None):
    """Assign an integer id to every distinct feature vector.

    decimals=None  -> exact equality (the reviewer's 'identical feature vectors')
    decimals=k     -> equality after rounding to k decimals (a defined equivalence
                      relation for the near-duplicate case)
    """
    A = X if decimals is None else np.round(X, decimals)
    _, inverse = np.unique(A, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def deterministic_ceiling(gid, y):
    """Majority-label upper bound: sum over groups of the most frequent label count,
    divided by the number of rows. This is a property of the data, not of any model."""
    n_groups = gid.max() + 1
    total = np.bincount(gid, minlength=n_groups).astype(np.int64)
    pos = np.bincount(gid, weights=(y == 1), minlength=n_groups).astype(np.int64)
    neg = total - pos
    majority = np.maximum(pos, neg)
    conflicting = (pos > 0) & (neg > 0)
    return dict(
        n_rows=int(len(y)),
        n_groups=int(n_groups),
        duplicate_rate=round(1 - n_groups / len(y), 4),
        conflicting_groups=int(conflicting.sum()),
        rows_in_conflicting_groups=int(total[conflicting].sum()),
        rows_in_conflicting_groups_pct=round(100 * total[conflicting].sum() / len(y), 2),
        deterministic_upper_bound=round(float(majority.sum() / len(y)), 4),
    )


def overlap_stats(gid, tr_idx, te_idx):
    g_tr, g_te = gid[tr_idx], gid[te_idx]
    shared = np.intersect1d(np.unique(g_tr), np.unique(g_te))
    in_shared = np.isin(g_te, shared)
    return dict(
        n_train=int(len(tr_idx)),
        n_test=int(len(te_idx)),
        shared_groups=int(len(shared)),
        test_rows_whose_vector_also_in_train=int(in_shared.sum()),
        test_rows_leaked_pct=round(100 * in_shared.sum() / len(te_idx), 2),
    )


def entity_overlap(series, tr_idx, te_idx):
    a, b = set(series.iloc[tr_idx].dropna()), set(series.iloc[te_idx].dropna())
    shared = a & b
    leaked = series.iloc[te_idx].isin(shared).sum()
    return dict(
        distinct_entities_train=len(a), distinct_entities_test=len(b),
        shared_entities=len(shared),
        test_rows_with_entity_seen_in_train=int(leaked),
        test_rows_with_entity_seen_in_train_pct=round(100 * leaked / len(te_idx), 2),
    )


def show(title, d):
    print("\n" + title)
    print("-" * len(title))
    for k, v in d.items():
        print(f"  {k:<45s} {v}")


# ----------------------------------------------------------------------------- optional model eval
def evaluate(Xtr, Xte, ytr, yte, use_ae=True):
    """Re-run the paper's pipeline (scaler -> AE -> angular encoding -> classifier),
    fitting every stage on the training partition only. Falls back to a Random Forest
    on raw features if TensorFlow is unavailable."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score,
                                 precision_score, f1_score)

    sc = StandardScaler().fit(Xtr)
    A_tr, A_te = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)

    if use_ae:
        try:
            import tensorflow as tf
            tf.random.set_seed(SEED)
            n = A_tr.shape[1]
            latent = 32 if n > 50 else 16
            inp = tf.keras.Input(shape=(n,))
            e = tf.keras.layers.Dense(64, activation='relu')(inp)
            e = tf.keras.layers.Dense(32, activation='relu')(e)
            z = tf.keras.layers.Dense(latent, activation='tanh')(e)
            d = tf.keras.layers.Dense(32, activation='relu')(z)
            d = tf.keras.layers.Dense(64, activation='relu')(d)
            out = tf.keras.layers.Dense(n, activation='linear')(d)
            ae = tf.keras.Model(inp, out)
            ae.compile(optimizer='adam', loss='mse')
            es = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
            sub = np.random.RandomState(SEED).choice(
                len(A_tr), size=min(100_000, len(A_tr)), replace=False)
            ae.fit(A_tr[sub], A_tr[sub], validation_split=0.1, epochs=40,
                   batch_size=512, callbacks=[es], verbose=0)
            enc = tf.keras.Model(inp, z)
            Z_tr = enc.predict(A_tr, batch_size=4096, verbose=0)
            Z_te = enc.predict(A_te, batch_size=4096, verbose=0)
            lo, hi = Z_tr.min(0), Z_tr.max(0)          # bounds from TRAINING data only

            def qie(Z):
                Zn = 2 * (Z - lo) / np.where(hi - lo == 0, 1, hi - lo) - 1
                th = np.pi * np.clip(Zn, -1, 1)        # clipping: no periodic aliasing
                return np.hstack([np.cos(th), np.sin(th)]).astype(np.float32)

            A_tr, A_te = qie(Z_tr), qie(Z_te)
        except ImportError:
            print("  [tensorflow not available - evaluating on raw standardized features]")

    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                 class_weight='balanced', random_state=SEED).fit(A_tr, ytr)
    pred = clf.predict(A_te)
    proba = clf.predict_proba(A_te)[:, 1]
    return dict(
        accuracy=round(accuracy_score(yte, pred), 4),
        auc_roc=round(roc_auc_score(yte, proba), 4),
        attack_recall=round(recall_score(yte, pred, pos_label=1), 4),
        attack_precision=round(precision_score(yte, pred, pos_label=1, zero_division=0), 4),
        attack_f1=round(f1_score(yte, pred, pos_label=1, zero_division=0), 4),
        false_positive_rate=round(float(((pred == 1) & (yte == 0)).sum() / max((yte == 0).sum(), 1)), 4),
    )


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("label_col")
    ap.add_argument("--round", type=int, default=None,
                    help="treat vectors equal after rounding to this many decimals")
    ap.add_argument("--entity-col", default=None,
                    help="categorical column (domain, capture session, source host, ...)")
    ap.add_argument("--grouped-eval", action="store_true",
                    help="also retrain and evaluate under both splits")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df, Xdf, y = load(args.csv, args.label_col)
    X = Xdf.to_numpy(dtype=np.float64)
    print(f"Loaded {args.csv}: {X.shape[0]:,} rows x {X.shape[1]} numeric features; "
          f"attack rate {y.mean():.4f}")

    results = {"file": args.csv, "seed": SEED, "rounding": args.round}

    # ---- 1. duplication, conflicts, deterministic ceiling (exact equality) -------------
    gid_exact = group_ids(X, decimals=None)
    ceil_exact = deterministic_ceiling(gid_exact, y)
    show("1. Data-level audit, exact identical feature vectors", ceil_exact)
    results["exact"] = ceil_exact

    # ---- 1b. same under a near-duplicate equivalence ----------------------------------
    if args.round is not None:
        gid_round = group_ids(X, decimals=args.round)
        ceil_round = deterministic_ceiling(gid_round, y)
        show(f"1b. Same audit under equality after rounding to {args.round} decimals", ceil_round)
        results["rounded"] = ceil_round
        gid = gid_round
    else:
        gid = gid_exact

    # ---- 2. conventional stratified random split -------------------------------------
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=args.test_size, stratify=y, random_state=SEED)
    ov_random = overlap_stats(gid, tr, te)
    show("2. Overlap under the conventional stratified random split (duplicate leakage)", ov_random)
    results["random_split_overlap"] = ov_random

    # ---- 3. group-disjoint split ------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=SEED)
    tr_g, te_g = next(gss.split(idx, y, groups=gid))
    ov_group = overlap_stats(gid, tr_g, te_g)
    ov_group["test_attack_rate"] = round(float(y[te_g].mean()), 4)
    ov_group["train_attack_rate"] = round(float(y[tr_g].mean()), 4)
    show("3. Overlap under a group-disjoint split (no group crosses the boundary)", ov_group)
    results["group_split_overlap"] = ov_group

    # ---- 4. entity-level -------------------------------------------------------------
    if args.entity_col:
        if args.entity_col not in df.columns:
            print(f"\n[warn] entity column '{args.entity_col}' not in file - skipped")
        else:
            ent = entity_overlap(df[args.entity_col], tr, te)
            show(f"4. Entity-level overlap on '{args.entity_col}' under the random split", ent)
            results["entity_overlap_random_split"] = ent

    # ---- 5. optional re-evaluation ----------------------------------------------------
    if args.grouped_eval:
        print("\n5. Re-evaluation of the pipeline under both splits (this takes a while)")
        m_rand = evaluate(X[tr], X[te], y[tr], y[te])
        show("   5a. Conventional stratified random split", m_rand)
        m_grp = evaluate(X[tr_g], X[te_g], y[tr_g], y[te_g])
        show("   5b. Group-disjoint split", m_grp)
        results["metrics_random_split"] = m_rand
        results["metrics_group_split"] = m_grp
        delta = {k: round(m_grp[k] - m_rand[k], 4) for k in m_rand}
        show("   5c. Change (group-disjoint minus random)", delta)
        results["metrics_delta"] = delta

    with open("leakage_audit_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 78)
    print("Summary for the manuscript")
    print("=" * 78)
    c = results.get("rounded", results["exact"])
    print(f"  Deterministic majority-label upper bound : {c['deterministic_upper_bound']:.4f} "
          f"({100 * c['deterministic_upper_bound']:.2f}%)")
    print(f"  Duplicate rate                           : {100 * c['duplicate_rate']:.2f}%")
    print(f"  Rows in label-conflicting groups         : {c['rows_in_conflicting_groups_pct']:.2f}%")
    print(f"  Test rows leaked, random split           : {ov_random['test_rows_leaked_pct']:.2f}%")
    print(f"  Test rows leaked, group-disjoint split   : {ov_group['test_rows_leaked_pct']:.2f}%")
    print("\n  The upper bound above is a property of the data and is independent of any model.")
    print("  It must be reported separately from the empirical accuracy of a Random Forest.")
    print("\nWritten to leakage_audit_results.json")


if __name__ == "__main__":
    main()
