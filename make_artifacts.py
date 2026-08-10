"""Generate the reproducibility artifacts promised in the paper's Appendix.

Fast: no model training. Produces checksums, feature lists, excluded columns,
label mapping, split indices and fold indices, plus a pinned requirements file.

NOTEBOOK-FRIENDLY: edit CONFIG and run. Writes everything under ./artifacts/.
"""

# ============================== CONFIG ==============================
DATASETS = {
    "CIC-Bell-DNS-EXF-2021": {"csv": "merged_heavy.csv", "label": "label"},
    "BCCC-CIC-Bell-DNS-2024": {"csv": "merged_mal_clean.csv", "label": "label"},
}
RAW_SOURCE_DIRS = []      # optional: folders holding the original downloaded CSVs
TEST_SIZE = 0.2
SEED = 42
OUT = "artifacts"
# ====================================================================

import hashlib, json, os, sys, subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

os.makedirs(OUT, exist_ok=True)
manifest = {"seed": SEED, "test_size": TEST_SIZE, "datasets": {}}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---- checksums of any original source files, if their folders were given -------------
if RAW_SOURCE_DIRS:
    src = []
    for d in RAW_SOURCE_DIRS:
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if f.lower().endswith((".csv", ".zip", ".parquet")):
                    p = os.path.join(root, f)
                    src.append({"file": os.path.relpath(p, d),
                                "bytes": os.path.getsize(p),
                                "sha256": sha256(p)})
                    print(f"  hashed {f}")
    manifest["source_files"] = src

# ---- per-dataset artifacts -----------------------------------------------------------
for name, cfg in DATASETS.items():
    path, label_col = cfg["csv"], cfg["label"]
    if not os.path.exists(path):
        print(f"[skip] {name}: {path} not found")
        continue
    print(f"\n=== {name} ({path}) ===")
    entry = {"file": os.path.basename(path),
             "bytes": os.path.getsize(path),
             "sha256": sha256(path)}

    df = pd.read_csv(path, low_memory=False)
    entry["rows"] = int(len(df))
    entry["columns_total"] = int(df.shape[1])

    y = df[label_col].astype(int).to_numpy()
    entry["label_column"] = label_col
    entry["label_mapping"] = {"0": "benign", "1": "attack"}
    entry["class_counts"] = {"benign": int((y == 0).sum()), "attack": int((y == 1).sum())}

    all_cols = [c for c in df.columns if c != label_col]
    num = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    retained = list(num.columns)
    excluded = [c for c in all_cols if c not in retained]
    entry["retained_features"] = retained
    entry["n_retained_features"] = len(retained)
    entry["excluded_columns"] = excluded
    entry["n_excluded_columns"] = len(excluded)

    tag = name.replace("-", "_")
    with open(f"{OUT}/features_{tag}.json", "w") as fh:
        json.dump({"retained": retained, "excluded": excluded}, fh, indent=2)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    np.save(f"{OUT}/train_indices_{tag}.npy", tr)
    np.save(f"{OUT}/test_indices_{tag}.npy", te)
    entry["split"] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                      "stratified": True, "seed": SEED}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = {}
    for k, (a, b) in enumerate(skf.split(idx, y)):
        folds[f"fold{k}_train"] = a
        folds[f"fold{k}_val"] = b
    np.savez_compressed(f"{OUT}/cv_folds_{tag}.npz", **folds)
    entry["cv"] = {"n_splits": 5, "stratified": True, "shuffle": True, "seed": SEED}

    print(f"  rows {entry['rows']:,} | retained {len(retained)} | excluded {len(excluded)}")
    print(f"  sha256 {entry['sha256'][:16]}...")
    print(f"  wrote features_{tag}.json, train/test indices, cv_folds_{tag}.npz")
    manifest["datasets"][name] = entry
    del df, num
# ---- environment ---------------------------------------------------------------------
try:
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, timeout=180).stdout
    with open("requirements-lock.txt", "w") as fh:
        fh.write(freeze)
    keep = ("numpy", "pandas", "scikit-learn", "tensorflow", "xgboost",
            "lightgbm", "optuna", "psutil", "joblib")
    manifest["environment"] = {
        "python": sys.version.split()[0],
        "packages": {l.split("==")[0]: l.split("==")[1]
                     for l in freeze.splitlines() if "==" in l
                     and l.split("==")[0].lower() in keep},
    }
    print("\nWrote requirements-lock.txt")
except Exception as ex:
    print(f"[warn] pip freeze failed: {ex}")

with open(f"{OUT}/manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)

print(f"\nAll artifacts written to ./{OUT}/ - upload this folder and requirements-lock.txt "
      f"to the repository.")
print("Remaining artifacts that need a run: per-fold predictions (statistical_tests.py), "
      "optuna_trials.csv (optuna_search.py), model_metadata.json (complexity_explainability.py).")
