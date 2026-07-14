"""
Data preparation for both benchmarks (download, merge, label, audit).
Reproduces exactly the corpora used in the paper.

Usage:  python prepare_data.py
Output: merged_heavy.csv (EXF-2021), merged_mal_clean.csv (BCCC-2024)
        + printed duplicate/label-conflict audit (paper Table 3 / Figure 3)
"""
import os, gc
import numpy as np
import pandas as pd
import kagglehub

SEED = 42

# ------------------------------------------------ 1) CIC-Bell-DNS-EXF-2021
print("Downloading CIC-Bell-DNS-EXF-2021 ...")
p1 = kagglehub.dataset_download("humera11/cicbelldnsexf2021")
frames = []
for root, dirs, files in os.walk(p1):
    for f in files:
        if 'stateful' in f and f.endswith('.csv') and 'heavy' in f:
            label = 1 if 'Attacks' in root.split(os.sep) else 0
            df = pd.read_csv(os.path.join(root, f), low_memory=False)
            df['label'] = label
            frames.append(df)
exf = pd.concat(frames, ignore_index=True)
exf.to_csv('merged_heavy.csv', index=False)
print(f"EXF heavy subset: {len(exf):,} rows, "
      f"distribution {exf['label'].value_counts().to_dict()}")

# audit (paper Section 4.2)
feat = exf.drop(columns=['label']).select_dtypes(include=[np.number])
dups = exf.duplicated(subset=feat.columns.tolist()).sum()
key = pd.util.hash_pandas_object(feat, index=False)
g = exf.groupby(key)['label'].nunique()
conf = exf[key.isin(g[g > 1].index)].shape[0]
att = exf[exf['label'] == 1]
uniq = att.drop(columns=['label']).drop_duplicates().shape[0]
print(f"AUDIT — duplicates: {dups:,} ({dups/len(exf):.1%}) | "
      f"label conflicts: {conf:,} ({conf/len(exf):.1%}) | "
      f"unique attack vectors: {uniq:,} of {len(att):,}")
del exf, frames; gc.collect()

# ------------------------------------------------ 2) BCCC-CIC-Bell-DNS-2024
print("\nDownloading BCCC-CIC-Bell-DNS-2024 ...")
p2 = kagglehub.dataset_download(
    "bcccdatasets/malicious-dns-and-attacks-bccc-cic-bell-dns-2024")
leak = {'Unnamed: 0','flow_id','timestamp','src_ip','src_port',
        'dst_ip','dst_port','protocol'}
all_files = []
for root, dirs, files in os.walk(p2):
    for f in files:
        if f.endswith('.csv') and ('benign-pcap' in f or 'spam' in f
                                    or 'malware' in f or 'phishing' in f):
            all_files.append(os.path.join(root, f))
sample = pd.read_csv(all_files[0], nrows=500, low_memory=False)
keep = [c for c in sample.columns if c not in leak
        and pd.api.types.is_numeric_dtype(sample[c])]
frames = []
for fp in all_files:
    name = os.path.basename(fp)
    label = 0 if 'benign' in name else 1
    parts = []
    for chunk in pd.read_csv(fp, usecols=keep, chunksize=150_000,
                             low_memory=False):
        chunk = chunk.astype(np.float32)
        if label == 0:                       # 10% benign sample (paper setup)
            chunk = chunk.sample(frac=0.10, random_state=SEED)
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df['label'] = np.int8(label)
    frames.append(df)
    del parts; gc.collect()
mal = pd.concat(frames, ignore_index=True)
mal.to_csv('merged_mal_clean.csv', index=False)
print(f"BCCC-2024 (clean): {len(mal):,} rows, {len(mal.columns)} cols, "
      f"distribution {mal['label'].value_counts().to_dict()}")
