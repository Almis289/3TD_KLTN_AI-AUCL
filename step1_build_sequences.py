import os, json, ijson
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict

FEATURE_FOLDER = r"D:\Study\khoa_luan\Dataset\CMOSE\secondFeature"
LABEL_FILE = r"D:\Study\khoa_luan\Dataset\CMOSE\label_results_w_audio_final.json"

MAX_LEN = 100
STRIDE = 50
PAD_VALUE = 0.0

os.makedirs("processed_data", exist_ok=True)

# ================= LOAD LABELS =================
label_map = {}
label_counts = Counter()
with open(LABEL_FILE, "r", encoding="utf-8") as f:
    for k, v in ijson.kvitems(f, ""):
        if "label" in v:
            label_map[k] = v["label"]
            label_counts[v["label"]] += 1

print("Labels:", label_counts)

# ================= FEATURE SELECTION =================
sample_file = os.listdir(FEATURE_FOLDER)[0]
sample = pd.read_csv(os.path.join(FEATURE_FOLDER, sample_file))
EXCLUDE = ['frame','face_id','timestamp','confidence','success','Unnamed: 0']
feature_cols = [c for c in sample.columns if c not in EXCLUDE]
feature_cols = feature_cols[:256]
NUM_FEATURES = len(feature_cols)

# ================= COUNT WINDOWS =================
windows = []
for f in os.listdir(FEATURE_FOLDER):
    cid = f.replace(".csv","")
    if cid not in label_map:
        continue
    df = pd.read_csv(os.path.join(FEATURE_FOLDER,f), usecols=feature_cols)
    n = len(df)
    for i in range(0, n-MAX_LEN+1, STRIDE):
        windows.append((f, cid, i))

N = len(windows)
print("Total windows:", N)

# ================= ALLOC MEMMAP =================
X_mem = np.memmap("processed_data/X.dat", dtype="float32",
                  mode="w+", shape=(N, MAX_LEN, NUM_FEATURES))
y = []

# ================= GROUP BY FILE =================
file_windows = defaultdict(list)
for f,cid,start in windows:
    file_windows[f].append((cid,start))

idx = 0
for f, items in tqdm(file_windows.items()):
    path = os.path.join(FEATURE_FOLDER, f)
    df = pd.read_csv(path, usecols=feature_cols)
    df = df.fillna(0).values.astype("float32")

    for cid,start in items:
        seq = df[start:start+MAX_LEN]
        if len(seq) < MAX_LEN:
            seq = np.pad(seq, ((0,MAX_LEN-len(seq)),(0,0)),
                         constant_values=PAD_VALUE)
        X_mem[idx] = seq
        y.append(label_map[cid])
        idx += 1

X_mem.flush()

# ================= SAVE LABELS =================
y = np.array(y)
np.save("processed_data/y.npy", y)

# nhãn nhị phân
y_binary = np.array([1 if v in ["Engage","Highly Engage"] else 0 for v in y])
np.save("processed_data/y_binary.npy", y_binary)

# ================= META =================
meta = {
    "max_len": MAX_LEN,
    "n_features": NUM_FEATURES,
    "pad_value": PAD_VALUE,
    "labels": dict(label_counts)
}
json.dump(meta, open("processed_data/meta.json","w"), indent=2)

np.save("processed_data/features.npy", np.array(feature_cols))

print("✓ Step1 done: X.dat, y.npy, y_binary.npy, meta.json saved")