import json, pickle, numpy as np
from sklearn.preprocessing import StandardScaler

meta = json.load(open("processed_data/meta.json"))
y = np.load("processed_data/y.npy", allow_pickle=True)

N = len(y)
T = meta["max_len"]
F = meta["n_features"]

X = np.memmap("processed_data/X.dat", dtype="float32", mode="r",
              shape=(N, T, F))

scaler = StandardScaler()
subset = int(N * 0.1)
chunk = 4

for i in range(0, subset, chunk):
    part = X[i:i+chunk].mean(axis=1)
    scaler.partial_fit(part)

pickle.dump(scaler, open("processed_data/scaler.pkl","wb"))
print("✓ Saved scaler.pkl")