import numpy as np
from sklearn.model_selection import train_test_split

y = np.load("processed_data/y.npy", allow_pickle=True)
y_bin = np.load("processed_data/y_binary.npy")
idx = np.arange(len(y))

# tầng 1 - Binary classification
tr_bin, va_bin = train_test_split(idx, test_size=0.2,
                                  stratify=y_bin, random_state=42)

# tầng Engage
mask_eng = np.isin(y, ["Engage","Highly Engage"])
idx_eng = idx[mask_eng]
y_eng = y[mask_eng]
y_eng_bin = np.array([0 if v=="Engage" else 1 for v in y_eng])
tr_eng, va_eng = train_test_split(idx_eng, test_size=0.2,
                                  stratify=y_eng_bin, random_state=42)

# tầng Disengage
mask_dis = np.isin(y, ["Disengage","Highly Disengage"])
idx_dis = idx[mask_dis]
y_dis = y[mask_dis]
y_dis_bin = np.array([0 if v=="Disengage" else 1 for v in y_dis])
tr_dis, va_dis = train_test_split(idx_dis, test_size=0.2,
                                  stratify=y_dis_bin, random_state=42)

np.save("processed_data/train_idx_bin.npy", tr_bin)
np.save("processed_data/val_idx_bin.npy", va_bin)
np.save("processed_data/train_idx_eng.npy", tr_eng)
np.save("processed_data/val_idx_eng.npy", va_eng)
np.save("processed_data/train_idx_dis.npy", tr_dis)
np.save("processed_data/val_idx_dis.npy", va_dis)

print("✓ Saved hierarchical indices")