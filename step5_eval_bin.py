import json, pickle, numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def focal_loss(alpha=0.8, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=2)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_oh * tf.math.log(y_pred)

        weight = alpha * tf.pow(1 - y_pred, gamma)
        fl = weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    return loss



# ===== Load =====
meta = json.load(open("processed_data/meta.json"))
scaler = pickle.load(open("processed_data/scaler.pkl","rb"))
model = tf.keras.models.load_model(
    "models/model_bin_best.keras",
    custom_objects={"loss": focal_loss(alpha=0.8, gamma=2.0)}
)

y = np.load("processed_data/y_binary.npy")
val_idx = np.load("processed_data/val_idx_bin.npy")

T = meta["max_len"]
F = meta["n_features"]

X = np.memmap("processed_data/X.dat", dtype="float32", mode="r",
              shape=(len(y), T, F))

def norm(x):
    return (x - scaler.mean_) / scaler.scale_

# ===== Predict =====
BATCH = 256
y_true = []
y_pred = []



for i in range(0, len(val_idx), BATCH):
    idx = val_idx[i:i+BATCH]
    xb = norm(X[idx])
    p = model.predict(xb, verbose=0)
    labs = (p[:,1] > 0.5).astype(int)

    y_true.extend(y[idx].tolist())
    y_pred.extend(labs.tolist())

# ===== Report =====
print("\n=== CLASSIFICATION REPORT (Binary) ===")
print(classification_report(
    y_true, y_pred,
    target_names=["Disengage", "Engage"],
    digits=4
))

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# ===== Save results =====
import pickle
res = {
    "y_true": y_true,
    "y_pred": y_pred,
    "confusion_matrix": cm.tolist()
}
with open("eval_bin_results.pkl", "wb") as f:
    pickle.dump(res, f)
print("\n✓ Saved eval_bin_results.pkl")

