import json, pickle, numpy as np, tensorflow as tf

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


# ===== Load meta & scaler =====
meta = json.load(open("processed_data/meta.json"))
scaler = pickle.load(open("processed_data/scaler.pkl","rb"))

T = meta["max_len"]
F = meta["n_features"]

# ===== Load model =====
model = tf.keras.models.load_model(
    "models/model_bin_best.keras",
    custom_objects={"loss": focal_loss(alpha=0.8, gamma=2.0)}
)

def norm(x):
    return (x - scaler.mean_) / scaler.scale_

def predict_one(seq):
    x = norm(seq)[None, ...]
    p = model.predict(x, verbose=0)[0]
    label = "Engage" if np.argmax(p) == 1 else "Disengage"
    return label, p.tolist()

def predict_batch(X, batch_size=128):
    Xn = norm(X)
    preds = []
    for i in range(0, len(Xn), batch_size):
        p = model.predict(Xn[i:i+batch_size], verbose=0)
        labs = np.argmax(p, axis=1)
        for v in labs:
            preds.append("Engage" if v == 1 else "Disengage")
    return preds

if __name__ == "__main__":
    print("Loading data...")
    y = np.load("processed_data/y_binary.npy")
    N = len(y)
    X = np.memmap("processed_data/X.dat", dtype="float32", mode="r",
                  shape=(N, T, F))

    print("\n=== Single test ===")
    lab, prob = predict_one(X[0])
    print("Pred:", lab, " Prob:", prob, " True:", y[0])

    print("\n=== Batch test ===")
    preds = predict_batch(X[:10])
    print("Preds:", preds)
    print("True :", y[:10].tolist())
