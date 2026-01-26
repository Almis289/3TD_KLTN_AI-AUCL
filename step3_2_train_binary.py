import os, json, pickle, numpy as np, tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

meta = json.load(open("processed_data/meta.json"))
scaler = pickle.load(open("processed_data/scaler.pkl","rb"))
y = np.load("processed_data/y_binary.npy")
tr = np.load("processed_data/train_idx_bin.npy")
va = np.load("processed_data/val_idx_bin.npy")

N = len(y)
T = meta["max_len"]
F = meta["n_features"]
X = np.memmap("processed_data/X.dat", dtype="float32", mode="r",
              shape=(N, T, F))

# ================= DATA GENERATOR =================
class Gen(tf.keras.utils.Sequence):
    def __init__(self, idx, batch= 32, shuffle=True, augment=False):
        self.idx = idx
        self.batch = batch
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.idx) / self.batch))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)
    
    def augment_seq(self, seq, noise_level=0.01):
        """Add Gaussian noise for augmentation"""
        noise = np.random.normal(0, noise_level, seq.shape)
        return seq + noise
    
    def __getitem__(self, i):
        b = self.idx[i*self.batch:(i+1)*self.batch]
        Xb = (X[b] - scaler.mean_) / scaler.scale_
        
        # Apply augmentation if enabled
        if self.augment:
            for j in range(len(Xb)):
                if np.random.random() < 0.5:  # 50% chance
                    Xb[j] = self.augment_seq(Xb[j])
        
        return Xb, y[b]

tr_gen = Gen(tr, augment=True)
va_gen = Gen(va, shuffle=False)

# ================= COMPUTE CLASS WEIGHTS =================
class_weight_dict = {
    0: 2.0,   # phạt nặng khi bỏ sót Disengage
    1: 1.0    # Engage nhẹ hơn
}

print("Final class weights:", class_weight_dict)

# ================= Focal loss =================
def focal_loss(alpha=0.75, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=2)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_oh * tf.math.log(y_pred)

        weight = alpha * tf.pow(1 - y_pred, gamma)
        fl = weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    return loss

# ================= BUILD CNN + BiLSTM MODEL =================
inp = Input(shape=(T, F))

# CNN layers for local feature extraction
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)

# BiLSTM layers for temporal dependencies
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(32))(x)
x = Dropout(0.3)(x)

# Dense output
out = Dense(2, activation="softmax")(x)

model = Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=focal_loss(alpha=0.8, gamma=2.0),
    metrics=["accuracy"]
)

model.summary()

# ================= CALLBACKS =================
os.makedirs("models", exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'models/model_bin_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# ================= TRAIN =================
history = model.fit(
    tr_gen,
    validation_data=va_gen,
    epochs=100,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

model.save("models/model_bin.keras")
print("✓ Saved model_bin.keras")

# Save training history
import pickle
with open("models/history_bin.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("✓ Saved training history")

