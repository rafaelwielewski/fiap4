import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU (remove if you want to use GPU and have it configured)

import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

SYMBOL = "AAPL"
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"

LOOKBACK = 60
HORIZON = 5

TEST_RATIO = 0.15
VAL_RATIO = 0.15

EPOCHS = 120
BATCH_SIZE = 32

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# =========================
# HELPERS
# =========================
def to_1d_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta.clip(upper=0))
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = to_1d_series(df["Close"]).astype(float)
    high  = to_1d_series(df["High"]).astype(float)
    low   = to_1d_series(df["Low"]).astype(float)
    openp = to_1d_series(df["Open"]).astype(float)
    vol   = to_1d_series(df["Volume"]).astype(float)

    feats = pd.DataFrame({
        "close": close,
        "high": high,
        "low": low,
        "open": openp,
        "volume": vol,
    })

    feats["ret_1"] = feats["close"].pct_change()
    feats["log_ret_1"] = np.log(feats["close"]).diff()

    feats["sma_7"]  = feats["close"].rolling(7).mean()
    feats["sma_21"] = feats["close"].rolling(21).mean()
    feats["ema_12"] = feats["close"].ewm(span=12, adjust=False).mean()
    feats["ema_26"] = feats["close"].ewm(span=26, adjust=False).mean()

    feats["macd"] = feats["ema_12"] - feats["ema_26"]
    feats["macd_signal"] = feats["macd"].ewm(span=9, adjust=False).mean()

    feats["rsi_14"] = rsi(feats["close"], 14)

    feats["vol_7"]  = feats["ret_1"].rolling(7).std()
    feats["vol_21"] = feats["ret_1"].rolling(21).std()

    feats = feats.dropna().copy()

    feats["close_t"] = feats["close"]
    feats["close_t_h"] = feats["close"].shift(-HORIZON)

    # ✅ TARGET ROBUSTO: delta em dólares
    feats["y_delta_h"] = feats["close_t_h"] - feats["close_t"]

    feats = feats.dropna().copy()
    return feats

def make_windows(X_all, y_all, close_t, close_t_h, dates, lookback):
    Xw, yw, c_t, c_th, dt = [], [], [], [], []
    n = len(X_all)
    for t in range(lookback - 1, n):
        Xw.append(X_all[t - lookback + 1:t + 1, :])
        yw.append(y_all[t])
        c_t.append(close_t[t])
        c_th.append(close_t_h[t])
        dt.append(dates[t])
    return (
        np.array(Xw, dtype=np.float32),
        np.array(yw, dtype=np.float32).reshape(-1, 1),
        np.array(c_t, dtype=np.float32).reshape(-1, 1),
        np.array(c_th, dtype=np.float32).reshape(-1, 1),
        pd.to_datetime(np.array(dt)),
    )

def metrics_price(y_true_price, y_pred_price):
    mae = mean_absolute_error(y_true_price, y_pred_price)
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mape = np.mean(np.abs((y_true_price - y_pred_price) / (y_true_price + 1e-9))) * 100
    return mae, rmse, mape

def directional_accuracy_price(true_price, pred_price, close_t):
    true_sign = np.sign((true_price - close_t).reshape(-1))
    pred_sign = np.sign((pred_price - close_t).reshape(-1))
    return float((true_sign == pred_sign).mean() * 100)

# =========================
# 1) DOWNLOAD
# =========================
print(f"Downloading {SYMBOL}...")
df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)

df = df.dropna()
col_check = ["Open", "High", "Low", "Close", "Volume"]
if isinstance(df.columns, pd.MultiIndex):
    pass

try:
    mask = (df["Close"] > 0) & (df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0)
    if isinstance(mask, pd.DataFrame):
        mask = mask.iloc[:, 0]
    df = df[mask]
except Exception as e:
    print(f"Warning during cleaning: {e}")

print(df.head())
print(df.shape)
print(df.columns)

# =========================
# 2) FEATURES + TARGET
# =========================
feats = build_features(df)

feature_cols = [
    "close", "high", "low", "open", "volume",
    "ret_1", "log_ret_1",
    "sma_7", "sma_21", "ema_12", "ema_26",
    "macd", "macd_signal",
    "rsi_14",
    "vol_7", "vol_21"
]
target_col = "y_delta_h"

X_raw = feats[feature_cols].values
y_raw = feats[[target_col]].values
close_t = feats[["close_t"]].values
close_t_h = feats[["close_t_h"]].values
dates = feats.index.values

print("Feature shape:", X_raw.shape, "Target shape:", y_raw.shape)

# =========================
# 3) SPLIT TEMPORAL
# =========================
n_total = len(feats)
n_test = int(n_total * TEST_RATIO)
n_val  = int(n_total * VAL_RATIO)
n_train = n_total - n_val - n_test

train_end_idx = n_train - 1
val_end_idx = n_train + n_val - 1
train_end_date = feats.index[train_end_idx]
val_end_date = feats.index[val_end_idx]

print("Split dates:")
print(f"  train end: {train_end_date.date()}  (n={n_train})")
print(f"  val end  : {val_end_date.date()}    (n={n_val})")
print(f"  test     : {feats.index[val_end_idx+1].date()} -> {feats.index[-1].date()} (n={n_test})")

# =========================
# 4) SCALING sem vazamento
# =========================
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_scaled = np.vstack([
    scaler_X.fit_transform(X_raw[:n_train]),
    scaler_X.transform(X_raw[n_train:])
])

y_scaled = np.vstack([
    scaler_y.fit_transform(y_raw[:n_train]),
    scaler_y.transform(y_raw[n_train:])
])

# =========================
# 5) WINDOWS
# =========================
Xw, yw, c_t_w, c_th_w, dt_w = make_windows(X_scaled, y_scaled, close_t, close_t_h, dates, LOOKBACK)

train_mask = dt_w <= train_end_date
val_mask   = (dt_w > train_end_date) & (dt_w <= val_end_date)
test_mask  = dt_w > val_end_date

X_train, y_train = Xw[train_mask], yw[train_mask]
X_val, y_val     = Xw[val_mask], yw[val_mask]
X_test, y_test   = Xw[test_mask], yw[test_mask]

c_t_test  = c_t_w[test_mask]
c_th_test = c_th_w[test_mask]
dt_test   = dt_w[test_mask]

print("Window shapes:")
print("  X_train", X_train.shape, "y_train", y_train.shape)
print("  X_val  ", X_val.shape,   "y_val  ", y_val.shape)
print("  X_test ", X_test.shape,  "y_test ", y_test.shape)

n_features = X_train.shape[-1]

# =========================
# 6) MODEL
# =========================
model = keras.Sequential([
    layers.Input(shape=(LOOKBACK, n_features)),
    layers.LSTM(64, return_sequences=True, recurrent_dropout=0.05),
    layers.Dropout(0.2),
    layers.LSTM(32, recurrent_dropout=0.05),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)
])

opt = keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0)
model.compile(optimizer=opt, loss=keras.losses.Huber(delta=1.0))
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint(str(ARTIFACTS_DIR / "best_model.keras"), monitor="val_loss", save_best_only=True),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# =========================
# 7) PREDICT + inverse scale
# =========================
y_pred_scaled = model.predict(X_test, verbose=0)
pred_delta = scaler_y.inverse_transform(y_pred_scaled)
true_delta = scaler_y.inverse_transform(y_test)

# reconstrói preço futuro
pred_price = c_t_test + pred_delta
true_price = c_th_test

# =========================
# 8) METRICS
# =========================
mae, rmse, mape = metrics_price(true_price, pred_price)
dir_acc = directional_accuracy_price(true_price, pred_price, c_t_test)

print(f"\n=== MODEL (LSTM) METRICS (D+{HORIZON}) ===")
print(f"MAE (price) : {mae:.4f}")
print(f"RMSE (price): {rmse:.4f}")
print(f"MAPE (%)    : {mape:.2f}%")
print(f"Dir Acc (%) : {dir_acc:.2f}%")

# =========================
# 9) BASELINES
# =========================
naive_pred = c_t_test
naive_mae, naive_rmse, naive_mape = metrics_price(true_price, naive_pred)

close_series = feats["close"].copy()
sma_lb = close_series.rolling(LOOKBACK).mean()
sma_pred = sma_lb.loc[dt_test].values.reshape(-1, 1)
sma_pred = np.where(np.isnan(sma_pred), naive_pred, sma_pred)
sma_mae, sma_rmse, sma_mape = metrics_price(true_price, sma_pred)

print(f"\n=== BASELINES (D+{HORIZON}) ===")
print(f"Naive  -> MAE: {naive_mae:.4f} | RMSE: {naive_rmse:.4f} | MAPE: {naive_mape:.2f}%")
print(f"SMA{LOOKBACK:02d} -> MAE: {sma_mae:.4f} | RMSE: {sma_rmse:.4f} | MAPE: {sma_mape:.2f}%")

# =========================
# 10) PLOTS
# =========================
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training curves (Huber)")
plt.legend()
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "training_curves.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(true_price, label=f"Real Close (t+{HORIZON})")
plt.plot(pred_price, label=f"Pred Close (t+{HORIZON})")
plt.title(f"{SYMBOL} - Real vs Pred (D+{HORIZON})")
plt.legend()
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "real_vs_pred_price.png", dpi=150)
plt.close()

labels = ["LSTM", "Naive", f"SMA{LOOKBACK}"]
mapes = [mape, naive_mape, sma_mape]
plt.figure(figsize=(8, 4))
plt.bar(labels, mapes)
plt.title(f"MAPE (%) comparison - D+{HORIZON}")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "mape_comparison.png", dpi=150)
plt.close()

# =========================
# 11) SAVE ARTIFACTS
# =========================
model.save(ARTIFACTS_DIR / "final_model.keras")
joblib.dump(scaler_X, ARTIFACTS_DIR / "scaler_X.joblib")
joblib.dump(scaler_y, ARTIFACTS_DIR / "scaler_y.joblib")

csv_path = DATA_DIR / "stock_data.csv"
df_save = df.copy()
if isinstance(df_save.columns, pd.MultiIndex):
    df_save.columns = df_save.columns.get_level_values(0)
df_save.index.name = "Date"
df_save.reset_index(inplace=True)
df_save["Date"] = df_save["Date"].dt.strftime("%Y-%m-%d")
df_save.to_csv(csv_path, index=False)
print(f"  ✅ Stock data CSV: {csv_path} ({len(df_save)} registros)")

metadata = {
    "symbol": SYMBOL,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "lookback": LOOKBACK,
    "horizon_days": HORIZON,
    "features": feature_cols,
    "target": f"delta_close = close_(t+{HORIZON}) - close_t",
    "trained_at": datetime.now().isoformat(),
    "splits": {"train_rows": int(n_train), "val_rows": int(n_val), "test_rows": int(n_test)},
    "notes": {
        "why_delta": "delta makes the target more stationary than raw price; evaluation still in price",
        "cudnn_note": "recurrent_dropout used to avoid cuDNN path for compatibility"
    }
}
with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

metrics_out = {
    "model": {
        "mae_price": float(mae),
        "rmse_price": float(rmse),
        "mape_price_pct": float(mape),
        "directional_accuracy_pct": float(dir_acc),
    },
    "baselines": {
        "naive": {"mae_price": float(naive_mae), "rmse_price": float(naive_rmse), "mape_price_pct": float(naive_mape)},
        f"sma_{LOOKBACK}": {"mae_price": float(sma_mae), "rmse_price": float(sma_rmse), "mape_price_pct": float(sma_mape)},
    }
}
with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, indent=2, ensure_ascii=False)

print("\nArtifacts saved in:", ARTIFACTS_DIR.resolve())
for p in sorted(ARTIFACTS_DIR.glob("*")):
    print(" -", p)