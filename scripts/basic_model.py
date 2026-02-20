"""
Script de treinamento do modelo LSTM para predição de preços de ações.

Este script:
1. Baixa dados históricos da ação Petrobras (PETR4.SA) usando yfinance
2. Pré-processa os dados com MinMaxScaler
3. Treina um modelo LSTM com Keras/TensorFlow
4. Avalia o modelo com métricas MAE, RMSE e MAPE
5. Salva os pesos do modelo em JSON (para inferência com numpy)
6. Salva os parâmetros do scaler em JSON
7. Salva os dados históricos em CSV

Uso:
    cd fiap4
    pip install tensorflow yfinance scikit-learn matplotlib
    python scripts/train_model.py
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
SYMBOL = 'PETR4.SA'
START_DATE = '2018-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
SEQUENCE_LENGTH = 60
LSTM_UNITS = 50
EPOCHS = 100
BATCH_SIZE = 32
TEST_SPLIT = 0.2
MODEL_VERSION = '1.0.0'
VAL_SPLIT = 0.1

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def download_data():
    """Download stock data from Yahoo Finance."""
    print(f'📥 Baixando dados de {SYMBOL} ({START_DATE} a {END_DATE})...')
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)

    if df.empty:
        print('❌ Nenhum dado encontrado!')
        sys.exit(1)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f'✅ {len(df)} registros baixados')
    return df


def prepare_data_no_leakage(df: pd.DataFrame):
    """
    Prepara dados SEM leakage:
    1) split temporal em train/val/test no eixo do tempo
    2) scaler fit somente no treino
    3) cria janelas (lookback=SEQUENCE_LENGTH)
       - train/val: só dentro do bloco de treino
       - test: usa contexto dos últimos lookback pontos do treino+val, mas targets só do bloco de teste
    """
    print("🔧 Preparando dados (SEM leakage)...")

    close = df["Close"].astype(float).values.reshape(-1, 1)
    n = len(close)

    # split temporal: train_val | test
    test_size = int(n * TEST_SPLIT)
    train_val_size = n - test_size
    test_dates_full = df.index[train_val_size:]

    # dentro do train_val, separa val no final
    val_size = int(train_val_size * VAL_SPLIT)
    train_size = train_val_size - val_size

    close_train = close[:train_size]
    close_val   = close[train_size:train_val_size]
    close_test  = close[train_val_size:]
    dates_test = test_dates_full

    print(f"  Total: {n}")
    print(f"  Train: {len(close_train)} | Val: {len(close_val)} | Test: {len(close_test)}")

    # scaler fit SOMENTE no treino
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_train)

    train_scaled = scaler.transform(close_train)
    val_scaled   = scaler.transform(close_val)
    test_scaled  = scaler.transform(close_test)

    def make_xy(series_scaled: np.ndarray):
        # series_scaled shape: (T,1)
        X, y = [], []
        s = series_scaled[:, 0]
        for i in range(SEQUENCE_LENGTH, len(s)):
            X.append(s[i - SEQUENCE_LENGTH:i])
            y.append(s[i])
        X = np.array(X, dtype=np.float32).reshape(-1, SEQUENCE_LENGTH, 1)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        return X, y

    # train: windows só do treino
    X_train, y_train = make_xy(train_scaled)

    # val: precisa de contexto do final do treino
    # concatena os últimos lookback do treino com todo o val e cria janelas, mas targets caem no bloco val
    val_with_context = np.vstack([train_scaled[-SEQUENCE_LENGTH:], val_scaled])
    X_val, y_val = make_xy(val_with_context)
    # como colocamos contexto, as primeiras janelas "caem" no começo do val — está ok.
    # X_val e y_val já correspondem a pontos dentro do bloco (contexto só alimenta a janela)

    # test: precisa de contexto do final do train_val (treino+val)
    train_val_scaled = np.vstack([train_scaled, val_scaled])
    test_with_context = np.vstack([train_val_scaled[-SEQUENCE_LENGTH:], test_scaled])
    X_test, y_test = make_xy(test_with_context)
    if len(dates_test) != len(y_test):
        # fallback seguro
        dates_test = dates_test[-len(y_test):]

    print(f"  Windows -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, dates_test


def build_model():
    """
    Experimento 1: LSTM simples, sem camadas adicionais.
    """
    print("🏗️  Construindo modelo LSTM...")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, 1)),
        tf.keras.layers.LSTM(
            LSTM_UNITS,
            activation="tanh",
            recurrent_activation="sigmoid",
            dropout=0.1,
            recurrent_dropout=0.1,
            unroll=False,
            use_bias=True,
        ),
        tf.keras.layers.Dense(1),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mean_squared_error")
    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    print("🚀 Treinando modelo...")

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    return history

def plot_training_curves(history, out_dir, prefix="exp1_petr4_d1"):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(history.history.get("loss", []), label="loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title("Curvas de treinamento (Loss vs Val Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_training_curves.png"), dpi=150)
    plt.close()

    print(f"📉 Curvas de treino salvas em: {out_dir}")

def evaluate_model(model, X_test, y_test, scaler, dates_test):
    """Evaluate model, baselines, and directional accuracy (t+1) with fair direction baselines."""
    print('📊 Avaliando modelo (TEST)...')

    # -------------------------
    # 1) Predictions (scaled)
    # -------------------------
    pred_scaled = model.predict(X_test, verbose=0)

    # Denormalize
    pred_price = scaler.inverse_transform(pred_scaled)   # y_hat(t+1)
    true_price = scaler.inverse_transform(y_test)        # y_true(t+1)

    # close_t and close_(t-1) from window (scaled -> denorm)
    close_t_scaled   = X_test[:, -1, 0].reshape(-1, 1)
    close_tm1_scaled = X_test[:, -2, 0].reshape(-1, 1)

    close_t   = scaler.inverse_transform(close_t_scaled)
    close_tm1 = scaler.inverse_transform(close_tm1_scaled)

    # -------------------------
    # 2) Baselines
    # -------------------------
    # Baseline A: naive-price (pred = close_t)
    naive_price = close_t

    # Baseline B: naive-momentum (pred move = last move)
    last_move = close_t - close_tm1
    naive_momentum = close_t + last_move

    # Baseline C: always-up / always-down (for direction reference)
    # For direction, we only need predicted move sign
    # We'll set tiny epsilon move to force sign (+/-)
    eps = 1e-6
    always_up = close_t + eps
    always_down = close_t - eps

    # -------------------------
    # 3) Metrics helpers
    # -------------------------
    def calc_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        r2 = r2_score(y_true, y_pred)
        return float(mae), float(rmse), float(mape), float(r2)

    def directional_accuracy(y_true_price, y_pred_price, close_t_price, ignore_zeros=True):
        """
        Direction defined by sign of (t+1 - t).
        If ignore_zeros=True: excludes cases where either true_move==0 or pred_move==0.
        """
        true_move = (y_true_price - close_t_price).reshape(-1)
        pred_move = (y_pred_price - close_t_price).reshape(-1)

        true_sign = np.sign(true_move)
        pred_sign = np.sign(pred_move)

        if ignore_zeros:
            mask = (true_sign != 0) & (pred_sign != 0)
            if mask.sum() == 0:
                return 0.0
            return float((true_sign[mask] == pred_sign[mask]).mean() * 100.0)

        return float((true_sign == pred_sign).mean() * 100.0)

    def rel_improve(baseline, model_value):
        # improvement % where lower is better
        return (baseline - model_value) / (baseline + 1e-9) * 100.0

    # -------------------------
    # 4) Compute metrics (price)
    # -------------------------
    mae, rmse, mape, r2 = calc_metrics(true_price, pred_price)

    naive_mae, naive_rmse, naive_mape, naive_r2 = calc_metrics(true_price, naive_price)
    mom_mae, mom_rmse, mom_mape, mom_r2 = calc_metrics(true_price, naive_momentum)

    # -------------------------
    # 5) Directional accuracy (fair)
    # -------------------------
    dir_acc = directional_accuracy(true_price, pred_price, close_t, ignore_zeros=True)

    naive_price_dir = directional_accuracy(true_price, naive_price, close_t, ignore_zeros=True)      # likely near 0 if ignore_zeros=False; with ignore_zeros=True it becomes "undefined-like" (drops zero preds)
    mom_dir = directional_accuracy(true_price, naive_momentum, close_t, ignore_zeros=True)

    up_dir = directional_accuracy(true_price, always_up, close_t, ignore_zeros=True)
    down_dir = directional_accuracy(true_price, always_down, close_t, ignore_zeros=True)

    # -------------------------
    # 6) Print summary
    # -------------------------
    print("\n=== LSTM (t+1) ===")
    print(f"  MAE :  {mae:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAPE:  {mape:.4f}%")
    print(f"  R²  :  {r2:.4f}")
    print(f"  Dir Acc (%): {dir_acc:.2f}")

    print("\n=== BASELINES (price) ===")
    print(f"  Naive-price  (pred=close_t)     -> MAE: {naive_mae:.4f} | RMSE: {naive_rmse:.4f} | MAPE: {naive_mape:.4f}% | R²: {naive_r2:.4f}")
    print(f"  Naive-momentum (pred=close_t+Δ) -> MAE: {mom_mae:.4f} | RMSE: {mom_rmse:.4f} | MAPE: {mom_mape:.4f}% | R²: {mom_r2:.4f}")

    print("\n=== Directional Accuracy (ignore zeros) ===")
    print(f"  LSTM          : {dir_acc:.2f}%")
    print(f"  Naive-momentum: {mom_dir:.2f}%")
    print(f"  Always-up     : {up_dir:.2f}%")
    print(f"  Always-down   : {down_dir:.2f}%")
    print(f"  Naive-price   : {naive_price_dir:.2f}%  (tende a ser baixo porque movimento previsto é ~0)")

    # -------------------------
    # 7) Improvements vs baselines
    # -------------------------
    improvements_vs_naive_price = {
        "mae_improve_pct": float(rel_improve(naive_mae, mae)),
        "rmse_improve_pct": float(rel_improve(naive_rmse, rmse)),
        "mape_improve_pct": float(rel_improve(naive_mape, mape)),
        "r2_delta": float(r2 - naive_r2),
    }

    improvements_vs_momentum = {
        "mae_improve_pct": float(rel_improve(mom_mae, mae)),
        "rmse_improve_pct": float(rel_improve(mom_rmse, rmse)),
        "mape_improve_pct": float(rel_improve(mom_mape, mape)),
        "r2_delta": float(r2 - mom_r2),
        "dir_acc_delta_pct_points": float(dir_acc - mom_dir),
    }

    # -------------------------
    # 8) Return (compatível com seu main + dados extras)
    # -------------------------
    metrics = {
        # compatível com seu main antigo
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 6),
        "r2": round(r2, 6),
        "directional_accuracy_pct": round(dir_acc, 6),

        # extras completos
        "lstm": {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "mape_pct": round(mape, 6),
            "r2": round(r2, 6),
            "directional_accuracy_pct": round(dir_acc, 6),
        },
        "baselines": {
            "naive_price": {
                "mae": round(naive_mae, 6),
                "rmse": round(naive_rmse, 6),
                "mape_pct": round(naive_mape, 6),
                "r2": round(naive_r2, 6),
                "directional_accuracy_pct": round(naive_price_dir, 6),
            },
            "naive_momentum": {
                "mae": round(mom_mae, 6),
                "rmse": round(mom_rmse, 6),
                "mape_pct": round(mom_mape, 6),
                "r2": round(mom_r2, 6),
                "directional_accuracy_pct": round(mom_dir, 6),
            },
            "always_up_dir_acc_pct": round(up_dir, 6),
            "always_down_dir_acc_pct": round(down_dir, 6),
        },
        "improvements_vs_naive_price": {k: round(v, 6) for k, v in improvements_vs_naive_price.items()},
        "improvements_vs_naive_momentum": {k: round(v, 6) for k, v in improvements_vs_momentum.items()},
        "notes": {
            "dir_acc_rule": "sign(Close(t+1)-Close(t)) compared to sign(Pred(t+1)-Close(t)); zeros ignored",
            "why_momentum_baseline": "fair baseline for direction; naive_price predicts ~0 move so direction becomes ill-defined",
        }
    }

    plot_experiment1_graphs(
    dates_test=dates_test,
        y_true_price=true_price,
        y_pred_price=pred_price,
        y_naive_price=naive_price,
        out_dir=DATA_DIR,
        prefix="exp1_petr4_d1"
    )

    return metrics



def extract_weights(model):
    # pega LSTM "de verdade" e também wrappers tipo Bidirectional(LSTM) ou RNN(LSTMCell)
    lstm_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            lstm_layers.append(layer)
        elif isinstance(layer, tf.keras.layers.Bidirectional) and isinstance(layer.layer, tf.keras.layers.LSTM):
            lstm_layers.append(layer.layer)
        elif isinstance(layer, tf.keras.layers.RNN) and isinstance(layer.cell, tf.keras.layers.LSTMCell):
            # RNN(LSTMCell) não tem get_weights igual ao LSTM? tem, mas vem do wrapper RNN.
            # Aqui guardamos o wrapper, que também expõe get_weights()
            lstm_layers.append(layer)

    if not lstm_layers:
        raise ValueError("Nenhuma camada LSTM encontrada no modelo.")

    extracted = []
    for i, lstm in enumerate(lstm_layers):
        w = lstm.get_weights()  # [kernel, recurrent_kernel, bias]
        extracted.append({
            "layer_index": i,
            "name": lstm.name,
            "kernel_shape": w[0].shape,
            "recurrent_shape": w[1].shape,
            "bias_shape": w[2].shape,
            "kernel": w[0],
            "recurrent_kernel": w[1],
            "bias": w[2],
        })

    return extracted

def plot_experiment1_graphs(
    dates_test,
    y_true_price,
    y_pred_price,
    y_naive_price,
    out_dir,
    prefix="exp1_petr4_d1"
):
    """
    Salva 3 gráficos:
    1) Real vs Pred (LSTM e Naive)
    2) Erro absoluto ao longo do tempo (LSTM e Naive)
    3) Histograma do erro absoluto (LSTM e Naive)
    """
    os.makedirs(out_dir, exist_ok=True)

    dates = np.asarray(dates_test)
    y_true = np.asarray(y_true_price).reshape(-1)
    y_pred = np.asarray(y_pred_price).reshape(-1)
    y_naive = np.asarray(y_naive_price).reshape(-1)

    # 1) Real vs Pred
    plt.figure(figsize=(12, 4))
    plt.plot(dates, y_true, label="Real Close (t+1)")
    plt.plot(dates, y_pred, label="Pred LSTM (t+1)")
    plt.plot(dates, y_naive, label="Pred Naive-price (t+1)")
    plt.title("Experimento 1 (PETR4 D+1) - Real vs Pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_real_vs_pred.png"), dpi=150)
    plt.close()

    # 2) Erro absoluto ao longo do tempo
    abs_err_lstm = np.abs(y_true - y_pred)
    abs_err_naive = np.abs(y_true - y_naive)

    plt.figure(figsize=(12, 4))
    plt.plot(dates, abs_err_lstm, label="|Erro| LSTM")
    plt.plot(dates, abs_err_naive, label="|Erro| Naive-price")
    plt.title("Experimento 1 (PETR4 D+1) - Erro absoluto ao longo do tempo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_abs_error_over_time.png"), dpi=150)
    plt.close()

    # 3) Histograma do erro absoluto
    plt.figure(figsize=(10, 4))
    plt.hist(abs_err_lstm, bins=50, alpha=0.7, label="|Erro| LSTM")
    plt.hist(abs_err_naive, bins=50, alpha=0.7, label="|Erro| Naive-price")
    plt.title("Experimento 1 (PETR4 D+1) - Distribuição do erro absoluto")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_abs_error_hist.png"), dpi=150)
    plt.close()

    print(f"📈 Gráficos salvos em: {out_dir}")

def save_artifacts(df, weights, scaler, metrics):
    """Save model artifacts to data directory."""
    print('💾 Salvando artefatos...')

    os.makedirs(DATA_DIR, exist_ok=True)

    # Save stock data CSV
    csv_path = os.path.join(DATA_DIR, 'stock_data.csv')
    df_save = df.copy()
    df_save.index.name = 'Date'
    df_save.reset_index(inplace=True)
    df_save['Date'] = df_save['Date'].dt.strftime('%Y-%m-%d')
    df_save.to_csv(csv_path, index=False)
    print(f'  ✅ Dados salvos: {csv_path} ({len(df_save)} registros)')

    # Save model weights
    weights_path = os.path.join(DATA_DIR, 'model_weights.json')
    weights_dict = weights_to_jsonable(weights)
    model_data = {
        'version': MODEL_VERSION,
        'symbol': SYMBOL,
        'sequence_length': SEQUENCE_LENGTH,
        'lstm_units': LSTM_UNITS,
        'training_start': START_DATE,
        'training_end': END_DATE,
        'trained_at': datetime.now().isoformat(),
        'metrics': metrics,
        'weights': weights_dict
    }

    with open(weights_path, 'w') as f:
        json.dump(model_data, f)
    print(f'  ✅ Pesos do modelo salvos: {weights_path}')

    # Save scaler parameters
    scaler_path = os.path.join(DATA_DIR, 'scaler_params.json')
    scaler_data = {
        'min': float(scaler.data_min_[0]),
        'max': float(scaler.data_max_[0]),
        'feature_range': [0, 1]
    }

    with open(scaler_path, 'w') as f:
        json.dump(scaler_data, f, indent=2)
    print(f'  ✅ Parâmetros do scaler salvos: {scaler_path}')

def weights_to_jsonable(weights):
    # weights: list[dict] (cada item: layer info + arrays numpy)
    out = {}
    for w in weights:
        name = w["name"]
        out[name] = {
            "kernel_shape": list(w["kernel_shape"]),
            "recurrent_shape": list(w["recurrent_shape"]),
            "bias_shape": list(w["bias_shape"]),
            "kernel": w["kernel"].tolist(),
            "recurrent_kernel": w["recurrent_kernel"].tolist(),
            "bias": w["bias"].tolist(),
        }
    return out


def main():
    """Main training pipeline."""
    print('=' * 60)
    print(f'  LSTM Stock Price Predictor - Training Pipeline')
    print(f'  Symbol: {SYMBOL} | Period: {START_DATE} to {END_DATE}')
    print('=' * 60)
    print()

    # Step 1: Download data
    df = download_data()

   # Step 2: Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, dates_test = prepare_data_no_leakage(df)

    # Step 3: Build model
    model = build_model()

    # Step 4: Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    plot_training_curves(history, out_dir=DATA_DIR, prefix="exp1_petr4_d1")

    # Step 5: Evaluate model
    metrics = evaluate_model(model, X_test, y_test, scaler, dates_test)

    # Step 6: Extract weights
    weights = extract_weights(model)

    # Step 7: Save everything
    save_artifacts(df, weights, scaler, metrics)

    print()
    print('=' * 60)
    print('  ✅ Treinamento concluído com sucesso!')
    print(f'  📊 MAE: {metrics["mae"]:.4f} | RMSE: {metrics["rmse"]:.4f} | MAPE: {metrics["mape"]:.4f}%')
    print('  📁 Artefatos salvos em: data/')
    print('  🚀 Execute "make dev" para iniciar a API')
    print('=' * 60)


if __name__ == '__main__':
    main()