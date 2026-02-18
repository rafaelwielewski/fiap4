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
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
SYMBOL = 'PETR4.SA'
START_DATE = '2018-01-01'
END_DATE = '2024-07-20'
SEQUENCE_LENGTH = 60
LSTM_UNITS = 50
EPOCHS = 100
BATCH_SIZE = 32
TEST_SPLIT = 0.2
MODEL_VERSION = '1.0.0'

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


def prepare_data(df):
    """Prepare data for LSTM training."""
    print('🔧 Preparando dados...')

    close_prices = df['Close'].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create sequences
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i - SEQUENCE_LENGTH:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape for LSTM: (samples, time_steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split into train/test
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f'  Train: {len(X_train)} samples, Test: {len(X_test)} samples')

    return X_train, X_test, y_train, y_test, scaler


def build_model():
    """Build LSTM model."""
    print('🏗️  Construindo modelo LSTM...')

    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
        Dropout(0.2),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """Train the LSTM model."""
    print('🚀 Treinando modelo...')

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
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    return history


def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model and calculate metrics."""
    print('📊 Avaliando modelo...')

    predictions = model.predict(X_test)

    # Denormalize
    predictions_denorm = scaler.inverse_transform(predictions)
    y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mae = mean_absolute_error(y_test_denorm, predictions_denorm)
    rmse = np.sqrt(mean_squared_error(y_test_denorm, predictions_denorm))
    mape = np.mean(np.abs((y_test_denorm - predictions_denorm) / y_test_denorm)) * 100

    metrics = {
        'mae': round(float(mae), 4),
        'rmse': round(float(rmse), 4),
        'mape': round(float(mape), 4)
    }

    print(f'  MAE:  {metrics["mae"]:.4f}')
    print(f'  RMSE: {metrics["rmse"]:.4f}')
    print(f'  MAPE: {metrics["mape"]:.4f}%')

    return metrics


def extract_weights(model):
    """Extract model weights for numpy inference."""
    print('💾 Extraindo pesos do modelo...')

    weights = {}

    # Get LSTM layers (we have 2 LSTM layers)
    lstm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]

    # For simplicity, we'll use only the last LSTM and last Dense for inference
    # The full model has: LSTM -> Dropout -> LSTM -> Dropout -> Dense(25) -> Dense(1)
    # We'll extract all needed weights

    # First LSTM layer
    lstm1_weights = lstm_layers[0].get_weights()
    weights['lstm1'] = {
        'kernel': lstm1_weights[0].tolist(),
        'recurrent': lstm1_weights[1].tolist(),
        'bias': lstm1_weights[2].tolist()
    }

    # Second LSTM layer
    lstm2_weights = lstm_layers[1].get_weights()
    weights['lstm'] = {
        'kernel': lstm2_weights[0].tolist(),
        'recurrent': lstm2_weights[1].tolist(),
        'bias': lstm2_weights[2].tolist()
    }

    # Dense layers
    dense1_weights = dense_layers[0].get_weights()
    weights['dense1'] = {
        'kernel': dense1_weights[0].tolist(),
        'bias': dense1_weights[1].tolist()
    }

    dense2_weights = dense_layers[1].get_weights()
    weights['dense'] = {
        'kernel': dense2_weights[0].tolist(),
        'bias': dense2_weights[1].tolist()
    }

    return weights


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
    model_data = {
        'version': MODEL_VERSION,
        'symbol': SYMBOL,
        'sequence_length': SEQUENCE_LENGTH,
        'lstm_units': LSTM_UNITS,
        'training_start': START_DATE,
        'training_end': END_DATE,
        'trained_at': datetime.now().isoformat(),
        'metrics': metrics,
        **weights
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
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Step 3: Build model
    model = build_model()

    # Step 4: Train model
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Step 5: Evaluate model
    metrics = evaluate_model(model, X_test, y_test, scaler)

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
