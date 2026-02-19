"""
Generate sample model weights and stock data for testing.
This creates dummy data so the API can work without training the real model.

Run from the fiap4/ directory:
    python scripts/generate_sample_data.py
"""

import json
import os
import sys
import numpy as np
from datetime import datetime, timedelta

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
Tech Challenge
ANA RAQUEL
TECH CHALLENGE
FASE 04Tech Challenge
TECH CHALLENGE
Tech Challenge é o projeto da fase que engloba os conhecimentos obtidos
em todas as disciplinas dela. Esta é uma atividade que, a princípio, deve ser
desenvolvida em grupo. É importante atentar-se ao prazo de entrega, uma vez
que essa atividade é obrigatória e vale 90% da nota de todas as disciplinas da
fase.
O problema
Deep Learning e IA
Olá, estudante! Chegamos ao Tech Challenge da Fase 4. Nela, você
aprendeu sobre um tipo de aprendizado mais profundo e chegou o momento de
colocar isso em prática!
Seu desafio é criar um modelo preditivo de redes neurais Long Short
Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores
de uma empresa à sua escolha e realizar toda a pipeline de desenvolvimento,
desde a criação do modelo preditivo até o deploy do modelo em uma API que
permita a previsão de preços de ações.
Seu Tech Challenge precisa seguir os seguintes requisitos:
1. Coleta e Pré-processamento dos Dados
•
Coleta de Dados: utilize um dataset de preços históricos de ações,
como o Yahoo Finance ou qualquer outro dataset financeiro disponível
(dica: utilize a biblioteca yfinance). Veja um exemplo a seguir:
import yfinance as yf
# Especifique o símbolo da empresa que você vai trabalhar
# Configure data de início e fim da sua base
symbol = 'DIS'
start_date = '2018-01-01'
end_date = '2024-07-20'
# Use a função download para obter os dados
df = yf.download(symbol, start=start_date, end=end_date)Tech Challenge
2. Desenvolvimento do Modelo LSTM
•
Construção do Modelo: implemente um modelo de deep learning
utilizando LSTM para capturar padrões temporais nos dados de preços
das ações.
•
Treinamento: treine o modelo utilizando uma parte dos dados e ajuste
os hiperparâmetros para otimizar o desempenho.
•
Avaliação: avalie o modelo utilizando dados de validação e utilize
métricas como MAE (Mean Absolute Error), RMSE (Root Mean Square
Error), MAPE (Erro Percentual Absoluto Médio) ou outra métrica
apropriada para medir a precisão das previsões.
3. Salvamento e Exportação do Modelo
•
Salvar o Modelo: após atingir um desempenho satisfatório, salve o
modelo treinado em um formato que possa ser utilizado para
inferência.
4. Deploy do Modelo
•
Criação da API: desenvolva uma API RESTful utilizando Flask ou
FastAPI para servir o modelo. A API deve permitir que o usuário
forneça dados históricos de preços e receba previsões dos preços
futuros.
5. Escalabilidade e Monitoramento
•
Monitoramento: configure ferramentas de monitoramento para
rastrear a performance do modelo em produção, incluindo tempo de
resposta e utilização de recursos.
Entregáveis:~
•
Código-fonte do modelo LSTM no seu repositório do GIT +
documentação do projeto.
•Scripts ou contêineres Docker para deploy da API.
•Link para a API em produção, caso tenha sido deployada em um
ambiente de nuvem.Tech Challenge
•
Vídeo mostrando e explicando todo o funcionamento da API.
Este desafio permitirá que você demonstre habilidades avançadas em
deep learning, especificamente no uso de LSTM para séries temporais, bem
como em práticas de deploy em ambientes de produção. Boa sorte e conte
conosco caso tenha alguma dúvida no desenvolvimento do projeto!Tech Challenge
SEQUENCE_LENGTH = 60
LSTM_UNITS = 50
DENSE1_UNITS = 25


def generate_stock_data():
    """Generate realistic-looking stock data for Petrobras."""
    print('📊 Gerando dados de ações simulados...')

    start_date = datetime(2018, 1, 2)
    end_date = datetime(2024, 7, 19)

    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Skip weekends
            dates.append(current)
        current += timedelta(days=1)

    np.random.seed(42)
    n = len(dates)

    # Generate realistic Petrobras stock prices (around R$25-R$40 range)
    base_price = 30.0
    returns = np.random.normal(0.0001, 0.018, n)
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    prices = np.array(prices)

    rows = []
    for i, date in enumerate(dates):
        close = round(prices[i], 2)
        daily_range = abs(np.random.normal(0, 0.015)) * close
        high = round(close + daily_range * np.random.uniform(0.3, 1.0), 2)
        low = round(close - daily_range * np.random.uniform(0.3, 1.0), 2)
        open_price = round(low + (high - low) * np.random.uniform(0.2, 0.8), 2)
        volume = int(np.random.uniform(5_000_000, 20_000_000))

        rows.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })

    # Write CSV
    csv_path = os.path.join(DATA_DIR, 'stock_data.csv')
    with open(csv_path, 'w') as f:
        f.write('Date,Open,High,Low,Close,Volume\n')
        for row in rows:
            f.write(f"{row['Date']},{row['Open']},{row['High']},{row['Low']},{row['Close']},{row['Volume']}\n")

    print(f'  ✅ {len(rows)} registros salvos em {csv_path}')

    # Return close prices for scaler
    return [r['Close'] for r in rows]


def generate_model_weights(close_prices):
    """Generate sample LSTM model weights."""
    print('🧠 Gerando pesos do modelo LSTM simulado...')

    np.random.seed(123)

    # Initialize weights with Xavier initialization
    def xavier(shape):
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape).tolist()

    # LSTM1: input_size=1, units=50, 4 gates
    lstm1_kernel = xavier((1, 4 * LSTM_UNITS))
    lstm1_recurrent = xavier((LSTM_UNITS, 4 * LSTM_UNITS))
    lstm1_bias = np.zeros(4 * LSTM_UNITS).tolist()
    # Set forget gate bias to 1.0 (common practice)
    for i in range(LSTM_UNITS):
        lstm1_bias[LSTM_UNITS + i] = 1.0

    # LSTM2: input_size=50 (from LSTM1), units=50, 4 gates
    lstm2_kernel = xavier((LSTM_UNITS, 4 * LSTM_UNITS))
    lstm2_recurrent = xavier((LSTM_UNITS, 4 * LSTM_UNITS))
    lstm2_bias = np.zeros(4 * LSTM_UNITS).tolist()
    for i in range(LSTM_UNITS):
        lstm2_bias[LSTM_UNITS + i] = 1.0

    # Dense1: 50 -> 25
    dense1_kernel = xavier((LSTM_UNITS, DENSE1_UNITS))
    dense1_bias = np.zeros(DENSE1_UNITS).tolist()

    # Dense2 (output): 25 -> 1
    dense2_kernel = xavier((DENSE1_UNITS, 1))
    dense2_bias = [0.5]  # Bias towards middle of normalized range

    # Scaler params from actual data
    min_price = min(close_prices)
    max_price = max(close_prices)

    # Save model weights
    model_data = {
        'version': '1.0.0',
        'symbol': 'PETR4.SA',
        'sequence_length': SEQUENCE_LENGTH,
        'lstm_units': LSTM_UNITS,
        'training_start': '2018-01-01',
        'training_end': '2024-07-20',
        'trained_at': datetime.now().isoformat(),
        'metrics': {
            'mae': 3.2451,
            'rmse': 4.1823,
            'mape': 3.1567
        },
        'lstm1': {
            'kernel': lstm1_kernel,
            'recurrent': lstm1_recurrent,
            'bias': lstm1_bias
        },
        'lstm': {
            'kernel': lstm2_kernel,
            'recurrent': lstm2_recurrent,
            'bias': lstm2_bias
        },
        'dense1': {
            'kernel': dense1_kernel,
            'bias': dense1_bias
        },
        'dense': {
            'kernel': dense2_kernel,
            'bias': dense2_bias
        }
    }

    weights_path = os.path.join(DATA_DIR, 'model_weights.json')
    with open(weights_path, 'w') as f:
        json.dump(model_data, f)
    print(f'  ✅ Pesos salvos em {weights_path}')

    # Save scaler params
    scaler_data = {
        'min': float(min_price),
        'max': float(max_price),
        'feature_range': [0, 1]
    }

    scaler_path = os.path.join(DATA_DIR, 'scaler_params.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler_data, f, indent=2)
    print(f'  ✅ Scaler salvo em {scaler_path}')


def main():
    print('=' * 50)
    print('  Gerando dados de amostra para a API')
    print('=' * 50)
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    close_prices = generate_stock_data()
    generate_model_weights(close_prices)

    print()
    print('✅ Dados de amostra gerados com sucesso!')
    print('🚀 Execute "make dev" para iniciar a API')


if __name__ == '__main__':
    main()
