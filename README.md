# Stock Price Prediction API - FIAP Tech Challenge Fase 4

API RESTful para predição de preços de ações usando modelo LSTM (Long Short-Term Memory), desenvolvida com FastAPI e deploy na Vercel.

## 🎯 Sobre o Projeto

Este projeto implementa um modelo preditivo de deep learning (LSTM) para predizer o valor de fechamento das ações da **Petrobras (PETR4.SA)**, incluindo toda a pipeline desde a coleta de dados até o deploy em produção.

### Arquitetura

O projeto segue Clean Architecture com as seguintes camadas:

```
api/
├── domain/          # Modelos, interfaces e regras de negócio
│   ├── models/      # Pydantic models
│   ├── repositories/# Interfaces abstratas
│   └── usecases/    # Casos de uso
├── infra/           # Implementações concretas
│   └── repositories/# Implementação dos repositórios
├── presentation/    # Interface HTTP
│   ├── routes/      # Endpoints da API
│   ├── middlewares/  # Middlewares (performance, errors)
│   └── factories/   # Factory para injeção de dependência
└── utils/           # Utilitários (logger)
```

## 🚀 Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/` | Informações da API |
| `GET` | `/api/v1/health` | Status de saúde da API |
| `GET` | `/api/v1/stocks/history?limit=100` | Dados históricos da ação |
| `GET` | `/api/v1/stocks/latest?n=30` | Dados mais recentes |
| `POST` | `/api/v1/predictions/predict` | Predição de preços futuros |
| `GET` | `/api/v1/predictions/model-info` | Informações do modelo LSTM |

### Exemplo de Predição

```bash
curl -X POST http://localhost:8081/api/v1/predictions/predict \
  -H "Content-Type: application/json" \
  -d '{"days_ahead": 7}'
```

**Resposta:**
```json
{
  "symbol": "PETR4.SA",
  "predictions": [
    {"date": "2024-07-22", "predicted_close": 98.45},
    {"date": "2024-07-23", "predicted_close": 99.12}
  ],
  "model_version": "1.0.0",
  "generated_at": "2024-07-20T15:30:00",
  "metrics": {"mae": 2.34, "rmse": 3.12, "mape": 2.89}
}
```

## 🛠️ Setup Local

### Pré-requisitos
- Python 3.12+
- Poetry

### Instalação

```bash
cd fiap4

# Setup completo do projeto
make setup-project

# Ou instalar apenas dependências
make install
```

### Treinar o Modelo

O modelo LSTM precisa ser treinado antes de usar a API de predição:

```bash
# Instalar dependências de treinamento (TensorFlow, yfinance, etc.)
make setup-training

# Executar treinamento
make train
```

O script irá:
1. Baixar dados históricos da Petrobras (PETR4.SA) via Yahoo Finance
2. Treinar modelo LSTM com 2 camadas (50 unidades cada)
3. Avaliar com métricas MAE, RMSE e MAPE
4. Salvar pesos em `data/model_weights.json`
5. Salvar dados em `data/stock_data.csv`

### Iniciar API

```bash
make dev
# API disponível em http://localhost:8081
# Documentação em http://localhost:8081/docs
```

## 📊 Modelo LSTM

### Arquitetura
```
Input (60 timesteps, 1 feature)
  → LSTM (50 units, return_sequences=True)
  → Dropout (0.2)
  → LSTM (50 units)
  → Dropout (0.2)
  → Dense (25 units)
  → Dense (1 unit) → Output (preço previsto)
```

### Dados
- **Empresa:** Petrobras (PETR4.SA)
- **Período:** 2018-01-01 a 2024-07-20
- **Feature:** Preço de fechamento (Close)
- **Sequência:** 60 dias anteriores para prever o próximo

### Estratégia de Deploy
- **Treinamento:** Local com TensorFlow/Keras
- **Inferência:** Numpy puro (sem TensorFlow na Vercel)
- Os pesos do modelo são exportados em JSON e a inferência é feita reconstruindo o forward pass do LSTM com numpy

## ☁️ Deploy na Vercel

```bash
# Deploy de produção
make deploy-prod

# Preview deploy
make deploy-dev
```

A API usa apenas `numpy` para inferência, mantendo o pacote leve o suficiente para a Vercel (sem TensorFlow).

## 📁 Estrutura de Dados

```
data/
├── stock_data.csv       # Dados históricos da ação
├── model_weights.json   # Pesos do modelo LSTM
└── scaler_params.json   # Parâmetros de normalização
```

## 🔧 Tecnologias

- **FastAPI** - Framework web
- **TensorFlow/Keras** - Treinamento do modelo LSTM
- **NumPy** - Inferência (forward pass)
- **Pandas** - Manipulação de dados
- **yfinance** - Coleta de dados financeiros
- **Vercel** - Deploy em produção
