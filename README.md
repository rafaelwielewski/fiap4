# Stock Price Prediction API - FIAP Tech Challenge Fase 4

API RESTful para predição de preços de ações usando modelo LSTM (Long Short-Term Memory), desenvolvida com FastAPI e deploy na Vercel.

## 🎯 Sobre o Projeto

Este projeto implementa um modelo preditivo de deep learning (LSTM) para predizer o valor de fechamento das ações da **Apple (AAPL)**, incluindo toda a pipeline desde a coleta de dados até o deploy em produção.

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
│   ├── middlewares/ # Middlewares (performance, errors)
│   └── factories/   # Factory para injeção de dependência
└── utils/           # Utilitários (logger)
├── artifacts/       # Artefatos do modelo (Keras, Scalers, Metadata)
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
curl http://localhost:8081/api/v1/predictions/predict
```

**Resposta:**
```json
{
  "symbol": "AAPL",
  "prediction": {"date": "2026-02-26", "predicted_close": 241.25},
  "model_version": "2.0.0",
  "generated_at": "2026-02-19T20:35:00",
  "metrics": {"mae": 7.32, "rmse": 9.86, "mape": 3.26, "directional_accuracy": 51.52}
}
```

### Dashboard

Acesse o dashboard interativo em: `http://localhost:8501`

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
1. Baixar dados históricos da Apple (AAPL) via Yahoo Finance
2. Calcular 16 indicadores técnicos (RSI, MACD, SMA, EMA, Volatilidade...)
3. Treinar modelo LSTM com features arquitetura multi-input
4. Avaliar com métricas MAE, RMSE, MAPE e Acurácia Direcional
5. Salvar modelo (`final_model.keras`) e artefatos em `artifacts/`

### Iniciar API

```bash
make dev
# API disponível em http://localhost:8081
# Documentação em http://localhost:8081/docs
```

## 📊 Modelo LSTM

### Arquitetura
```
Input (60 timesteps, 16 features)
  → LSTM (50 units, return_sequences=True)
  → Dropout (0.2)
  → LSTM (50 units)
  → Dropout (0.2)
  → Dense (25 units)
  → Dense (1 unit) → Output (preço previsto)
```

### Dados
- **Empresa:** Apple (AAPL)
- **Período:** 2018-01-01 a Presente
- **Features:** 16 (Close, Open, High, Low, Volume, RSI, MACD, etc.)
- **Sequência:** 60 dias anteriores para prever o próximo

### Estratégia de Deploy
- **Treinamento:** Local com TensorFlow/Keras
- **Inferência:** API carrega o modelo Keras otimizado (`.keras`)
- O modelo prevê a variação (delta) do preço para maior estabilidade

## ☁️ Deploy na Vercel

```bash
# Deploy de produção
make deploy-prod

# Preview deploy
make deploy-dev
```

O deploy na Vercel pode exigir configuração de tamanho devido ao TensorFlow. Recomenda-se Docker/Render/Railway para produção full.

## 📁 Estrutura de Dados

```
artifacts/
├── final_model.keras    # Modelo treinado
├── scaler_X.joblib      # Scaler de features
├── scaler_y.joblib      # Scaler de target
├── metadata.json        # Metadados do treinamento
└── metrics.json         # Métricas de avaliação
data/
└── stock_data.csv       # Dados históricos (backup)
```

## 🔧 Tecnologias

- **FastAPI** - Framework web
- **TensorFlow/Keras** - Treinamento do modelo LSTM
- **NumPy** - Inferência (forward pass)
- **Pandas** - Manipulação de dados
- **yfinance** - Coleta de dados financeiros
- **Vercel** - Deploy em produção
