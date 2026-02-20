import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title='Stock Prediction - Dashboard',
    page_icon='📈',
    layout='wide',
)

API_BASE_URL = os.getenv('BASE_URL', 'http://localhost:8081')


def fetch_api_data(endpoint, method='GET', json_body=None):
    try:
        url = f'{API_BASE_URL}{endpoint}'
        if method == 'POST':
            response = requests.post(url, json=json_body, timeout=10)
        else:
            response = requests.get(url, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f'Erro ao buscar {endpoint}: {response.status_code}')
            return None
    except requests.exceptions.ConnectionError:
        st.error(f'❌ Não foi possível conectar à API em {API_BASE_URL}. Verifique se a API está rodando (make dev).')
        return None
    except Exception as e:
        st.error(f'Erro: {str(e)}')
        return None


def main():
    st.title('📈 Stock Prediction - Dashboard de Monitoramento')
    st.markdown('---')

    st.sidebar.title('Navegação')
    page = st.sidebar.selectbox(
        'Escolha uma página',
        ['Overview', 'API Metrics', 'Performance', 'Predições', 'Real-time Logs'],
    )

    if page == 'Overview':
        show_overview()
    elif page == 'API Metrics':
        show_api_metrics()
    elif page == 'Performance':
        show_performance()
    elif page == 'Predições':
        show_predictions()
    elif page == 'Real-time Logs':
        show_realtime_logs()


def show_overview():
    st.header('📊 Overview')

    health_data = fetch_api_data('/api/v1/health')
    model_data = fetch_api_data('/api/v1/predictions/model-info')
    metrics_data = fetch_api_data('/api/v1/analytics/metrics')

    col1, col2, col3, col4 = st.columns(4)

    if health_data:
        with col1:
            status = health_data.get('status', 'unknown')
            icon = '🟢' if status == 'healthy' else '🔴'
            st.metric('Status da API', f'{icon} {status}')

        with col2:
            data_info = health_data.get('data', {})
            records = data_info.get('total_records', 0)
            last_date = data_info.get('last_date', 'N/A')
            st.metric('Registros', records)
            st.metric('Último Dado', last_date)

    if model_data:
        with col3:
            version = model_data.get('version', 'N/A')
            st.metric('Versão do Modelo', version)

        with col4:
            symbol = model_data.get('symbol', 'N/A')
            st.metric('Ação', symbol)
            
        currency = 'R$' if symbol.endswith('.SA') else '$'

    # Métricas do modelo
    if model_data and model_data.get('metrics'):
        st.subheader('🤖 Métricas do Modelo LSTM')
        metrics = model_data['metrics']
        baselines = model_data.get('baselines', {})
        horizon = model_data.get('horizon_days', 5)

        col1, col2, col3, col4 = st.columns(4)
        
        mape = metrics.get('mape', 0)
        dir_acc = metrics.get('directional_accuracy', 0)
        
        with col1:
            st.metric(f'MAPE (D+{horizon})', f"{mape:.2f}%")
        with col2:
            st.metric('Acurácia Direcional', f"{dir_acc:.2f}%")
        with col3:
            st.metric('MAE', f"{metrics.get('mae', 0):.4f}")
        with col4:
            st.metric('RMSE', f"{metrics.get('rmse', 0):.4f}")

        st.info(
            f'**Horizonte de Previsão**: {horizon} dias.\n'
            '**Acurácia Direcional**: % de acerto na direção (sobe/desce).\n'
            '**MAPE**: Erro percentual médio.'
        )
        
        if baselines:
            st.markdown("### 📉 Comparação com Baselines")
            naive = baselines.get('naive', {})
            sma = list(baselines.keys())[-1] if len(baselines) > 1 else 'sma' 
            sma_data = baselines.get(sma, {}) if sma != 'naive' else {} # simple heuristic
            
            # If sma key is like "sma_60", use it
            for k in baselines:
                if k.startswith('sma'):
                    sma_data = baselines[k]
                    sma = k
                    break

            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                st.metric("LSTM MAPE", f"{mape:.2f}%")
            with b_col2:
                st.metric("Naive MAPE", f"{naive.get('mape_price_pct', 0):.2f}%", 
                         delta=f"{naive.get('mape_price_pct', 0) - mape:.2f}%")
            with b_col3:
                st.metric(f"{sma.upper()} MAPE", f"{sma_data.get('mape_price_pct', 0):.2f}%",
                         delta=f"{sma_data.get('mape_price_pct', 0) - mape:.2f}%")

        st.subheader('🏗️ Arquitetura do Modelo')
        training_period = model_data.get('training_period', {})
        last_trained = model_data.get('last_trained', '')
        
        features = model_data.get('features_used', [])
        features_str = ', '.join(features[:5]) + ('...' if len(features) > 5 else '')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            - **Tipo:** LSTM Multi-Feature
            - **Sequência:** {model_data.get('sequence_length', 'N/A')} dias
            - **Horizonte:** {horizon} dias
            - **Features ({len(features)}):** {features_str}
            """)
        with col2:
            st.markdown(f"""
            - **Período treino:** {training_period.get('start', 'N/A')} a {training_period.get('end', 'N/A')}
            - **Treinado em:** {last_trained[:10] if last_trained else 'N/A'}
            """)

    if metrics_data:
        st.subheader('📋 Resumo de Atividade')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total de Requests', metrics_data.get('total_requests', 0))
        with col2:
            st.metric('Tempo Médio', f"{metrics_data.get('average_response_time', 0):.2f}ms")
        with col3:
            st.metric('Taxa de Erros', f"{metrics_data.get('error_rate', 0):.2f}%")

    st.subheader('📈 Dados Recentes da Ação')
    stock_data = fetch_api_data('/api/v1/stocks/latest?n=60')
    if stock_data and isinstance(stock_data, list) and len(stock_data) > 0:
        df = pd.DataFrame(stock_data)
        if 'date' in df.columns and 'close' in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['close'],
                mode='lines',
                name='Preço de Fechamento',
                line=dict(color='#00d4aa', width=2),
            ))
            fig.update_layout(
                title='Preço de Fechamento - Últimos 60 dias',
                xaxis_title='Data',
                yaxis_title=f'Preço ({currency})',
                template='plotly_dark',
            )
            st.plotly_chart(fig, use_container_width=True)


def show_api_metrics():
    st.header('📊 API Metrics')

    analytics_data = fetch_api_data('/api/v1/analytics/metrics')

    if not analytics_data:
        st.warning('Sem dados de métricas. Faça algumas requisições à API primeiro.')
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total de Requests', analytics_data.get('total_requests', 0))
    with col2:
        st.metric('Tempo Médio de Resposta', f"{analytics_data.get('average_response_time', 0):.2f}ms")
    with col3:
        st.metric('Taxa de Erros', f"{analytics_data.get('error_rate', 0):.2f}%")

    col1, col2 = st.columns(2)

    if analytics_data.get('requests_by_endpoint'):
        with col1:
            st.subheader('📈 Requests por Endpoint')
            endpoint_data = analytics_data['requests_by_endpoint']
            fig = px.bar(
                x=list(endpoint_data.keys()),
                y=list(endpoint_data.values()),
                title='Distribuição por Endpoint',
                labels={'x': 'Endpoint', 'y': 'Requests'},
                color=list(endpoint_data.values()),
                color_continuous_scale='teal',
            )
            fig.update_layout(template='plotly_dark', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    if analytics_data.get('requests_by_method'):
        with col2:
            st.subheader('🔧 Requests por Método HTTP')
            method_data = analytics_data['requests_by_method']
            fig = px.pie(
                values=list(method_data.values()),
                names=list(method_data.keys()),
                title='Distribuição por Método',
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

    if analytics_data.get('requests_by_status'):
        st.subheader('📊 Requests por Status Code')
        status_data = analytics_data['requests_by_status']
        fig = px.bar(
            x=list(status_data.keys()),
            y=list(status_data.values()),
            title='Distribuição por Status Code',
            labels={'x': 'Status Code', 'y': 'Requests'},
            color=list(status_data.keys()),
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    if analytics_data.get('recent_activity'):
        st.subheader('📋 Atividade Recente')
        recent_df = pd.DataFrame(analytics_data['recent_activity'])
        if not recent_df.empty:
            display_cols = [c for c in ['timestamp', 'method', 'path', 'status_code', 'duration_ms'] if c in recent_df.columns]
            st.dataframe(recent_df[display_cols], use_container_width=True)


def show_performance():
    st.header('⚡ Performance')

    performance_data = fetch_api_data('/api/v1/analytics/performance')

    if not performance_data:
        st.warning('Sem dados de performance. Faça algumas requisições à API primeiro.')
        return

    if performance_data.get('system_health'):
        st.subheader('🏥 Saúde do Sistema')
        health = performance_data['system_health']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Uptime', f"{health.get('uptime_percentage', 0):.1f}%")
        with col2:
            st.metric('Total Requests', health.get('total_requests', 0))
        with col3:
            st.metric('Erros', health.get('error_count', 0))
        with col4:
            st.metric('Taxa de Sucesso', f"{health.get('success_rate', 0):.1f}%")

    if performance_data.get('response_time_distribution'):
        st.subheader('⏱️ Distribuição de Tempo de Resposta')
        rt_dist = performance_data['response_time_distribution']

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric('Min', f"{rt_dist.get('min', 0):.2f}ms")
        with col2:
            st.metric('Mediana', f"{rt_dist.get('median', 0):.2f}ms")
        with col3:
            st.metric('Média', f"{rt_dist.get('mean', 0):.2f}ms")
        with col4:
            st.metric('P95', f"{rt_dist.get('p95', 0):.2f}ms")
        with col5:
            st.metric('P99', f"{rt_dist.get('p99', 0):.2f}ms")

    if performance_data.get('endpoint_performance'):
        st.subheader('🎯 Performance por Endpoint')
        endpoint_perf = performance_data['endpoint_performance']

        perf_data = []
        for endpoint, data in endpoint_perf.items():
            perf_data.append({
                'endpoint': endpoint,
                'avg_ms': data['avg_response_time'],
                'min_ms': data['min_response_time'],
                'max_ms': data['max_response_time'],
                'requests': data['request_count'],
            })

        if perf_data:
            perf_df = pd.DataFrame(perf_data)

            fig = px.bar(
                perf_df,
                x='endpoint',
                y='avg_ms',
                title='Tempo Médio de Resposta por Endpoint',
                labels={'avg_ms': 'Tempo Médio (ms)', 'endpoint': 'Endpoint'},
                color='avg_ms',
                color_continuous_scale='RdYlGn_r',
            )
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(perf_df, use_container_width=True)

    if performance_data.get('error_breakdown'):
        st.subheader('❌ Breakdown de Erros')
        error_data = performance_data['error_breakdown']
        if error_data:
            fig = px.pie(
                values=list(error_data.values()),
                names=list(error_data.keys()),
                title='Distribuição de Erros por Tipo',
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)


def show_predictions():
    st.header('🔮 Predições')

    st.markdown('Teste o modelo LSTM de predição de preços diretamente pelo dashboard.')

    predict_btn = st.button('🚀 Gerar Predição', use_container_width=True)

    if predict_btn:
        with st.spinner('Gerando predição...'):
            result = fetch_api_data('/api/v1/predictions/predict', method='GET')

        if result and result.get('prediction'):
            prediction = result['prediction']
            currency = 'R$' if result.get('symbol', '').endswith('.SA') else '$'

            st.subheader('📊 Resultado da Predição')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Símbolo', result.get('symbol', 'N/A'))
            with col2:
                st.metric('Data Prevista', prediction.get('date', 'N/A'))
            with col3:
                if result.get('metrics'):
                    st.metric('MAPE do Modelo', f"{result['metrics'].get('mape', 'N/A')}%")

            st.metric(
                f'Preço Previsto ({currency})',
                f"{currency} {prediction.get('predicted_close', 'N/A')}",
            )

            st.subheader('📋 Dados da Predição')
            st.dataframe(pd.DataFrame([prediction]), use_container_width=True)

    # Info do modelo
    st.markdown('---')
    st.subheader('ℹ️ Informações do Modelo')
    model_info = fetch_api_data('/api/v1/predictions/model-info')
    if model_info:
        col1, col2 = st.columns(2)
        with col1:
            st.json({
                'symbol': model_info.get('symbol'),
                'model_version': model_info.get('model_version'),
                'lstm_units': model_info.get('lstm_units'),
                'sequence_length': model_info.get('sequence_length'),
            })
        with col2:
            if model_info.get('metrics'):
                st.json(model_info['metrics'])


def show_realtime_logs():
    st.header('📝 Real-time Logs')

    log_file = 'logs/api.log'

    if os.path.exists(log_file):
        st.subheader('📄 Entradas de Log Recentes')

        with open(log_file, 'r') as f:
            lines = f.readlines()

        if lines:
            recent_lines = lines[-50:] if len(lines) > 50 else lines
            log_content = ''.join(recent_lines)
            st.text_area('Logs Recentes', log_content, height=400)

            if st.button('🔄 Atualizar Logs'):
                st.rerun()
        else:
            st.info('Nenhuma entrada de log encontrada.')

        st.subheader('📊 Estatísticas de Log')

        api_calls = [line for line in lines if 'API_CALL:' in line]
        errors = [line for line in lines if 'API_ERROR:' in line]
        predictions_log = [line for line in lines if 'Predição' in line]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total de Linhas', len(lines))
        with col2:
            st.metric('API Calls', len(api_calls))
        with col3:
            st.metric('Erros', len(errors))
        with col4:
            st.metric('Predições', len(predictions_log))
    else:
        st.warning('Arquivo de log não encontrado. Verifique se a API está rodando com ENV=LOCAL (para habilitar logs em arquivo).')
        st.info('Dica: exporte `ENV=LOCAL` antes de rodar `make dev` para habilitar logs em arquivo.')


if __name__ == '__main__':
    main()
