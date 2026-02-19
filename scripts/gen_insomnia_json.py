import json
import os

import yfinance as yf

WORKSPACE_PATH = 'insomnia_workspace.json'


def fetch_aapl_data(days: int = 90) -> list[dict]:
    ticker = yf.Ticker('AAPL')
    # Fetch enough trading days (calendar days buffer ~1.5x to cover weekends/holidays)
    df = ticker.history(period=f'{int(days * 1.5)}d', interval='1d', auto_adjust=True)
    df = df.dropna(subset=['Close'])
    df = df.tail(days).reset_index()

    data = []
    for _, row in df.iterrows():
        data.append({
            'date': row['Date'].strftime('%Y-%m-%d'),
            'close': round(float(row['Close']), 2),
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'volume': int(row['Volume']),
        })

    return data


def main():
    if not os.path.exists(WORKSPACE_PATH):
        print(f'Error: {WORKSPACE_PATH} not found')
        return

    with open(WORKSPACE_PATH, 'r') as f:
        workspace = json.load(f)

    print('Fetching real AAPL data from Yahoo Finance...')
    historical_data = fetch_aapl_data(days=90)
    print(f'Fetched {len(historical_data)} records from {historical_data[0]["date"]} to {historical_data[-1]["date"]}')

    resources = workspace
    if isinstance(workspace, dict) and 'resources' in workspace:
        resources = workspace['resources']

    updated_count = 0
    for res in resources:
        if res.get('_type') == 'request':
            if res.get('name') == 'Predict Custom (User Data)' or 'predict-custom' in res.get('url', ''):
                try:
                    new_body = {
                        'days_ahead': 7,
                        'historical_data': historical_data,
                    }
                    res['body']['text'] = json.dumps(new_body, indent=2)
                    print(f'Updated {res["name"]} with real AAPL data ending {historical_data[-1]["date"]}')
                    updated_count += 1
                except Exception as e:
                    print(f'Failed to update custom: {e}')

    with open(WORKSPACE_PATH, 'w') as f:
        json.dump(workspace, f, indent=2)

    print(f'Saved {WORKSPACE_PATH}. Updated {updated_count} requests.')


if __name__ == '__main__':
    main()
