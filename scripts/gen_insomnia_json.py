import json
import os
from datetime import datetime, timedelta
import random

WORKSPACE_PATH = 'insomnia_workspace.json'

def generate_historical_data():
    days = 90
    # End date: Yesterday or Today (if market open). Let's use 2026-02-18 (Wednesday)
    end_date = datetime(2026, 2, 18)
    
    data = []
    current = end_date
    price = 37.0 # Close to current real price
    
    while len(data) < days:
        if current.weekday() < 5:
            change = random.uniform(-0.5, 0.5)
            price -= change # Going backwards, so subtract change (approx)
            data.append({
                "date": current.strftime('%Y-%m-%d'),
                "close": round(price, 2)
            })
        current -= timedelta(days=1)
    
    # Reverse to be chronological
    data.reverse()
    
    # Recalculate prices forward to look more natural (random walk forward)
    # Start from the first price we found (approx)
    start_price = data[0]['close']
    current_price = start_price
    for item in data:
         change = random.uniform(-0.8, 0.8)  # slightly more volatility
         current_price += change
         item['close'] = round(current_price, 2)
         
    return data

def main():
    if not os.path.exists(WORKSPACE_PATH):
        print(f"Error: {WORKSPACE_PATH} not found")
        return

    with open(WORKSPACE_PATH, 'r') as f:
        workspace = json.load(f)

    updated_count = 0
    
    resources = workspace
    if isinstance(workspace, dict) and 'resources' in workspace:
        resources = workspace['resources']
    
    historical_data = generate_historical_data()
    print(f"Generated {len(historical_data)} records from {historical_data[0]['date']} to {historical_data[-1]['date']}")
    
    for res in resources:
        if res.get('_type') == 'request':
            # Update "Predict Custom (User Data)"
            if res.get('name') == 'Predict Custom (User Data)' or 'predict-custom' in res.get('url', ''):
                try:
                    # We rewrite the whole body
                    new_body = {
                        "days_ahead": 7,
                        "historical_data": historical_data
                    }
                    res['body']['text'] = json.dumps(new_body, indent=2)
                    print(f"Updated {res['name']} with 90 days of data ending {historical_data[-1]['date']}")
                    updated_count += 1
                except Exception as e:
                    print(f"Failed to update custom: {e}")

    with open(WORKSPACE_PATH, 'w') as f:
        json.dump(workspace, f, indent=2)
        
    print(f"Saved {WORKSPACE_PATH}. Updated {updated_count} requests.")

if __name__ == '__main__':
    main()
