"""
Decision Intelligence & Inventory Replenishment Recommendation Engine
Implements Operational Formulas & Manager Action Cards for IntelliData 2026.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib

def generate_recommendations():
    print("--- [1/3] Loading Models & Latest Store State ---")
    reg_model = joblib.load('models/demand_forecast_model.joblib')
    clf_model = joblib.load('models/stockout_risk_model.joblib')
    with open('models/model_metadata.json', 'r') as f:
        meta = json.load(f)
    feature_cols = meta['feature_cols']

    # Load latest operational snapshot from master dataset with full coverage
    df = pd.read_parquet('data/processed/featured_dataset.parquet')
    date_counts = df['date'].value_counts()
    valid_dates = date_counts[date_counts >= 1500].index
    latest_date = valid_dates.max()
    print(f"Generating live recommendations for latest fully-covered operational date: {latest_date.strftime('%Y-%m-%d')}")
    
    snapshot = df[df['date'] == latest_date].copy().reset_index(drop=True)
    print(f"Snapshot contains {len(snapshot):,} Store x Product combinations across {snapshot['store_id'].nunique()} stores.")

    # Encode categoricals
    for c in meta['cat_cols']:
        snapshot[c + '_code'] = snapshot[c].astype('category').cat.codes

    X_snap = snapshot[feature_cols]

    print("\n--- [2/3] Generating Model Predictions ---")
    # Model 1: 7-Day Demand Forecast
    snapshot['predicted_7d_demand'] = np.round(np.maximum(0, reg_model.predict(X_snap)), 1)
    
    # Model 2: Stockout Probability & Risk Classification
    snapshot['stockout_prob'] = np.round(clf_model.predict_proba(X_snap)[:, 1], 3)
    
    # Triage Risk Tiers according to Hackathon Specifications
    # High: >= 0.70, Medium: 0.40 - 0.70, Low: < 0.40
    def classify_risk(p):
        if p >= 0.70:
            return 'HIGH'
        elif p >= 0.40:
            return 'MEDIUM'
        else:
            return 'LOW'
            
    snapshot['risk_level'] = snapshot['stockout_prob'].apply(classify_risk)

    print("\n--- [3/3] Calculating Replenishment Logic & Action Cards ---")
    # Z-factor for 95% service level = 1.65
    Z = 1.65
    daily_sigma = np.maximum(0.5, snapshot['rolling_std_7'])
    lead_time = np.maximum(1, snapshot['lead_days'])
    
    # 1. Safety Stock = Z * sigma * sqrt(Lead Days)
    snapshot['safety_stock'] = np.ceil(Z * daily_sigma * np.sqrt(lead_time)).astype(int)
    
    # 2. Recommended Stock = Forecast 7d Demand + Safety Stock
    snapshot['recommended_stock'] = np.ceil(snapshot['predicted_7d_demand'] + snapshot['safety_stock']).astype(int)
    
    # 3. Incoming stock simulation (heuristic: received in transit)
    snapshot['incoming_stock'] = snapshot['received'].fillna(0).astype(int)
    
    # 4. Reorder Quantity = max(0, Recommended Stock - Current Stock - Incoming Stock)
    snapshot['recommended_reorder_qty'] = np.maximum(
        0, 
        snapshot['recommended_stock'] - snapshot['current_stock'] - snapshot['incoming_stock']
    ).astype(int)

    # 5. Manager Action & Driver Reasoning Generator
    def generate_action_text(row):
        risk = row['risk_level']
        reorder = row['recommended_reorder_qty']
        doi = row['days_of_inventory']
        promo = row['current_promo_active']
        weekend = row['is_weekend']
        lead = row['lead_days']
        
        reasons = []
        if risk == 'HIGH':
            action = f"URGENT: Raise Purchase Order for {reorder:,} units today."
        elif risk == 'MEDIUM':
            action = f"SCHEDULE: Plan reorder of {reorder:,} units within {lead} days." if reorder > 0 else "MONITOR: Stock levels adequate for current cycle."
        else:
            action = "MAINTAIN: Stock healthy; no immediate replenishment required."
            
        if doi < 2.0:
            reasons.append(f"Critical inventory depth ({doi:.1f} days remaining)")
        elif doi < 4.0:
            reasons.append(f"Low inventory coverage ({doi:.1f} days)")
            
        if promo == 1:
            reasons.append("Active promotional campaign driving +168% demand lift")
        if weekend == 1:
            reasons.append("Upcoming weekend footfall acceleration")
        if row['current_stock'] <= row['reorder_lvl']:
            reasons.append(f"Breached safety reorder point ({row['current_stock']} <= {row['reorder_lvl']})")
        if not reasons:
            reasons.append(f"Stable demand pattern (7d forecast: {row['predicted_7d_demand']} units)")
            
        return action, "; ".join(reasons)

    actions_and_reasons = snapshot.apply(generate_action_text, axis=1)
    snapshot['manager_action'] = [ar[0] for ar in actions_and_reasons]
    snapshot['key_drivers_reason'] = [ar[1] for ar in actions_and_reasons]

    # Select Manager Action Centre Output Columns
    output_columns = [
        'store_id', 'city', 'store_type', 'product_id', 'category', 'sub_category', 'brand',
        'current_stock', 'reorder_lvl', 'lead_days', 'predicted_7d_demand', 'stockout_prob',
        'risk_level', 'safety_stock', 'recommended_stock', 'incoming_stock',
        'recommended_reorder_qty', 'days_of_inventory', 'manager_action', 'key_drivers_reason'
    ]
    recom_df = snapshot[output_columns].sort_values(['stockout_prob', 'recommended_reorder_qty'], ascending=[False, False])

    print(f"\nReplenishment Summary:")
    print(f"Total Products Analyzed: {len(recom_df):,}")
    print(f"Risk Breakdown:\n{recom_df['risk_level'].value_counts()}")
    print(f"Total Replenishment Units Recommended: {recom_df['recommended_reorder_qty'].sum():,}")
    
    # Save recommendation dataset
    output_csv = 'data/processed/manager_replenishment_recommendations.csv'
    recom_df.to_csv(output_csv, index=False)
    print(f"Saved recommendations to {output_csv}")

    # Display Top 5 Urgent High-Risk Cards
    print("\n--- SAMPLE MANAGER RECOMMENDATION CARDS ---")
    for idx, r in recom_df[recom_df['risk_level'] == 'HIGH'].head(3).iterrows():
        print(f"""
=============================================================
STORE {r['store_id']} ({r['city']} - {r['store_type']}) | PRODUCT {r['product_id']} ({r['brand']} - {r['category']})
Predicted 7-Day Demand: {r['predicted_7d_demand']} units | Safety Stock: {r['safety_stock']} units
Current Stock: {r['current_stock']} units | Incoming Stock: {r['incoming_stock']} units
Stock-out Probability: {r['stockout_prob']*100:.1f}% | Risk Tier: {r['risk_level']}
RECOMMENDED REORDER: {r['recommended_reorder_qty']} units
Why: {r['key_drivers_reason']}
MANAGER ACTION: {r['manager_action']}
=============================================================
        """)

if __name__ == '__main__':
    generate_recommendations()
