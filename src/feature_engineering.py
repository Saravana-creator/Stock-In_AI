"""
Feature Engineering & Target Formulation Pipeline
Strictly Avoids Data Leakage for IntelliData 2026 Hackathon.
"""

import os
import pandas as pd
import numpy as np

def run_feature_engineering():
    print("--- [1/4] Loading Master Analytics Dataset ---")
    df = pd.read_parquet('data/processed/master_analytics_data.parquet')
    
    # Sort deterministically
    df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded master table: {df.shape[0]:,} rows across {df['date'].nunique()} dates.")

    print("\n--- [2/4] Engineering Leakage-Free Predictive Features ---")
    grouped_demand = df.groupby(['store_id', 'product_id'])['daily_demand']
    
    # 1. Historical Demand Lags (shifted by 1 day to represent past observed demand at decision time T)
    print("Generating lag features (lag_1, lag_7, lag_14)...")
    df['lag_1'] = grouped_demand.shift(1)
    df['lag_7'] = grouped_demand.shift(7)
    df['lag_14'] = grouped_demand.shift(14)
    
    # 2. Rolling Demand Statistics (strictly historical window up to T-1)
    print("Generating rolling statistics (rolling_mean_7, rolling_mean_14, rolling_std_7)...")
    df['rolling_mean_7'] = grouped_demand.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df['rolling_sum_7'] = grouped_demand.transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
    df['rolling_mean_14'] = grouped_demand.transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    df['rolling_std_7'] = grouped_demand.transform(lambda s: s.shift(1).rolling(7, min_periods=1).std()).fillna(0)

    # 3. Inventory Features (Current state at decision point T)
    print("Generating inventory velocity and coverage metrics...")
    df['current_stock'] = df['reconciled_closing']
    df['reorder_gap'] = df['current_stock'] - df['reorder_lvl']
    df['days_of_inventory'] = df['current_stock'] / (df['rolling_mean_7'] + 0.01)
    df['inventory_to_reorder_ratio'] = df['current_stock'] / (df['reorder_lvl'] + 0.01)
    df['is_below_reorder'] = (df['current_stock'] <= df['reorder_lvl']).astype(int)

    # 4. Pricing & Margin Features
    df['profit_margin'] = (df['mrp'] - df['cost_price']) / (df['mrp'] + 1e-5)
    df['current_discount_pct'] = df['avg_discount_pct']
    df['current_promo_active'] = df['promotion_active']

    # 5. Temporal & Calendar Features
    print("Generating temporal calendar features...")
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter

    print("\n--- [3/4] Formulating Forward Prediction Targets ---")
    # Model 1 Target: next_7_day_demand = sum(demand) over [T+1, ..., T+7]
    indexer_7 = pd.api.indexers.FixedForwardWindowIndexer(window_size=7)
    df['next_7_day_demand'] = grouped_demand.transform(
        lambda s: s.shift(-1).rolling(window=indexer_7, min_periods=7).sum()
    )

    # Model 2 Target: stockout_flag = 1 if min(reconciled_closing) over [T+1, ..., T+7] <= 0
    grouped_stock = df.groupby(['store_id', 'product_id'])['reconciled_closing']
    min_stock_next_7 = grouped_stock.transform(
        lambda s: s.shift(-1).rolling(window=indexer_7, min_periods=7).min()
    )
    df['stockout_flag'] = (min_stock_next_7 <= 0).astype(int)

    # Drop edges: first 14 days have NaN lags, last 7 days lack forward target
    print("\n--- [4/4] Pruning Unobservable Boundary Rows ---")
    valid_mask = df['lag_14'].notna() & df['next_7_day_demand'].notna()
    featured_df = df[valid_mask].copy().reset_index(drop=True)
    
    print(f"Features ready! Original rows: {len(df):,} -> Usable modeling records: {len(featured_df):,}")
    print(f"Target 1 (next_7_day_demand) mean: {featured_df['next_7_day_demand'].mean():.2f} units, "
          f"median: {featured_df['next_7_day_demand'].median():.2f}")
    print(f"Target 2 (stockout_flag) prevalence: {featured_df['stockout_flag'].mean()*100:.2f}% positive")

    # Save to parquet
    output_path = 'data/processed/featured_dataset.parquet'
    featured_df.to_parquet(output_path, index=False)
    print(f"Saved featured dataset to {output_path}")

    # Also save a sample for quick inspection
    featured_df.head(5000).to_csv('data/processed/featured_dataset_sample.csv', index=False)
    print("Saved sample to data/processed/featured_dataset_sample.csv")

if __name__ == '__main__':
    run_feature_engineering()
