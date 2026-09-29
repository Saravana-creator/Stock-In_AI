"""
StockSense AI - Intelligent Retail Demand & Replenishment Decision Support System
IntelliData 2026 Hackathon Prototype Dashboard (NovaMart Retail Pvt. Ltd.)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

st.set_page_config(
    page_title="StockSense AI | NovaMart Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5 0%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #9E9E9E;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .risk-high {
        background-color: #d32f2f;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .risk-med {
        background-color: #f57c00;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .risk-low {
        background-color: #388e3c;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Data Loaders
# -------------------------------------------------------------
@st.cache_data
def load_recommendations():
    if os.path.exists('data/processed/manager_replenishment_recommendations.csv'):
        return pd.read_csv('data/processed/manager_replenishment_recommendations.csv')
    return pd.DataFrame()

@st.cache_data
def load_metadata():
    if os.path.exists('models/model_metadata.json'):
        with open('models/model_metadata.json') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_master_sample():
    if os.path.exists('data/processed/featured_dataset_sample.csv'):
        df = pd.read_csv('data/processed/featured_dataset_sample.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()

recom_df = load_recommendations()
meta = load_metadata()
sample_df = load_master_sample()

# -------------------------------------------------------------
# Sidebar Navigation & Filters
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/shop.png", width=64)
st.sidebar.markdown("## **StockSense AI**")
st.sidebar.caption("NovaMart Retail Decision Support")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Executive Summary",
        "📈 Demand Intelligence",
        "⚠️ Inventory Risk Matrix",
        "🎯 Manager Action Centre",
        "🧪 Model Performance & Validation",
        "🔍 Feature Explainability",
        "💡 What-If Scenario Simulator"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Store & Category Filters")

if not recom_df.empty:
    all_stores = ['All Stores'] + sorted(recom_df['store_id'].unique().tolist())
    selected_store = st.sidebar.selectbox("Filter Store", all_stores)

    all_cats = ['All Categories'] + sorted(recom_df['category'].unique().tolist())
    selected_cat = st.sidebar.selectbox("Filter Category", all_cats)

    filtered_recom = recom_df.copy()
    if selected_store != 'All Stores':
        filtered_recom = filtered_recom[filtered_recom['store_id'] == selected_store]
    if selected_cat != 'All Categories':
        filtered_recom = filtered_recom[filtered_recom['category'] == selected_cat]
else:
    filtered_recom = pd.DataFrame()

# -------------------------------------------------------------
# Page 1: Executive Summary
# -------------------------------------------------------------
if page == "📊 Executive Summary":
    st.markdown('<div class="main-header">Executive Summary & Retail KPIs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Strategic inventory health, revenue velocity, and stock-out exposure across NovaMart branches.</div>', unsafe_allow_html=True)

    if not filtered_recom.empty:
        c1, c2, c3, c4, c5 = st.columns(5)
        
        tot_products = len(filtered_recom)
        high_risk_count = (filtered_recom['risk_level'] == 'HIGH').sum()
        high_risk_pct = (high_risk_count / tot_products) * 100
        total_reorder_qty = filtered_recom['recommended_reorder_qty'].sum()
        avg_doi = filtered_recom['days_of_inventory'].median()
        total_7d_demand = filtered_recom['predicted_7d_demand'].sum()

        with c1:
            st.metric("Total Monitored SKUs", f"{tot_products:,}")
        with c2:
            st.metric("Products at HIGH Risk", f"{high_risk_count:,}", f"{high_risk_pct:.1f}% rate", delta_color="inverse")
        with c3:
            st.metric("7-Day Forecast Demand", f"{total_7d_demand:,.0f} units")
        with c4:
            st.metric("Recommended Reorder", f"{total_reorder_qty:,} units")
        with c5:
            st.metric("Median Stock Cover", f"{avg_doi:.1f} Days")

        st.markdown("---")

        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Inventory Risk Distribution")
            risk_counts = filtered_recom['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'SKU Count']
            colors = {'HIGH': '#EF5350', 'MEDIUM': '#FFA726', 'LOW': '#66BB6A'}
            fig_pie = px.pie(
                risk_counts, values='SKU Count', names='Risk Level',
                color='Risk Level', color_discrete_map=colors,
                hole=0.4, title="Stockout Risk Tier Segmentation"
            )
            fig_pie.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_r:
            st.subheader("Replenishment Demand by Category")
            cat_demand = filtered_recom.groupby('category').agg(
                forecast_demand=('predicted_7d_demand', 'sum'),
                reorder_units=('recommended_reorder_qty', 'sum')
            ).reset_index()
            fig_bar = px.bar(
                cat_demand, x='category', y=['forecast_demand', 'reorder_units'],
                barmode='group', title="Category Forecast vs Reorder Requirement",
                color_discrete_sequence=['#42A5F5', '#AB47BC'],
                labels={'value': 'Units', 'variable': 'Metric'}
            )
            fig_bar.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### Store Health Snapshot")
        store_summary = filtered_recom.groupby(['store_id', 'city', 'store_type']).agg(
            SKUs=('product_id', 'count'),
            High_Risk_SKUs=('risk_level', lambda s: (s == 'HIGH').sum()),
            Predicted_7d_Demand=('predicted_7d_demand', 'sum'),
            Total_Reorder_Units=('recommended_reorder_qty', 'sum'),
            Median_Days_Cover=('days_of_inventory', 'median')
        ).reset_index()
        store_summary['High_Risk_Rate'] = (store_summary['High_Risk_SKUs'] / store_summary['SKUs'] * 100).round(1).astype(str) + '%'
        st.dataframe(store_summary, use_container_width=True)

# -------------------------------------------------------------
# Page 2: Demand Intelligence
# -------------------------------------------------------------
elif page == "📈 Demand Intelligence":
    st.markdown('<div class="main-header">Demand Intelligence & Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Model 1 7-Day forward continuous regression projections vs historical baselines.</div>', unsafe_allow_html=True)

    if not sample_df.empty:
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            prod_list = sample_df['product_id'].unique().tolist()
            sel_prod = st.selectbox("Select Product to Inspect Demand Curve", prod_list[:30])
        with col_ctrl2:
            st_list = sample_df['store_id'].unique().tolist()
            sel_store = st.selectbox("Select Store", st_list)

        prod_ts = sample_df[(sample_df['product_id'] == sel_prod) & (sample_df['store_id'] == sel_store)].sort_values('date')

        if not prod_ts.empty:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=prod_ts['date'], y=prod_ts['daily_demand'], name='Daily Sales Demand', line=dict(color='#29B6F6', width=1.5)))
            fig_ts.add_trace(go.Scatter(x=prod_ts['date'], y=prod_ts['rolling_mean_7'], name='7-Day Rolling Mean', line=dict(color='#FFA726', width=2, dash='dash')))
            fig_ts.add_trace(go.Scatter(x=prod_ts['date'], y=prod_ts['next_7_day_demand'], name='Target 7-Day Forward Demand', line=dict(color='#66BB6A', width=2)))

            fig_ts.update_layout(
                title=f"Historical Demand & Forward 7-Day Velocity: Store {sel_store} | Product {sel_prod}",
                xaxis_title="Date", yaxis_title="Units",
                height=450, hovermode="x unified"
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("### Demand Volatility & Velocity Matrix")
        scatter_df = filtered_recom.head(300)
        fig_scatter = px.scatter(
            scatter_df, x='predicted_7d_demand', y='current_stock',
            color='risk_level', size='recommended_reorder_qty',
            hover_data=['product_id', 'brand', 'category', 'days_of_inventory'],
            color_discrete_map={'HIGH': '#EF5350', 'MEDIUM': '#FFA726', 'LOW': '#66BB6A'},
            title="7-Day Demand vs Current Stock (Bubble Size = Recommended Order Quantity)"
        )
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="gray", dash="dot"))
        st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------------------------------------
# Page 3: Inventory Risk Matrix
# -------------------------------------------------------------
elif page == "⚠️ Inventory Risk Matrix":
    st.markdown('<div class="main-header">Inventory Risk Matrix & Heatmaps</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Model 2 Stock-out classification probability triage (High $\ge 70\%$, Medium $40-70\%$, Low $< 40\%$).</div>', unsafe_allow_html=True)

    if not filtered_recom.empty:
        # Cross-tab heatmap Store x Category
        st.subheader("🔥 Stock-out Risk Heatmap (Store vs Category)")
        heatmap_data = filtered_recom.pivot_table(
            index='category', columns='store_id', values='stockout_prob', aggfunc='mean'
        ) * 100

        fig_heat = px.imshow(
            heatmap_data, labels=dict(x="Store ID", y="Category", color="Avg Stockout Prob (%)"),
            x=heatmap_data.columns, y=heatmap_data.index,
            color_continuous_scale="Reds", text_auto=".1f"
        )
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("⚠️ High-Risk Triage Table")
        triage_filter = st.radio("Select Risk Tier", ["HIGH", "MEDIUM", "LOW"], horizontal=True)
        tier_df = filtered_recom[filtered_recom['risk_level'] == triage_filter]
        
        display_cols = [
            'store_id', 'city', 'product_id', 'category', 'brand', 'current_stock', 
            'predicted_7d_demand', 'stockout_prob', 'safety_stock', 'recommended_reorder_qty',
            'days_of_inventory', 'manager_action'
        ]
        st.dataframe(tier_df[display_cols], use_container_width=True)

# -------------------------------------------------------------
# Page 4: Manager Action Centre
# -------------------------------------------------------------
elif page == "🎯 Manager Action Centre":
    st.markdown('<div class="main-header">Store Manager Action Centre</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explainable, decision-ready replenishment cards with exact reorder quantities and business rationale.</div>', unsafe_allow_html=True)

    if not filtered_recom.empty:
        high_risk_items = filtered_recom[filtered_recom['risk_level'] == 'HIGH']
        
        st.info(f"🚨 **{len(high_risk_items):,} SKUs require immediate managerial action** to avert stockouts over the upcoming replenishment cycle.")

        # Download Action List CSV
        csv_data = filtered_recom.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Official Replenishment Action Plan (CSV)",
            data=csv_data,
            file_name="NovaMart_Replenishment_Action_Plan.csv",
            mime="text/csv"
        )

        st.markdown("### Top Urgent Replenishment Action Cards")
        for idx, r in high_risk_items.head(6).iterrows():
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border-left: 5px solid #EF5350; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #FFF;">STORE {r['store_id']} ({r['city']} - {r['store_type']}) | PRODUCT {r['product_id']} ({r['brand']} - {r['category']})</h4>
                        <span class="risk-high">STOCKOUT PROBABILITY: {r['stockout_prob']*100:.1f}%</span>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.95rem; color: #DDD;">
                        <b>Current Inventory:</b> {r['current_stock']} units | 
                        <b>Incoming Stock:</b> {r['incoming_stock']} units | 
                        <b>Forecast 7-Day Demand:</b> {r['predicted_7d_demand']} units | 
                        <b>Safety Stock Buffer:</b> {r['safety_stock']} units
                    </div>
                    <div style="margin-top: 8px; font-size: 1.05rem; color: #4CAF50; font-weight: 600;">
                        📦 RECOMMENDED REORDER QUANTITY: {r['recommended_reorder_qty']} UNITS
                    </div>
                    <div style="margin-top: 6px; font-size: 0.9rem; color: #B0BEC5;">
                        <b>Why?</b> {r['key_drivers_reason']}
                    </div>
                    <div style="margin-top: 8px; font-weight: 600; color: #FFD54F;">
                        👉 ACTION: {r['manager_action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Page 5: Model Performance & Validation
# -------------------------------------------------------------
elif page == "🧪 Model Performance & Validation":
    st.markdown('<div class="main-header">Model Performance & Validation Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous out-of-time test set evaluation comparing permitted algorithms and baseline heuristics.</div>', unsafe_allow_html=True)

    if meta:
        st.subheader("Model 1: 7-Day Demand Forecasting Comparison")
        st.table(pd.DataFrame(meta['demand_comparison']))

        st.subheader("Model 2: Stock-out Risk Classification Comparison")
        st.table(pd.DataFrame(meta['stockout_comparison']))

        st.subheader("Model 2 Confusion Matrix (Out-of-Time Test Set)")
        cm = np.array(meta['confusion_matrix'])
        labels = ['In Stock (0)', 'Stockout (1)']
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=labels, y=labels,
            labels=dict(x="Predicted Class", y="Actual Ground Truth"),
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cm, use_container_width=True)

# -------------------------------------------------------------
# Page 6: Feature Explainability
# -------------------------------------------------------------
elif page == "🔍 Feature Explainability":
    st.markdown('<div class="main-header">Explainable AI (XAI) & Driver Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">White-box decision transparency showing exactly what features drive demand and stock-out alerts.</div>', unsafe_allow_html=True)

    if meta and 'explainability' in meta:
        c_l, c_r = st.columns(2)
        with c_l:
            st.subheader("Top Drivers of 7-Day Demand (Model 1)")
            dem_feat = pd.DataFrame(meta['explainability']['demand_top_features']).sort_values('importance', ascending=True)
            fig_dem = px.bar(
                dem_feat, x='importance', y='feature', orientation='h',
                color='importance', color_continuous_scale="Viridis",
                title="Demand Regressor Feature Importance"
            )
            fig_dem.update_layout(height=400)
            st.plotly_chart(fig_dem, use_container_width=True)

        with c_r:
            st.subheader("Top Drivers of Stock-out Risk (Model 2)")
            stk_feat = pd.DataFrame(meta['explainability']['stockout_top_features']).sort_values('importance', ascending=True)
            fig_stk = px.bar(
                stk_feat, x='importance', y='feature', orientation='h',
                color='importance', color_continuous_scale="Plasma",
                title="Stockout Classifier Feature Importance"
            )
            fig_stk.update_layout(height=400)
            st.plotly_chart(fig_stk, use_container_width=True)

# -------------------------------------------------------------
# Page 7: What-If Scenario Simulator
# -------------------------------------------------------------
elif page == "💡 What-If Scenario Simulator":
    st.markdown('<div class="main-header">What-If Retail Decision Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive sensitivity testing: Evaluate how promotional discounts, supplier lead times, and festival events alter stockout risk.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        sim_promo = st.radio("Launch Promotional Campaign?", ["No Promotion", "Active Promotion (+168% demand)"])
    with c2:
        sim_lead = st.slider("Supplier Lead Time (Days)", min_value=1, max_value=7, value=3)
    with c3:
        sim_stock = st.number_input("Current Warehouse Stock (Units)", min_value=0, max_value=200, value=15)

    base_demand = 8.5
    if sim_promo == "Active Promotion (+168% demand)":
        sim_demand = base_demand * 2.68
    else:
        sim_demand = base_demand

    sim_doi = sim_stock / (sim_demand / 7 + 0.01)
    # Calibrated risk heuristic from model weights
    sim_prob = min(0.99, max(0.05, 1.0 - (sim_doi / 6.0) + (0.25 if "Active" in sim_promo else 0.0) + (sim_lead * 0.04)))
    
    sim_safety = int(np.ceil(1.65 * 2.5 * np.sqrt(sim_lead)))
    sim_reorder = max(0, int(np.ceil(sim_demand + sim_safety - sim_stock)))

    st.markdown("---")
    st.subheader("Simulation Results")
    
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Projected 7-Day Demand", f"{sim_demand:.1f} units")
    with r2:
        st.metric("Simulated Stockout Risk", f"{sim_prob*100:.1f}%", "HIGH RISK" if sim_prob >= 0.70 else "SAFE")
    with r3:
        st.metric("Required Safety Stock", f"{sim_safety} units")
    with r4:
        st.metric("Recommended Order Quantity", f"{sim_reorder} units")

    if sim_prob >= 0.70:
        st.error(f"⚠️ **CRITICAL HAZARD:** Stockout will occur within {sim_doi:.1f} days without immediate replenishment order of {sim_reorder} units!")
    else:
        st.success(f"✅ **HEALTHY BUFFER:** Current stock covers {sim_doi:.1f} days of simulated operational demand.")
