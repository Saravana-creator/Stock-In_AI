# StockSense AI — Intelligent Demand Forecasting & Stock-Out Decision Support System

> **Event:** IntelliData 2026 – Data Science Hackathon  
> **Institution:** Sri Eshwar College of Engineering, Coimbatore  
> **Department:** Computer Science and Engineering  
> **Challenge Line:** *“Every empty shelf is lost revenue. Can Data Science predict it before it happens?”*  
> **Client:** NovaMart Retail Pvt. Ltd. (Multi-City Supermarket Chain)

---

## 📌 Executive Summary
NovaMart Retail operates supermarkets across six major cities selling groceries, beverages, dairy, snacks, and household essentials. Traditional replenishment was based on gut feeling and static rules, leading to **costly stock-outs that lose revenue** and **overstocking that traps working capital**.

**StockSense AI** delivers an end-to-end, decision-ready intelligence solution that:
1. **Forecasts 7-Day Demand** for every Store $\times$ Product combination using leakage-free lag and rolling features ($R^2 = 0.920$, MAE $= 2.81$ units).
2. **Predicts Stock-out Risk** as a calibrated probability and classifies urgency into **High ($\ge 70\%$)**, **Medium ($40-70\%$)**, and **Low ($< 40\%$)** tiers (Recall $= 77.9\%$, ROC-AUC $= 0.817$).
3. **Calculates Optimal Replenishment** via Safety Stock and Reorder Quantity formulas:
   $$\text{Safety Stock} = Z \times \sigma_d \times \sqrt{\text{Lead Days}}$$
   $$\text{Reorder Quantity} = \max(0, \text{Forecast Demand} + \text{Safety Stock} - \text{Current Stock} - \text{Incoming Stock})$$
4. **Delivers Manager-Friendly Action Cards** explaining *why* a product is at risk and *what action* to take today.
5. **Provides an Interactive Prototype Dashboard** with Executive Summary, Risk Heatmaps, Action Centre, and What-If Simulator.

---

## 🏗️ Repository & Solution Architecture

```
Stockin-AI/
├── data/
│   ├── raw/ or cleaned/             # Original provided CSV tables
│   └── processed/
│       ├── master_analytics_data.parquet     # Master grain: 1 Date x 1 Store x 1 Product
│       ├── featured_dataset.parquet          # Leakage-free features & forward targets
│       └── manager_replenishment_recommendations.csv # Live actionable decision cards
├── src/
│   ├── data_prep.py                 # Resolves 6 intentional data quality traps
│   ├── statistical_analysis.py      # Executes 3 mandatory hypothesis tests
│   ├── feature_engineering.py       # Computes lags, rolling stats, inventory velocity
│   ├── train_models.py              # Out-of-time validation of Model 1 & Model 2
│   └── recommendation_engine.py     # Safety stock, reorder quantities, managerial explanations
├── models/
│   ├── demand_forecast_model.json   # Model 1: Portable native XGBoost Regressor
│   ├── demand_forecast_model.joblib
│   ├── stockout_risk_model.json     # Model 2: Portable native XGBoost Classifier
│   ├── stockout_risk_model.joblib
│   └── model_metadata.json          # Metrics, confusion matrix, and feature importances
├── dashboard/
│   └── app.py                       # Streamlit interactive management decision prototype
├── notebooks/
│   └── IntelliData_2026_StockSense_Pipeline.ipynb # End-to-end evaluation notebook
├── reports/
│   ├── data_quality_report.md       # Audit of intentional traps & remediations
│   ├── statistical_reasoning_report.md # Formally documented hypothesis tests
│   └── model_comparison_report.md   # Model comparison tables & business metric rationale
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Master documentation and pitch guide
```

---

## 🔍 Round 1: Data Understanding, Cleaning & Statistical Reasoning

### 1. Remediation of the 6 Intentional Data Quality Traps
| Trap Issue | Detected Anomaly | Remediation Action | Business Justification |
| :--- | :--- | :--- | :--- |
| **Category Inconsistencies** | `Foods`, `FOODS`, `foods`, `Household`, `HOUSEHOLD` | Standardized via `.str.strip().str.title()` | Prevents split categories and erroneous Pareto analysis. |
| **Duplicate Transactions** | **1,256 identical transaction IDs** | Dropped duplicates | Eliminates artificial inflation of revenue and sales volume. |
| **Impossible Quantity** | **1,209 records** with quantity $\le 0$ ($q = -4$) | Filtered out non-positive quantities | Negative sales distort actual customer demand. |
| **Formatting / Keys** | Mixed casing (`s01`), whitespace, `"unknown"` date | Uppercased & stripped IDs; standardized to `YYYY-MM-DD` | Ensures 100% referential join integrity across all tables. |
| **Inventory Mismatches** | **24,046 rows** where $\text{closing} \neq \text{opening} + \text{received} - \text{sold}$ | Flagged via `inventory_mismatch_flag`; reconciled with non-negative lower bound | Preserves audit trail for stock shrinkage while maintaining physical balance. |
| **Sparse History / Zero Days** | Days with zero sales transactions | Imputed $0$ sales on inventory calendar backbone | Critical for intermittent retail demand modeling. |

* **Master Table Grain:** **ONE ROW = ONE DATE × ONE STORE × ONE PRODUCT** (1,546,858 records across 6 stores and 300 products).

### 2. Statistical Hypothesis Testing
We executed three formal statistical analyses on the full master dataset:
1. **Promotion Lift Significance:**
   * $H_0: \mu_{\text{promo}} = \mu_{\text{non-promo}}$ vs. $H_1: \mu_{\text{promo}} > \mu_{\text{non-promo}}$
   * **Result:** Promo mean = **3.35 units** vs Non-promo mean = **1.25 units** (**+168.02% Lift**).
   * Welch's $t = 112.08$, $p < 10^{-16}$ $\rightarrow$ **Reject $H_0$**. Promotions drive massive demand surges.
2. **Demand Across Store Formats:**
   * $H_0: \mu_{\text{Hypermarket}} = \mu_{\text{Supermarket}} = \mu_{\text{Express}}$
   * **Result:** Hypermarket = **1.93 units/day**, Supermarket = **1.31 units/day**, Express = **0.82 units/day**.
   * One-way ANOVA $F = 16,162.39$, $p < 10^{-16}$ $\rightarrow$ **Reject $H_0$**. Replenishment must be store-type parameterized.
3. **Stock-Out Association with Promotions:**
   * $H_0$: Stock-out incidence is independent of promotion status.
   * **Result:** Stockout rate on promotion = **11.84%** vs Non-promotion = **8.29%**.
   * Pearson's $\chi^2 = 1,182.47$, $p = 3.9 \times 10^{-259}$ $\rightarrow$ **Reject $H_0$**. Promotions significantly outpace static replenishment.

---

## ⚙️ Round 2: Feature Engineering & Machine Learning

### 1. Leakage-Free Feature Design
All features are computed using only data available up to decision day $T$:
* **Demand Lags:** `lag_1`, `lag_7`, `lag_14` (historical daily sales).
* **Rolling Statistics:** `rolling_mean_7`, `rolling_mean_14`, `rolling_std_7`, `rolling_sum_7`.
* **Inventory Coverage:** `current_stock`, `days_of_inventory` ($\frac{\text{current\_stock}}{\text{rolling\_mean\_7} + 0.01}$), `reorder_gap` ($\text{stock} - \text{reorder\_lvl}$), `is_below_reorder`.
* **Commercial & External:** `mrp`, `cost_price`, `profit_margin`, `temp_c`, `rain_mm`, `festival_flag`, `local_event_flag`, `day_of_week`, `is_weekend`, `month`.

### 2. Time-Aware Split (No Random Shuffling)
* **Training Set:** 2014-02-10 to 2015-12-31 (1,218,953 rows)
* **Validation Set:** 2016-01-01 to 2016-05-31 (268,818 rows)
* **Out-of-Time Test Set:** 2016-06-01 to 2016-06-16 (21,767 rows)

### 3. Model 1 — 7-Day Demand Forecasting (Regression)
| Model | MAE | RMSE | $R^2$ Score | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Previous 7-Day Sum)** | 3.58 | 7.27 | 0.8468 | 59.0% |
| **Ridge Regression** | 2.98 | 5.70 | 0.9058 | 48.6% |
| **XGBoost Regressor (Selected)** | **2.81** | **5.25** | **0.9200** | **44.4%** |

* **Selected Model Rationale:** XGBoost reduced MAE by over 30% against the baseline and explains **92% of demand variance** ($R^2 = 0.920$) across seasonal and promotional shifts.

### 4. Model 2 — Stock-out Risk Classification
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Reorder Point Rule** | 0.654 | 0.531 | 0.336 | 0.412 | 0.500 |
| **Random Forest Classifier** | 0.710 | 0.573 | 0.763 | 0.655 | 0.802 |
| **XGBoost Classifier (Selected)** | **0.720** | **0.583** | **0.779** | **0.667** | **0.817** |

* **Selected Model Rationale:** In stockout prevention, **Recall is paramount**—missing a stockout loses customer loyalty and revenue, whereas a false alarm costs only minor holding interest. The selected XGBoost Classifier achieves **77.9% Recall** and **0.817 ROC-AUC**.

---

## 🎯 Round 3: Decision Intelligence Layer & Explainability

### 1. Operational Recommendation Engine
* **Safety Stock:** $\text{Safety Stock} = \lceil 1.65 \times \sigma_d \times \sqrt{\text{Lead Days}} \rceil$ (95% service level)
* **Recommended Stock:** $\lceil \text{Forecast 7d Demand} + \text{Safety Stock} \rceil$
* **Reorder Quantity:** $\max(0, \text{Recommended Stock} - \text{Current Stock} - \text{Incoming Stock})$
* **Risk Triage:**
  * **HIGH Risk:** Probability $\ge 0.70$ $\rightarrow$ Raise PO today.
  * **MEDIUM Risk:** $0.40 \le P < 0.70$ $\rightarrow$ Schedule PO within lead time.
  * **LOW Risk:** $P < 0.40$ $\rightarrow$ Inventory healthy.

### 2. Sample Manager Action Card
```
=============================================================
STORE S05 (Madurai - Hypermarket) | PRODUCT P328 (Suvai Home - Household)
Predicted 7-Day Demand: 63.6 units | Safety Stock: 32 units
Current Stock: 2 units | Incoming Stock: 0 units
Stock-out Probability: 96.7% | Risk Tier: HIGH
RECOMMENDED REORDER: 94 units
Why: Critical inventory depth (0.2 days remaining); Upcoming weekend footfall acceleration; Breached safety reorder point (2 <= 18)
MANAGER ACTION: URGENT: Raise Purchase Order for 94 units today.
=============================================================
```

---

## 🖥️ Interactive Prototype Dashboard

The prototype is built with **Streamlit** and **Plotly** and includes 7 modular sections:
1. **Executive Summary:** High-level KPIs (Total SKUs, % High Risk, 7-Day Demand, Total Reorder Units, Median Stock Cover).
2. **Demand Intelligence:** Interactive actual vs. forecast curves, category velocity comparison, demand vs. stock scatter plots.
3. **Inventory Risk Matrix:** Store $\times$ Category Stockout Heatmap and risk tier triage tables.
4. **Manager Action Centre:** Interactive recommendation cards and **one-click CSV download** of the official purchase order plan.
5. **Model Validation Suite:** Out-of-time test set comparisons and confusion matrix.
6. **Feature Explainability:** Horizontal feature importance charts showing primary drivers.
7. **What-If Scenario Simulator:** Real-time simulation of promotion campaigns, supplier delays, and warehouse stock levels.

---

## 🚀 Setup & Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (From Scratch)
```bash
# 1. Clean data, fix traps, generate master table & data quality report
python src/data_prep.py

# 2. Run the 3 statistical hypothesis tests
python src/statistical_analysis.py

# 3. Engineer leakage-free features and forward targets
python src/feature_engineering.py

# 4. Train Model 1 (Regression) & Model 2 (Classification)
python src/train_models.py

# 5. Generate live managerial recommendations
python src/recommendation_engine.py
```

### 3. Launch the Prototype Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🏆 Final Pitch & Evaluation Summary
* **What did we discover?** Static reorder rules fail during promotional spikes (+168% demand surge) and across varying store formats (Hypermarket volume is 2.4x Express).
* **What can we predict?** Next 7-day demand with $R^2 = 0.920$ and stockout events with 77.9% recall and 0.817 ROC-AUC.
* **Why can the result be trusted?** Trained strictly with chronological out-of-time validation without feature leakage; portable native models saved in JSON.
* **What is the business impact?** Automates the daily triage of 1,752 store-product combinations, prioritizing the 448 high-risk items and prescribing exact purchase order quantities before shelves go empty.
