# IntelliData 2026 - Data Quality & Audit Report

## Executive Summary
An exhaustive audit of raw datasets across retail transactions, store inventories, product catalogs, store profiles, and external weather/event factors identified **all 6 intentional data quality traps** embedded in the challenge. This report documents the exact issue, incidence count, remediation action taken, and business justification.

---

## Data Quality Traps Remediation Matrix

| Trap Issue | Detected Anomaly | Count / Incidence | Action Taken | Business Justification |
| :--- | :--- | :--- | :--- | :--- |
| **1. Category Inconsistency** | Casing divergence: `Foods`, `FOODS`, `foods`, `Household`, `HOUSEHOLD` | 28 inconsistent records across 300 products | Standardized using `.str.strip().str.title()` | Prevents categorical fragmentation in tree models and Pareto revenue skew. |
| **2. Duplicate Transactions** | Identical `transaction_id` repeated across identical time/store stamps | **1,256 duplicate rows** | Dropped duplicate rows using `.drop_duplicates(subset=['transaction_id'])` | Prevents artificial inflation of sales volume, revenue, and customer counts. |
| **3. Impossible Quantity** | Non-positive sales quantities ($\le 0$, e.g., $q = -4$) | **1,209 records** | Filtered out rows with `quantity <= 0` | Negative quantities represent unhandled returns or system errors; cannot represent demand. |
| **4. Date & ID Formatting** | Trailing/leading whitespace, lowercase IDs (`s01`, `S01 `), string `"unknown"` | 1 invalid date (`"unknown"`), 24 store variants, 1,186 product variants | Parsed mixed date formats to `YYYY-MM-DD`, stripped and uppercased all IDs | Harmonized primary keys to achieve 100% referential integrity across all tables. |
| **5. Inventory Conservation Mismatch** | Arithmetic violation: $\text{closing} \neq \text{opening} + \text{received} - \text{sold}$ | **24,046 records** (1.55%) | Flagged via `inventory_mismatch_flag`, calculated discrepancy, reconciled with non-negative lower bound | Preserves audit trail for stock shrinkage/spoilage while maintaining physical stock balance. |
| **6. Sparse / Missing Sales Days** | Days with zero transactions for active store $\times$ product pairs | Natural retail intermittency | Imputed `trans_units_sold = 0` and `daily_demand = sold` from daily inventory backbone | Essential to prevent zero-truncation bias in intermittent demand forecasting. |

---

## Final Master Table Specifications
- **Master Grain**: **ONE ROW = ONE DATE × ONE STORE × ONE PRODUCT**
- **Total Master Records**: **1,546,858 observations**
- **Date Range**: 2014-01-29 to 2016-07-04 (874 full operating days)
- **Stores**: 6 stores (Chennai, Coimbatore, Madurai, Salem, Tiruchirappalli, Vellore)
- **Products**: 300 products across Foods and Household categories
- **Integrity**: 0 missing values in core target, inventory balance, or pricing fields.
