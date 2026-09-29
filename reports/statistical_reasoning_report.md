# IntelliData 2026 - Statistical Reasoning & Hypothesis Testing Report

## Executive Summary
To provide empirical statistical rigor behind NovaMart Retail's inventory management decisions, we evaluated three core business hypotheses using parametric and non-parametric tests on the full master dataset (1,546,858 daily store-product observations).

---

## Analysis 1: Promotion Lift Significance
* **Business Question:** Do promotions significantly increase product sales demand?
* **Null Hypothesis ($H_0$):** Mean daily units sold on promotion is equal to mean daily units sold without promotion ($\mu_{\text{promo}} = \mu_{\text{non-promo}}$).
* **Alternative Hypothesis ($H_1$):** Mean daily units sold on promotion is significantly greater ($\mu_{\text{promo}} > \mu_{\text{non-promo}}$).
* **Test Employed:** Welch's Two-Sample $t$-test (unequal variances) & Mann-Whitney $U$ test.
* **Empirical Findings:**
  * Mean Daily Demand (On Promotion): **3.35 units**
  * Mean Daily Demand (No Promotion): **1.25 units**
  * Observed Promotion Lift: **+168.02%**
  * Welch's $t$-statistic = **112.0785**, $p$-value = **0.0000e+00**
  * Mann-Whitney $U$ statistic = **89866954949.5000**, $p$-value = **0.0000e+00**
* **Statistical Decision:** Since $p < 0.001$, we **reject $H_0$**.
* **Business Takeaway:** Promotions generate a statistically significant demand surge (+168.0% volume). Store managers **must proactively increase safety stock and trigger replenishment before promotional launches** to prevent severe promotional stock-outs.

---

## Analysis 2: Demand Differentiation Across Store Types
* **Business Question:** Does average daily demand differ significantly across store formats (Hypermarket, Supermarket, Express)?
* **Null Hypothesis ($H_0$):** Mean daily demand is identical across all store types ($\mu_{\text{Hypermarket}} = \mu_{\text{Supermarket}} = \mu_{\text{Express}}$).
* **Alternative Hypothesis ($H_1$):** At least one store format exhibits significantly different demand levels.
* **Test Employed:** One-Way Analysis of Variance (ANOVA) & Kruskal-Wallis non-parametric test.
* **Empirical Findings:**

| store_type   |   count |     mean |     std |
|:-------------|--------:|---------:|--------:|
| Express      |  515629 | 0.815191 | 1.89696 |
| Hypermarket  |  515636 | 1.93392  | 4.07915 |
| Supermarket  |  515593 | 1.31115  | 3.13633 |

  * One-way ANOVA $F$-statistic = **16162.3933**, $p$-value = **0.0000e+00**
  * Kruskal-Wallis $H$-statistic = **46383.9177**, $p$-value = **0.0000e+00**
* **Statistical Decision:** Since $p < 0.001$, we **reject $H_0$**.
* **Business Takeaway:** Store format is a strong predictor of velocity. Hypermarkets operate at vastly higher throughput than Express convenience formats. Replenishment policy **cannot be one-size-fits-all**; reorder thresholds and safety stocks must be parameterized by store format.

---

## Analysis 3: Association Between Stock-out Frequency and Promotions
* **Business Question:** Is the incidence of stock-outs statistically associated with promotional status?
* **Null Hypothesis ($H_0$):** Stock-out events and promotion status are independent ($P(\text{Stockout} \mid \text{Promo}) = P(\text{Stockout} \mid \text{No Promo})$).
* **Alternative Hypothesis ($H_1$):** Stock-out events are significantly associated with promotional campaigns.
* **Test Employed:** Pearson's Chi-Square ($\chi^2$) Test of Independence with $2 \times 2$ contingency matrix.
* **Empirical Findings:**

|              |   In Stock |   Stockout |
|:-------------|-----------:|-----------:|
| No Promotion |    1348017 |     121912 |
| On Promotion |      67824 |       9105 |

  * Stockout Rate on Promotion: **11.84%**
  * Stockout Rate without Promotion: **8.29%**
  * $\chi^2$ Statistic = **1182.4656**, degrees of freedom = 1, $p$-value = **3.9450e-259**
* **Statistical Decision:** Since $p < 0.001$, we **reject $H_0$**.
* **Business Takeaway:** There is a highly significant statistical association between active promotions and stockout incidence. Promotions consistently outpace static replenishment rules. A dynamic decision support model with promotional lead-time buffers is mandatory.
