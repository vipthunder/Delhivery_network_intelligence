# Delhivery Network Intelligence
## Graph-Based Logistics Optimization | Machine Learning | Consulting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NetworkX-Graph_Analysis-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/XGBoost-ML_Model-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>Identifying delivery bottlenecks and improving ETA predictions for India's largest logistics provider</b>
</p>

---

## Table of Contents

- [Background](#background)
- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Strategy Memo](#strategy-memo)

---

## Background

Delhivery is India's largest fully-integrated logistics provider, operating a vast network of facilities, intercity routes, and last-mile delivery across every major state. At the core of its operations is a hub-and-spoke model — shipments travel from a source facility through one or more intermediate hubs before reaching the destination.

To estimate delivery times, Delhivery uses **OSRM** (Open Source Routing Machine), a standard routing engine that assumes clean traffic and shortest paths. But real-world logistics is far messier.

---

## Problem Statement

> *Can a graph-based model — one that treats the logistics network as a connected graph of facilities and corridors — produce more accurate ETAs and identify which corridors and hubs are systematically causing delays?*

**The core issue:**

| Problem | Impact |
|---------|--------|
| OSRM underestimates delivery time by **2.21×** on average | SLAs missed, customers unhappy |
| **94%** of corridors chronically delayed beyond 1.2× threshold | Systematic network-wide failure |
| No way to identify which hubs cause the most damage | No targeted intervention possible |
| Route-type decisions made without graph position context | Suboptimal FTL vs Carting choices |

---

## Approach

Rather than treating each delivery in isolation, this project models Delhivery's entire logistics network as a **directed weighted graph**:

```
Nodes  = 1,657 facilities (warehouses, hubs, delivery centers)
Edges  = 2,783 corridors  (routes between facilities)
Weight = Median delay ratio per corridor (actual ÷ OSRM predicted time)
```

This graph structure reveals what tabular models cannot see: **delays cascade through connected hubs**. A bottleneck at one hub propagates across the entire downstream network.

---

## Key Results

### ETA Prediction

| Metric | Baseline XGBoost | Graph-Enhanced XGBoost | Improvement |
|--------|:----------------:|:---------------------:|:-----------:|
| MAE | 55.85 min | 41.81 min | **↓ 25% better** |
| Within 15% Accuracy | 44.44% | 54.99% | **↑ +10.55 pp** |

### Route-Type Decision Framework

| Model | Accuracy | Use Case |
|-------|:--------:|----------|
| Decision Tree | 68.21% | Field operations (interpretable rules) |
| XGBoost | 69.90% | Automated backend routing |

### Network Health Summary

| Metric | Value |
|--------|-------|
| Trip segments analyzed | 142,502 |
| Unique facilities (nodes) | 1,657 |
| Unique corridors (edges) | 2,783 |
| Average OSRM underestimation | 2.21× |
| Chronic delay corridors | 94% of network |
| Trips through top 5 bottleneck hubs | 47,307 |
| Estimated unnecessary delay | 75,000+ hours/year |

---

## Project Structure

```
delhivery-network-intelligence/
│
├── delhivery_analysis.ipynb      # Main analysis notebook
│   ├── Section 1: EDA & Data Cleaning
│   ├── Section 2: Graph Construction
│   ├── Section 3: Bottleneck Analysis
│   ├── Section 4: ETA Prediction Model
│   └── Section 5: FTL vs Carting Framework
│
├── Delhivery_Strategy_Memo.docx  # Operations strategy memo
│
├── bottleneck_network.html       # Interactive network graph
├── model_comparison.png          # Baseline vs graph model
├── decision_tree.png             # FTL decision tree rules
├── hourly_delay.png              # Delay by hour of day
├── ftl_carting_tradeoff.png      # Distance band analysis
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## Methodology

### 1. Data Pipeline & Graph Construction

**Dataset:** 144,867 raw trip segments → 142,502 clean rows after removing 2,365 corrupted records (1.63%).

**Key decisions:**

| Decision | Choice | Justification |
|----------|--------|---------------|
| Edge weight | Median segment_factor | Right-skewed distribution — median more robust than mean |
| Graph type | Directed (DiGraph) | Mumbai→Delhi ≠ Delhi→Mumbai in real logistics |
| Stratification | By route_type and departure_hour | Same corridor behaves differently by time and vehicle |

### 2. Bottleneck Detection

Three graph metrics combined into a composite risk score:

```
Risk Score = Betweenness Centrality (40%)
           + Average Delay Ratio    (40%)
           + Total Degree           (20%)
```

| Metric | What it measures |
|--------|-----------------|
| Betweenness Centrality | % of all network paths passing through this hub |
| Average Delay Ratio | How delayed this hub's corridors are vs OSRM |
| Total Degree | Number of direct connections |

### 3. Graph-Enhanced ETA Model

Graph features added over baseline:

| Feature | What it captures |
|---------|-----------------|
| `source_betweenness` | Congestion risk at departure hub |
| `dest_betweenness` | Congestion risk at arrival hub |
| `corridor_delay` | Historical median delay on this specific route |

> **Note on node2vec:** Structural graph features were used as a computationally efficient and interpretable alternative to node2vec embeddings — capturing equivalent network position information while remaining explainable to stakeholders.

### 4. FTL vs Carting Framework

| Distance | FTL Delay | Carting Delay | Winner | Gap |
|----------|:---------:|:-------------:|:------:|:---:|
| 0–50 km | 2.19× | 2.79× | FTL | 0.60× |
| 50–200 km | 2.17× | 2.47× | FTL | 0.30× |
| 200–500 km | 1.95× | 2.22× | FTL | 0.27× |
| 500+ km | 1.96× | N/A | FTL only | — |

**Key insight:** Corridor delay history accounts for **92.4% of feature importance** — structural corridor characteristics matter more than vehicle type.

---

## Key Findings

### Top 5 Bottleneck Hubs

| Rank | Hub | Risk Score | Betweenness | Avg Delay | Connections | Action |
|:----:|-----|:----------:|:-----------:|:---------:|:-----------:|--------|
| 1 | IND000000ACB | 0.634 | 23.3% | 1.60× | 94 | Capacity Upgrade |
| 2 | IND712311AAA | 0.522 | 8.1% | 2.20× | 46 | Process Optimization |
| 3 | IND421302AAG | 0.393 | 5.3% | 2.02× | 58 | Route-Type Shift |
| 4 | IND110037AAM | 0.362 | 4.7% | 2.06× | 45 | Process Optimization |
| 5 | IND562132AAA | 0.355 | 15.3% | 1.54× | 71 | Parallel Route |

### If Top 3 Hubs Are Upgraded

| Scenario | Trips Impacted | Est. Delay Reduction | Est. SLA Improvement |
|----------|:--------------:|:--------------------:|:--------------------:|
| IND000000ACB only | 23,279 | 15–20% | 18–24% fewer breaches |
| IND712311AAA only | 2,591 | 25–30% | 8–10% fewer breaches |
| IND421302AAG only | 9,012 | 10–15% | 12–15% fewer breaches |
| **All 3 combined** | **34,882** | **20–25% avg** | **25–35% fewer breaches** |

### Bonus Insights
- **Peak delay hour:** 11 AM (3.11×) — urban rush hour congestion
- **Non-cutoff trips:** 3.74× delay vs 1.98× for standard trips
- **Network density:** 0.001 — extremely sparse, high hub dependency
- **IND000000ACB clustering:** 0.037 — 97% of connected facilities have no alternative path

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data Processing | pandas, numpy |
| Graph Analysis | NetworkX |
| Machine Learning | XGBoost, scikit-learn |
| Visualization | pyvis, matplotlib, seaborn |
| Environment | Google Colab / Jupyter Notebook |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/delhivery-network-intelligence
cd delhivery-network-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset
# Place delivery_data.csv in root directory
# (dataset not included — proprietary)

# 4. Run the notebook
jupyter notebook delhivery_analysis.ipynb
```

> **Google Colab:** Upload `delhivery_analysis.ipynb` and `delivery_data.csv` directly for zero-setup execution.

---

## Strategy Memo

A complete **[Network Operations Strategy Memo](Delhivery_Strategy_Memo.docx)** was prepared for Delhivery's Head of Network Operations covering:

- Executive summary with network health scorecard
- Top 5 bottleneck hubs with SLA breach contribution
- Three corridor-specific interventions with timelines
- Revenue impact estimates for upgrading top 3 hubs
- 7-point prioritized action plan

---

## Skills Demonstrated

```
Graph Theory       →  Directed graphs, betweenness centrality,
                      clustering coefficients, network density
Machine Learning   →  XGBoost, Decision Trees, feature engineering,
                      MAE optimization, data leakage prevention
Data Engineering   →  Pipeline design, missing value handling,
                      stratified aggregation, reproducible code
Business Analysis  →  SLA quantification, revenue impact estimation,
                      executive memo writing, KPI translation
```

---

## License

This project is licensed under the MIT License.

---

<p align="center">
<i>Built as part of a data science consulting engagement analyzing Delhivery's logistics network.</i>
<br><br>
<i>"The best model is not the most accurate one — it's the most useful one in context."</i>
</p>
