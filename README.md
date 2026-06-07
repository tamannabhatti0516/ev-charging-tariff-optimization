# Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

**Society of Business ? Open Project 2026**

A self-improving agentic AI pricing engine that autonomously predicts 
EV charging demand, recommends dynamic tariffs in real time, and 
continuously learns from outcomes to maximize revenue, balance grid 
demand, and optimize EV infrastructure efficiency.

---

## Project Structure
```
projects_antig/
??? config/
?   ??? settings.py           # Path configuration
├── dataset/
│   ├── stations.csv          # Station metadata
│   ├── kpi_summary.csv       # KPI outputs from evaluation
│   └── simulation_results.csv # Simulation run outputs
??? notebooks/
?   ??? main_analysis.ipynb   # Full analysis and simulation pipeline
?   ??? eda_plots.png         # EDA visualizations
?   ??? simulation_report.png # Simulation output chart
??? src/
?   ??? data_loader.py        # Loads stations and session data
?   ??? preprocessing.py      # Cleans and aggregates hourly data
?   ??? models.py             # All three AI agents
?   ??? evaluator.py          # KPI computation and visualization
??? .gitignore
??? main.py                   # Main simulation entry point
??? README.md
??? requirements.txt
```

---

## Dataset

| Property | Value |
|----------|-------|
| Total Sessions | 14,299 |
| Unique Stations | 54 |
| Unique Sites | 2 (Caltech & JPL) |
| Date Range | April ? November 2018 |
| Avg kWh per Session | 8.98 kWh |
| Median kWh per Session | 7.47 kWh |
| Total Energy in Dataset | 128,411.65 kWh |

**Source**: ACN-Data ? https://ev.caltech.edu/dataset.html

**Secondary**: UrbanEV (ST-EVCDP) ? https://github.com/IntelligentSystemsLab/ST-EVCDP  
(informed threshold selection for off-peak and surge pricing)

---

## Setup Instructions

1. Clone the repository:
```
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd projects_antig
```

2. Install dependencies:
```
   pip install -r requirements.txt
```

3. Download ACN dataset from https://ev.caltech.edu/dataset.html  
   and place in `dataset/` folder as `acndata_sessions.json`

4. Place `stations.csv` in the `dataset/` folder

5. Open and run `notebooks/main_analysis.ipynb`

---

## Agentic AI Pipeline

### 1. Demand Prediction Agent
- Lookup table conditioned on day-of-week ? hour-of-day
- Trained on 4,463 hourly samples (April – November 2018)
- MAE: 17.59 kWh | RMSE: 27.47 kWh | R?: 0.293

### 2. Tariff Pricing Agent
- Surge pricing when utilization exceeds 80%
- Discount signals when utilization falls below 30%
- Net Revenue Lift: +$230.47 (+6.89% over flat-rate USD baseline)
- INR Revenue Gain vs ?15/kWh baseline: +47.86%

### 3. Monitoring & Learning Agent
- Evaluates revenue, utilization, satisfaction per hour
- Continuously calibrates pricing sensitivity parameters
- 96.3% user satisfaction maintained

---

## Key Results

| Metric | Value |
|--------|-------|
| Total Energy Delivered | 13,388.59 kWh |
| Flat Rate Revenue ($0.25/kWh) | $3,347.15 |
| Dynamic Tariff Revenue | $3,577.62 |
| Net Revenue Lift (USD) | +$230.47 (+6.89%) |
| Flat Revenue (?15/kWh baseline) | ?2,00,828.83 |
| Dynamic Revenue (converted @ ?83/USD) | ?2,96,942.08 |
| Revenue Gain vs ?15 baseline | +47.86% |
| Average Grid Utilization | 37.5% |
| User Satisfaction Score | 96.3% |
| Demand Forecast MAE | 17.59 kWh |
| Demand Forecast RMSE | 27.47 kWh |
| R? Score | 0.293 |
| Off-Peak Hours Identified | 411 hrs |
| Evaluation Window | 752 hours |
| Pricing Efficiency (Dynamic) | $0.2672/kWh |
| Pricing Efficiency (Flat) | $0.2500/kWh |
| Efficiency Gain | +$0.0172/kWh |

---

## Evaluation Metrics

### Demand Prediction Agent
- **RMSE**: 27.47 ? penalizes large errors in predicted load
- **MAE**: 17.59 ? average absolute error across time slots
- **R? Score**: 0.293 ? variance explained by temporal patterns

### Tariff Pricing Agent
- **Revenue Gain (USD)**: +6.89% over flat-rate baseline
- **Revenue Gain (INR)**: +47.86% vs ?15/kWh baseline (converted at ?83/USD)
- **Charger Utilization**: 37.5% average
- **Off-Peak Uplift**: 411 hours identified for discount pricing

### Monitoring & Learning Agent
- **User Satisfaction**: 96.3% proxy score
- **Pricing Efficiency**: $0.2672/kWh dynamic vs $0.2500/kWh flat baseline
- **Efficiency Gain**: $0.0172/kWh ? tracks whether the feedback loop is improving revenue decisions over time

---

## Revenue Comparison: USD vs INR Baseline

| Metric | USD | INR (@ ?83/USD) |
|--------|-----|-----------------|
| Flat Rate Revenue | $3,347.15 | ?2,00,828.83 |
| Dynamic Tariff Revenue | $3,577.62 | ?2,96,942.08 |
| Revenue Gain | +$230.47 (+6.89%) | +?96,113.25 (+47.86%) |

> **Note**: Dynamic tariffs were computed in USD ($0.25/kWh base rate) and converted to INR at 1 USD = ?83 for comparison against the ?15/kWh Indian market baseline.

---

## Assumptions & Limitations

- Satisfaction is a proxy metric from tariff deviation ? not direct user feedback
- R?=0.293 ? rule-based model underfits high-variance peaks
- Revenue reported in USD; INR ?15/kWh comparison included for Indian market context
- Utilization uses actual/predicted proxy ? no per-charger availability data
- No causal claims ? all findings are correlational
- UrbanEV informed thresholds but not directly integrated into pipeline

---

## Requirements

See `requirements.txt`. Key dependencies:
- Python 3.13
- pandas, numpy, matplotlib, seaborn, scikit-learn
