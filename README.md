# EV Charging Dynamic Tariff Optimization System

This repository implements a multi-agent dynamic tariff pricing and demand prediction system for EV charging hubs. It leverages charging session data and station metadata to predict load requirements and dynamically adjust grid tariffs to co-optimize operational revenue, average user satisfaction, and grid capacity constraints.

## Project Structure

- `config/settings.py`: Directory and file path configuration.
- `dataset/`: Storage for project datasets.
- `src/data_loader.py`: In-memory data loading and JSON malformed file repair.
- `src/preprocessing.py`: Feature engineering (durations) and hourly timeseries aggregation.
- `src/models.py`: Prediction, Tariff, and Monitoring collaborative learning agents.
- `src/evaluator.py`: KPI analysis metrics and simulation plotting utilities.
- `main.py`: Main CLI script to run the end-to-end simulation.
- `requirements.txt`: Python package dependency list.

## Core Agents

1. **DemandPredictionAgent**: Learns and forecasts historical load patterns based on day-of-week and hour-of-day.
2. **TariffPricingAgent**: Dynamically computes pricing rates per kWh based on predicted vs. baseline demand.
3. **MonitoringLearningAgent**: Evaluates real-time performance, tracks metrics (utilization, user satisfaction, revenue), and automatically adjusts pricing agent sensitivity parameters.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python main.py
   ```
