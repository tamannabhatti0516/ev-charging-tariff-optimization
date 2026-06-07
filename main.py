# main.py

"""
Main execution script for the EV Charging Dynamic Tariff Optimization System.
"""

import os
import pandas as pd
import numpy as np

from src.data_loader import load_stations_data, load_sessions_data
from src.preprocessing import aggregate_hourly
from src.models import DemandPredictionAgent, TariffPricingAgent, MonitoringLearningAgent
from src.evaluator import compute_kpis, plot_simulation_results

def run_simulation():
    # 1. Load Data
    stations_df = load_stations_data()
    sessions_df = load_sessions_data()
    
    # 2. Preprocess & Aggregate Hourly
    df_hourly = aggregate_hourly(sessions_df)
    
    # 3. Split into Train & Evaluation sets (evaluation = last 752 hours)
    eval_steps = 752
    if len(df_hourly) <= eval_steps:
        print("[Main] Warning: Dataset too small. Running evaluation on whole dataset.")
        df_train = df_hourly
        df_eval = df_hourly
    else:
        df_train = df_hourly.iloc[:-eval_steps].copy()
        df_eval = df_hourly.iloc[-eval_steps:].copy()
        
    print(f"[Main] Split data: Train={len(df_train)} hours, Eval={len(df_eval)} hours.")
    
    # 4. Initialize and Train Prediction Agent
    prediction_agent = DemandPredictionAgent()
    prediction_agent.fit(df_train)
    
    # 5. Initialize Pricing and Monitoring Agents
    pricing_agent = TariffPricingAgent(base_price_per_kwh=0.25, sensitivity=0.05)
    monitoring_agent = MonitoringLearningAgent(target_occupancy=0.65)
    
    # 6. Run Simulation Loop
    simulation_logs = []
    
    for idx, row in df_eval.iterrows():
        dayofweek = int(row["dayofweek"])
        hour = int(row["hour"])
        actual_kwh = float(row["kwh_demand"])
        
        # Predict Demand
        pred = prediction_agent.predict(dayofweek, hour)
        predicted_kwh = pred["predicted_kwh"]
        
        # Determine Dynamic Tariff Price
        tariff_price = pricing_agent.calculate_tariff(predicted_kwh, prediction_agent.global_mean_kwh)
        
        # Observe Performance and Calibrate Tariff Sensitivity
        log_entry = monitoring_agent.observe_and_adapt(
            pricing_agent=pricing_agent,
            actual_kwh=actual_kwh,
            predicted_kwh=predicted_kwh,
            tariff_price=tariff_price,
            max_capacity_kwh=150.0 # Standard hub physical limit
        )
        
        simulation_logs.append(log_entry)
        
    # Convert logs to DataFrame
    df_results = pd.DataFrame(simulation_logs)
    
    # 7. Compute KPIs
    kpis = compute_kpis(df_results, base_price_per_kwh=0.25)
    
    # Print Summary Report
    print("\n" + "="*50)
    print("      SIMULATION RESULTS COMPARISON SUMMARY")
    print("="*50)
    print(f"Evaluation Hours            : {len(df_results)}")
    print(f"Total Energy Delivered      : {kpis['total_energy_kwh']:.2f} kWh")
    print(f"Static Revenue ($0.25/kWh)  : ${kpis['flat_revenue']:.2f}")
    print(f"Dynamic Tariff Revenue      : ${kpis['dynamic_revenue']:.2f}")
    print(f"Net Revenue Lift            : ${kpis['net_revenue_lift']:.2f} ({kpis['pct_revenue_lift']:.2f}%)")
    print(f"Average Grid Utilization    : {kpis['avg_utilization']:.2f}")
    print(f"Average User Satisfaction   : {kpis['avg_satisfaction']:.2f}")
    print(f"Demand Forecasting MAE      : {kpis['mae_demand']:.2f} kWh")
    print(f"Demand Forecasting RMSE     : {kpis['rmse_demand']:.2f} kWh")
    print("="*50 + "\n")
    
    # 8. Export Plots
    os.makedirs("C:/projects_antig/notebooks", exist_ok=True)
    plot_save_path = "C:/projects_antig/notebooks/simulation_report.png"
    plot_simulation_results(df_results, plot_save_path)
    
    # Also save to the brain directory
    brain_plot_path = r"C:\Users\hp\.gemini\antigravity-ide\brain\1ee500e6-b137-4909-87d9-1f14aa7fc865\simulation_report.png"
    try:
        import shutil
        shutil.copy(plot_save_path, brain_plot_path)
        print(f"[Main] Copied report chart to brain artifacts directory.")
    except Exception as e:
        print(f"[Main] Error copying chart to brain directory: {e}")

if __name__ == "__main__":
    run_simulation()
