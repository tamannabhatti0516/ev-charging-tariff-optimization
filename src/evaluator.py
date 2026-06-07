# src/evaluator.py

"""
Evaluation metrics and plotting utilities for the EV Charging Optimization System.
Computes KPIs (Revenue Lift, Utilization, Satisfaction, MAE, RMSE) and exports results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_kpis(df_results: pd.DataFrame, base_price_per_kwh: float = 0.25) -> dict:
    """
    Computes system and operational KPIs.
    
    Args:
        df_results (pd.DataFrame): Simulation output logs.
        base_price_per_kwh (float): Price for the flat-rate baseline.
        
    Returns:
        dict: A dictionary of calculated KPI metrics.
    """
    actual_kwh = df_results["actual_kwh"]
    predicted_kwh = df_results["predicted_kwh"]
    tariff_price = df_results["tariff_price"]
    utilization = df_results["utilization"]
    satisfaction = df_results["satisfaction"]
    
    total_energy = actual_kwh.sum()
    flat_revenue = total_energy * base_price_per_kwh
    dynamic_revenue = (actual_kwh * tariff_price).sum()
    
    net_lift = dynamic_revenue - flat_revenue
    pct_lift = (net_lift / flat_revenue * 100.0) if flat_revenue > 0 else 0.0
    
    mae = (actual_kwh - predicted_kwh).abs().mean()
    rmse = np.sqrt(((actual_kwh - predicted_kwh) ** 2).mean())
    
    return {
        "total_energy_kwh": float(total_energy),
        "flat_revenue": float(flat_revenue),
        "dynamic_revenue": float(dynamic_revenue),
        "net_revenue_lift": float(net_lift),
        "pct_revenue_lift": float(pct_lift),
        "avg_utilization": float(utilization.mean()),
        "avg_satisfaction": float(satisfaction.mean()),
        "mae_demand": float(mae),
        "rmse_demand": float(rmse)
    }

def plot_simulation_results(df_results: pd.DataFrame, save_path: str):
    """
    Generates and saves a premium visualization of the simulation run.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. Demand Forecast vs Actual
    axes[0].plot(df_results.index, df_results["actual_kwh"], label="Actual Demand (kWh)", color="#2b5c8f", alpha=0.8, lw=1.5)
    axes[0].plot(df_results.index, df_results["predicted_kwh"], label="Predicted Demand (kWh)", color="#e67e22", linestyle="--", lw=1.5)
    axes[0].set_title("EV Grid Demand: Actual vs Forecasted", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_ylabel("Load (kWh)", fontsize=12)
    axes[0].legend(loc="upper right", frameon=True)
    
    # 2. Tariff Rates
    axes[1].plot(df_results.index, df_results["tariff_price"], label="Dynamic Price ($/kWh)", color="#27ae60", lw=2)
    axes[1].axhline(y=0.25, color="red", linestyle=":", label="Flat Rate ($0.25/kWh)", lw=1.5)
    axes[1].set_title("Dynamic Electricity Tariff Pricing Response", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_ylabel("Tariff Rate ($)", fontsize=12)
    axes[1].legend(loc="upper right", frameon=True)
    
    # 3. System Metrics: Utilization & Satisfaction
    axes[2].plot(df_results.index, df_results["utilization"], label="Grid Utilization", color="#8e44ad", alpha=0.8, lw=1.5)
    axes[2].plot(df_results.index, df_results["satisfaction"], label="User Satisfaction", color="#f1c40f", alpha=0.8, lw=1.5)
    axes[2].set_title("System Co-optimization: Grid Utilization vs. Customer Satisfaction", fontsize=14, fontweight="bold", pad=10)
    axes[2].set_ylabel("Metrics Score", fontsize=12)
    axes[2].set_xlabel("Simulation Hours", fontsize=12)
    axes[2].legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluator] Saved simulation report chart to {save_path}")
