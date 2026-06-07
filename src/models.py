# src/models.py

"""
Agent classes for the EV Tariff Optimization System:
1. DemandPredictionAgent: Forecasts EV grid load based on temporal historical patterns.
2. TariffPricingAgent: Computes dynamic pricing rates.
3. MonitoringLearningAgent: Evaluates pipeline performance and calibrates parameters.
"""

import pandas as pd
import numpy as np

class DemandPredictionAgent:
    """
    Predicts EV charging demand (kWh and session counts) for given timestamps.
    Uses historical averages conditioned on day of week and hour of day.
    """
    def __init__(self):
        self.lookup_table = None
        self.global_mean_sessions = 0.0
        self.global_mean_kwh = 0.0

    def fit(self, df_hourly: pd.DataFrame):
        """
        Trains the demand prediction lookup index from historical hourly aggregations.
        
        Args:
            df_hourly (pd.DataFrame): Aggregated hourly timeseries with 'dayofweek',
                                      'hour', 'session_count', and 'kwh_demand'.
        """
        if df_hourly.empty:
            print("[DemandPredictionAgent] Warning: Fit received empty DataFrame.")
            return
            
        # Create lookup mapping for dayofweek and hour to mean count/kwh
        grouped = df_hourly.groupby(["dayofweek", "hour"]).agg(
            mean_sessions=("session_count", "mean"),
            mean_kwh=("kwh_demand", "mean")
        )
        
        self.lookup_table = grouped.to_dict()
        self.global_mean_sessions = df_hourly["session_count"].mean()
        self.global_mean_kwh = df_hourly["kwh_demand"].mean()
        print(f"[DemandPredictionAgent] Trained on {len(df_hourly)} hourly samples. Global avg kwh: {self.global_mean_kwh:.2f}")

    def predict(self, dayofweek: int, hour: int) -> dict:
        """
        Predicts charging volume and load for a specific hour/day combination.
        
        Args:
            dayofweek (int): 0 (Monday) to 6 (Sunday).
            hour (int): 0 to 23.
            
        Returns:
            dict: Containing 'predicted_sessions' and 'predicted_kwh'.
        """
        if self.lookup_table is None:
            # Fallback if fit hasn't been called
            return {"predicted_sessions": 1.0, "predicted_kwh": 10.0}
            
        sessions = self.lookup_table["mean_sessions"].get((dayofweek, hour), self.global_mean_sessions)
        kwh = self.lookup_table["mean_kwh"].get((dayofweek, hour), self.global_mean_kwh)
        
        return {
            "predicted_sessions": float(sessions),
            "predicted_kwh": float(kwh)
        }


class TariffPricingAgent:
    """
    Determines optimal energy tariff pricing ($ per kWh) dynamically.
    Increases prices during peak demand and reduces them during low utilization.
    """
    def __init__(self, base_price_per_kwh: float = 0.25, sensitivity: float = 0.05):
        """
        Args:
            base_price_per_kwh (float): Baseline price rate.
            sensitivity (float): Price variation sensitivity response rate.
        """
        self.base_price_per_kwh = base_price_per_kwh
        self.sensitivity = sensitivity

    def calculate_tariff(self, predicted_kwh: float, avg_kwh: float) -> float:
        """
        Computes dynamic electricity tariff based on predicted vs. baseline demand.
        
        Args:
            predicted_kwh (float): Predicted demand in kWh.
            avg_kwh (float): System historical baseline average kWh.
            
        Returns:
            float: Dynamic price per kWh.
        """
        denom = avg_kwh if avg_kwh > 0 else 10.0
        demand_ratio = predicted_kwh / denom
        
        # Calculate adjustment delta
        price_delta = (demand_ratio - 1.0) * self.sensitivity * self.base_price_per_kwh
        tariff = self.base_price_per_kwh + price_delta
        
        # Keep prices within realistic operational bounds (e.g. $0.10 to $0.75 per kWh)
        min_price = max(0.10, self.base_price_per_kwh * 0.4)
        max_price = self.base_price_per_kwh * 3.0
        return float(np.clip(tariff, min_price, max_price))


class MonitoringLearningAgent:
    """
    Monitors pricing performance, computes operational KPI metrics,
    and dynamically adapts the pricing agent's parameters to align with system goals.
    """
    def __init__(self, target_occupancy: float = 0.65):
        """
        Args:
            target_occupancy (float): Ideal average grid occupancy rate (0.0 to 1.0).
        """
        self.target_occupancy = target_occupancy
        self.history = []

    def observe_and_adapt(self, pricing_agent: TariffPricingAgent, 
                          actual_kwh: float, predicted_kwh: float, 
                          tariff_price: float, max_capacity_kwh: float = 150.0) -> dict:
        """
        Records the operational performance, computes KPIs, and tunes pricing agent sensitivity.
        
        Args:
            pricing_agent (TariffPricingAgent): Reference to pricing agent to update.
            actual_kwh (float): Realized electricity demand.
            predicted_kwh (float): Forecasted demand.
            tariff_price (float): Price charged per kWh.
            max_capacity_kwh (float): Maximum physical capacity threshold of charging hubs.
            
        Returns:
            dict: Operational metrics for the period.
        """
        utilization = min(actual_kwh / max_capacity_kwh, 1.0) if max_capacity_kwh > 0 else 0.0
        revenue = actual_kwh * tariff_price
        
        # Customer satisfaction modeled negatively by price increases, positively by availability
        price_ratio = tariff_price / pricing_agent.base_price_per_kwh if pricing_agent.base_price_per_kwh > 0 else 1.0
        satisfaction = max(0.0, 1.0 - 0.4 * (price_ratio - 1.0) - 0.3 * utilization)
        
        log_entry = {
            "actual_kwh": actual_kwh,
            "predicted_kwh": predicted_kwh,
            "tariff_price": tariff_price,
            "utilization": utilization,
            "revenue": revenue,
            "satisfaction": satisfaction
        }
        self.history.append(log_entry)
        
        # Adaptive learning loop adjustment:
        # If utilization is higher than target, increase sensitivity to suppress excess load.
        # If lower, adjust sensitivity downwards to encourage utilization.
        utilization_gap = utilization - self.target_occupancy
        pricing_agent.sensitivity = float(np.clip(pricing_agent.sensitivity + utilization_gap * 0.02, 0.01, 0.40))
        
        return log_entry

    def get_summary_metrics(self) -> dict:
        """
        Aggregates operational feedback logs.
        
        Returns:
            dict: Summary metrics.
        """
        if not self.history:
            return {}
        df_hist = pd.DataFrame(self.history)
        return {
            "total_revenue": float(df_hist["revenue"].sum()),
            "avg_tariff": float(df_hist["tariff_price"].mean()),
            "avg_utilization": float(df_hist["utilization"].mean()),
            "avg_satisfaction": float(df_hist["satisfaction"].mean()),
            "mae_demand": float((df_hist["actual_kwh"] - df_hist["predicted_kwh"]).abs().mean())
        }
