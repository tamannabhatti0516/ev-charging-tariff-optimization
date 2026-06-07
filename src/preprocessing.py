# src/preprocessing.py

"""
Preprocessing module for EV charging session data.
Includes function to engineer durations and aggregate sessions into hourly timeseries.
"""

import pandas as pd
import numpy as np

def preprocess_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data types, parses datetime columns, and engineers duration metrics.
    """
    df = df.copy()
    
    # Ensure datetimes are parsed
    date_cols = ["connectionTime", "disconnectTime", "doneChargingTime"]
    for col in date_cols:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            
    # Drop rows without critical timestamps
    df = df.dropna(subset=["connectionTime", "disconnectTime"])
    
    # Calculate connection duration (hours)
    df["connection_duration"] = (df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 3600.0
    
    # Calculate charging duration (hours)
    has_done = "doneChargingTime" in df.columns
    if has_done:
        df["charging_duration"] = (df["doneChargingTime"].fillna(df["disconnectTime"]) - df["connectionTime"]).dt.total_seconds() / 3600.0
    else:
        df["charging_duration"] = df["connection_duration"]
        
    # Ensure non-negative and clip outliers
    df["connection_duration"] = np.clip(df["connection_duration"], 0.0, 168.0)
    df["charging_duration"] = np.clip(df["charging_duration"], 0.0, df["connection_duration"])
    
    # Calculate idle duration (hours)
    df["idle_duration"] = df["connection_duration"] - df["charging_duration"]
    
    # Clean kWhDelivered
    if "kWhDelivered" in df.columns:
        df["kWhDelivered"] = pd.to_numeric(df["kWhDelivered"], errors="coerce").fillna(0.0)
        df["kWhDelivered"] = np.clip(df["kWhDelivered"], 0.0, 100.0)
        
    return df

def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactions into a continuous hourly timeseries with:
    - session_count
    - kwh_demand
    - dayofweek
    - hour
    """
    df_clean = preprocess_sessions(df)
    
    # Floor connection times to nearest hour
    df_clean["hour_dt"] = df_clean["connectionTime"].dt.floor("h")
    
    # Group by hourly period
    grouped = df_clean.groupby("hour_dt").agg(
        session_count=("sessionID", "count"),
        kwh_demand=("kWhDelivered", "sum")
    ).reset_index()
    
    if grouped.empty:
        return pd.DataFrame(columns=["hour_dt", "session_count", "kwh_demand", "dayofweek", "hour"])
        
    # Reindex to cover a complete range of hours
    min_time = grouped["hour_dt"].min()
    max_time = grouped["hour_dt"].max()
    full_range = pd.date_range(start=min_time, end=max_time, freq="h")
    
    grouped = grouped.set_index("hour_dt").reindex(full_range, fill_value=0).reset_index()
    grouped.rename(columns={"index": "hour_dt"}, inplace=True)
    
    # Extract features for prediction agents
    grouped["dayofweek"] = grouped["hour_dt"].dt.dayofweek
    grouped["hour"] = grouped["hour_dt"].dt.hour
    
    print(f"[Preprocessing] Aggregated sessions into {len(grouped)} hourly steps.")
    return grouped
