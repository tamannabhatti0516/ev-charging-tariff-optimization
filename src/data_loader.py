# src/data_loader.py

"""
Data Loader module for EV charging station metadata and session transactions.
Includes logic to repair and load the malformed ACN-Data JSON dataset.
"""

import os
import json
import pandas as pd
from config.settings import STATIONS_CSV_PATH, ACNDATA_JSON_PATH

def load_stations_data() -> pd.DataFrame:
    """
    Loads the EV charging stations dataset from the CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame containing EV station details.
    """
    if not os.path.exists(STATIONS_CSV_PATH):
        raise FileNotFoundError(f"Stations CSV file not found at: {STATIONS_CSV_PATH}")
    
    df = pd.read_csv(STATIONS_CSV_PATH)
    print(f"[DataLoader] Loaded {len(df)} stations from CSV.")
    return df

def load_sessions_data() -> pd.DataFrame:
    """
    Loads charging session transactions from the JSON file.
    Resolves trailing commas and missing brackets (]} ) automatically.
    
    Returns:
        pd.DataFrame: A DataFrame containing charging sessions.
    """
    if not os.path.exists(ACNDATA_JSON_PATH):
        raise FileNotFoundError(f"ACN-Data JSON file not found at: {ACNDATA_JSON_PATH}")
    
    print("[DataLoader] Reading and repairing ACN-Data JSON file...")
    with open(ACNDATA_JSON_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    # Repair missing closing elements if they are absent
    if not content.endswith("]}"):
        if content.endswith(","):
            content = content[:-1]
        content += "\n]}"
        
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse ACN-Data JSON even after formatting: {e}")
        
    if "_items" not in data:
        raise KeyError("JSON missing required '_items' root array key.")
        
    df = pd.DataFrame(data["_items"])
    
    # Process and clean datetime fields
    date_cols = ["connectionTime", "disconnectTime", "doneChargingTime"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            
    print(f"[DataLoader] Loaded {len(df)} charging sessions from JSON.")
    return df
