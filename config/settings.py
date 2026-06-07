import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIONS_CSV_PATH = os.path.join(BASE_DIR, "dataset", "stations.csv")
ACNDATA_JSON_PATH = os.path.join(BASE_DIR, "dataset", "acndata_sessions.json")
