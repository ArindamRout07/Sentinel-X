# simulate_vessels.py
import random
import time
import json

# Sample vessels
vessels = [
    {"name": "Vessel-A", "imo": "100001", "lat": 10.0, "lon": 70.0},
    {"name": "Vessel-B", "imo": "100002", "lat": 12.0, "lon": 72.0},
    {"name": "Vessel-C", "imo": "100003", "lat": 15.0, "lon": 74.0},
    {"name": "Vessel-D", "imo": "100004", "lat": 8.0, "lon": 71.0},  # Suspicious
]

# Shipping lane boundaries (simplified)
normal_lat_range = (9.0, 16.0)
normal_lon_range = (70.0, 75.0)

# Historical data storage
historical = []

while True:
    vessel_data = []
    timestamp = time.time()
    for v in vessels:
        # Random movement
        v["lat"] += random.uniform(-0.05, 0.05)
        v["lon"] += random.uniform(-0.05, 0.05)

        # Anomaly detection
        if not (normal_lat_range[0] <= v["lat"] <= normal_lat_range[1] and
                normal_lon_range[0] <= v["lon"] <= normal_lon_range[1]):
            risk = "High"
        else:
            risk = "Normal"

        vessel_data.append({
            "name": v["name"],
            "imo": v["imo"],
            "lat": v["lat"],
            "lon": v["lon"],
            "risk": risk,
            "timestamp": timestamp
        })

    # Update current data
    with open("vessel_data.json", "w") as f:
        json.dump(vessel_data, f, indent=2)

    # Update historical data
    historical.extend(vessel_data)
    if len(historical) > 500:  # limit size
        historical = historical[-500:]
    
    time.sleep(2)
