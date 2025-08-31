# alerts.py
import json
import time

while True:
    with open("vessel_data.json") as f:
        vessels = json.load(f)
    
    for v in vessels:
        if v["risk"] == "High":
            print(f"[ALERT] {v['name']} is suspicious at ({v['lat']:.2f}, {v['lon']:.2f})")

    time.sleep(2)
