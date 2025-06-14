import pandas as pd
import xml.etree.ElementTree as et
import os

def get_traffic_metrics(tripinfo_file):
    # Check if file exists
    if not os.path.exists(tripinfo_file):
        print(f"Error: {tripinfo_file} does not exist.")
        return 0, 0, 0, 0, 0  # Added normal waiting time to return

    # Check if file is empty
    if os.path.getsize(tripinfo_file) == 0:
        print(f"Error: {tripinfo_file} is empty.")
        return 0, 0, 0, 0, 0

    try:
        xtree = et.parse(tripinfo_file)
        xroot = xtree.getroot()
        rows = []
        for node in xroot:
            if node.tag == "tripinfo":
                vehicle_type = node.attrib.get("vType", "default")
                rows.append({
                    "travel_time": float(node.attrib.get("duration", 0)),
                    "waiting_time": float(node.attrib.get("waitingTime", 0)),
                    "vehicle_type": vehicle_type
                })
        df = pd.DataFrame(rows)
        avg_travel_time = df["travel_time"].mean() if not df.empty else 0
        emergency_df = df[df["vehicle_type"] == "emergency"]
        avg_emergency_waiting_time = emergency_df["waiting_time"].mean() if not emergency_df.empty else 0
        avg_emergency_travel_time = emergency_df["travel_time"].mean() if not emergency_df.empty else 0
        normal_df = df[df["vehicle_type"] != "emergency"]
        avg_normal_travel_time = normal_df["travel_time"].mean() if not normal_df.empty else 0
        avg_normal_waiting_time = normal_df["waiting_time"].mean() if not normal_df.empty else 0  # New metric
        return avg_travel_time, avg_emergency_waiting_time, avg_emergency_travel_time, avg_normal_travel_time, avg_normal_waiting_time
    except et.ParseError as e:
        print(f"XML Parse Error in {tripinfo_file}: {e}")
        return 0, 0, 0, 0, 0
    except Exception as e:
        print(f"Unexpected error parsing {tripinfo_file}: {e}")
        return 0, 0, 0, 0, 0