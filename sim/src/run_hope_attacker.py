import json
import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def run_hope_evaluation():
    print("[HOPE_EVAL] Starting Security Analysis...")
    
    # 1. Load Simulation Data
    if os.path.exists("/data_in/hope_simulation.json"): 
        input_path = "/data_in/hope_simulation.json"
    else: 
        input_path = "build/metrics/hope_simulation.json"

    # added error handling just in case file is not found. 
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)
    
    with open(input_path, "r") as f:
        data = json.load(f)
        
    hope_log = data["hope_log"]
    
    # 2. Calculate Lifetime Extension
    base_life = data.get("baseline_death_time", 0.0)
    hope_life = data.get("hope_death_time", 0.0)

    # avoid division by zero
    if base_life == 0:
        base_life = 0.001

    extension_factor = hope_life / base_life
    
    print(f"\n--- METRIC 1: LIFETIME ---")
    print(f"Baseline Death: {base_life:.1f}s")
    print(f"HOPE Death:     {hope_life:.1f}s")
    print(f"Extension Factor: {extension_factor:.2f}x (Target: >1.5x)")
    
    # 3. Analyze Privacy Gradient (The Sigmoid Effect)
    # we want to see how 'Inter-Arrival Time' (IAT) changes over battery life
    
    timestamps = [entry["t"] for entry in hope_log]
    batteries  = [entry["E"] for entry in hope_log]
    
    # Calculate IAT
    iats = np.diff(timestamps)
    # Align battery to IAT (remove last point)
    batt_aligned = batteries[:-1]
    
    # ATTACK SIMULATION:
    # can an attacker predict the Battery Level based on the IAT?
    # if they can, they know exactly when we are vulnerable.
    # a sigmoid function should make this regression difficult (Non-linear).
    
    X = np.array(iats).reshape(-1, 1)
    y = np.array(batt_aligned)
    
    reg = LinearRegression()
    reg.fit(X, y)
    preds = reg.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    print(f"\n--- METRIC 2: ATTACK RESISTANCE ---")
    print(f"Attacker RMSE (Predicting Battery): {rmse:.2f}")
    print("(Higher RMSE is better - means attacker cannot easily reverse-engineer state)")
    
    # 4. Calculate Time to Privacy Failure (TTPF)
    # we define failure as when IAT variance becomes high (Economy Mode)
    
    # Simple heuristic: when did we switch to mostly 'REAL_BURST' or 'SKIP'?
    # we count the portion of packets that were 'SECURE' or 'TWILIGHT_DUMMY'
    secure_packets = sum(1 for e in hope_log if e["type"] in ["SECURE", "TWILIGHT_DUMMY"])
    total_packets = len(hope_log)
    
    # handled case where total_packets is 0
    if total_packets > 0:
        privacy_retention = (secure_packets/ total_packets) * 100
    else:
        privacy_retention = 0.0
    
    print(f"\n--- METRIC 3: PRIVACY RETENTION ---")
    print(f"Privacy Retention Rate: {privacy_retention:.1f}%")
    
    # 5. Save Final Metrics
    metrics = {
        "lifetime_extension": extension_factor,
        "attacker_rmse": rmse,
        "privacy_retention": privacy_retention
    }
    
    if os.path.exists("/data_out"): output_dir = "/data_out"
    else: output_dir = "build/metrics"
    
    with open(os.path.join(output_dir, "hope_metrics.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    run_hope_evaluation()
