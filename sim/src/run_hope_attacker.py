import json
import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def run_hope_evaluation():
    print("[HOPE_EVAL] Starting Security Analysis...")
    
    # 1. Load Simulation Data
    if os.path.exists("/data_in/hope_simulation.json"): input_path = "/data_in/hope_simulation.json"
    else: input_path = "build/metrics/hope_simulation.json"
    
    with open(input_path, "r") as f:
        data = json.load(f)
        
    hope_log = data["hope_log"]
    
    # 2. Calculate Lifetime Extension
    base_life = data["baseline_death"]
    hope_life = data["hope_death"]
    extension_factor = hope_life / base_life
    
    print(f"\n--- METRIC 1: LIFETIME ---")
    print(f"Baseline Death: {base_life:.1f}s")
    print(f"HOPE Death:     {hope_life:.1f}s")
    print(f"Extension Factor: {extension_factor:.2f}x (Target: >2.0x)")
    
    # 3. Analyze Privacy Gradient (The Sigmoid Effect)
    # We want to see how 'Inter-Arrival Time' (IAT) changes over battery life
    
    timestamps = [entry["t"] for entry in hope_log]
    batteries  = [entry["E"] for entry in hope_log]
    
    # Calculate IAT
    iats = np.diff(timestamps)
    # Align battery to IAT (remove last point)
    batt_aligned = batteries[:-1]
    
    # ATTACK SIMULATION:
    # Can an attacker predict the Battery Level based on the IAT?
    # If they can, they know exactly when we are vulnerable.
    # A Sigmoid function should make this regression difficult (Non-linear).
    
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
    # We define failure as when IAT variance becomes high (Economy Mode)
    
    # Simple heuristic: When did we switch to mostly 'REAL_BURST' or 'SKIP'?
    # We count the portion of packets that were 'SECURE' or 'TWILIGHT_DUMMY'
    secure_packets = sum(1 for e in hope_log if e["type"] in ["SECURE", "TWILIGHT_DUMMY"])
    total_packets = len(hope_log)
    privacy_retention = (secure_packets / total_packets) * 100
    
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
