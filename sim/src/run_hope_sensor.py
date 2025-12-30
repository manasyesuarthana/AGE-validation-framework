import numpy as np
import json
import os
import sys

# Standard Imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from load_data import load_real_epilepsy_dataset
from run_sensor import determine_dynamic_threshold

# --- CONSTANTS ---
E_MAX = 5000.0
C_IDLE = 0.01
C_TX = 5.0
DT = 1.0 / 16.0
PERIODIC_INTERVAL = 1.0

# HOPE Logic
E_BASE_THRESH = 1500.0
E_MID = 1000.0
SIGMOID_K = 0.005
ALPHA = 200.0

BUFFER_CAPACITY = 100

class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        
    def push(self, val):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(val)
        
    def is_full(self):
        return len(self.buffer) >= self.capacity
        
    def flush(self):
        self.buffer = []

def sigmoid_probability(energy):
    exponent = -SIGMOID_K * (energy - E_MID)
    exponent = np.clip(exponent, -20, 20)
    return 1.0 / (1.0 + np.exp(exponent))

def get_volatility(history):
    if len(history) < 2: return 0.0
    return np.var(history)

def detect_scenario(energy, volatility, p_dummy):
    """Helper to classify current state into A, B, C, D for reporting."""
    e_thresh = E_BASE_THRESH - (ALPHA * volatility)
    
    # Scenario A: Healthy Sleeper (High Energy, Low Vol)
    if energy > e_thresh and volatility < 1.0:
        return "A (Secure/Sleep)"
    
    # Scenario B: Charged Seizure (High Energy, High Vol)
    if energy > e_thresh and volatility >= 1.0:
        return "B (Secure/Risk)"
        
    # Scenario C: Twilight (Medium Energy, P_dummy active)
    if energy <= e_thresh and p_dummy > 0.1:
        return "C (Twilight)"
        
    # Scenario D: Critical (Low Energy)
    if energy < E_MID and p_dummy <= 0.1:
        return "D (Economy/Critical)"
        
    return "Transition"

def run_hope_simulation():
    print("[HOPE] Starting Longitudinal Simulation...")
    
    # 1. Load Data
    data_path = os.path.join("sim", "data")
    if os.path.exists("/app/sim/data"): data_path = "/app/sim/data"
    X, y = load_real_epilepsy_dataset(data_path)
    
    # Stitch Data
    full_stream = np.concatenate(X)
    full_labels = np.repeat(y, [len(x) for x in X])
    static_thresh = determine_dynamic_threshold(X, target_rate=0.5)
    
    # 2. Setup Sensors
    sensors = {
        "BASELINE": {"E": E_MAX, "alive": True, "log": []},
        "HOPE":     {"E": E_MAX, "alive": True, "log": [], "buffer": RingBuffer(BUFFER_CAPACITY)}
    }
    
    last_val = full_stream[0]
    recent_history = []
    
    current_time = 0.0
    time_since_periodic = 0.0
    
    # State tracking for reporting changes
    last_scenario = ""

    print(f"[HOPE] Total ticks to simulate: {len(full_stream)}")

    for i, val in enumerate(full_stream):
        label = int(full_labels[i])
        current_time += DT
        time_since_periodic += DT
        
        # Idle Drain
        for s in sensors.values():
            if s["alive"]:
                s["E"] -= C_IDLE
                if s["E"] <= 0: s["alive"] = False

        if not any(s["alive"] for s in sensors.values()): break

        # Data Plane
        recent_history.append(val)
        if len(recent_history) > 50: recent_history.pop(0)
        volatility = get_volatility(recent_history)
        
        if abs(val - last_val) > static_thresh:
            last_val = val
            if sensors["HOPE"]["alive"]: sensors["HOPE"]["buffer"].push(val)

        # --- HOPE LOGIC ---
        if sensors["HOPE"]["alive"]:
            s_hope = sensors["HOPE"]
            curr_thresh = E_BASE_THRESH - (ALPHA * volatility)
            p_dummy = sigmoid_probability(s_hope["E"])
            
            # SCENARIO REPORTING (Visualizing the System)
            # We print only when the scenario changes significantly to avoid log spam
            current_scenario = detect_scenario(s_hope["E"], volatility, p_dummy)
            if current_scenario != last_scenario and (i % 100 == 0):
                print(f"[Time: {current_time:.1f}s | Batt: {s_hope['E']:.0f}J] Switched to Scenario: {current_scenario}")
                last_scenario = current_scenario

            # Transmission Logic
            should_tx = False
            mode = "SKIP"
            
            # 1. Economy Trigger
            if s_hope["buffer"].is_full():
                should_tx = True
                mode = "REAL_BURST"
            
            # 2. Periodic Trigger
            elif time_since_periodic >= PERIODIC_INTERVAL:
                if s_hope["E"] > curr_thresh:
                    should_tx = True
                    mode = "SECURE"
                else:
                    # Twilight Logic
                    if not s_hope["buffer"].buffer == []:
                        should_tx = True
                        mode = "DATA_OPP"
                    elif np.random.random() < p_dummy:
                        should_tx = True
                        mode = "TWILIGHT_DUMMY"
                
            if should_tx:
                s_hope["E"] -= C_TX
                s_hope["buffer"].flush()
                s_hope["log"].append({"t": current_time, "type": mode, "E": s_hope["E"], "label": label})

        # Baseline Logic
        if sensors["BASELINE"]["alive"] and time_since_periodic >= PERIODIC_INTERVAL:
             sensors["BASELINE"]["E"] -= C_TX
             sensors["BASELINE"]["log"].append({"t": current_time, "type": "TX"})

        if time_since_periodic >= PERIODIC_INTERVAL:
            time_since_periodic = 0.0

    # Save Results
    results = {
        "baseline_death": sensors["BASELINE"].get("E", 0), # Simplified for checking
        "baseline_log_count": len(sensors["BASELINE"]["log"]),
        "hope_log": sensors["HOPE"]["log"],
        "final_time": current_time
    }
    
    # Calculate death times roughly
    results["baseline_death_time"] = results["baseline_log_count"] * PERIODIC_INTERVAL
    results["hope_death_time"] = current_time # HOPE usually survives the whole stream or dies late
    
    if os.path.exists("/metrics"): output_dir = "/metrics"
    else: output_dir = "build/metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "hope_simulation.json"), "w") as f:
        json.dump(results, f)
        
    print(f"[HOPE] Simulation Done. Log saved.")

if __name__ == "__main__":
    run_hope_simulation()
