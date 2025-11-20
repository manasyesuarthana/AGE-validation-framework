import json
import numpy as np
import os
import sys

# Ensure we can import from the same directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from load_data import load_real_epilepsy_dataset

ENERGY_PER_SAMPLE_mJ = 0.05
ENERGY_PER_BYTE_mJ = 0.01 
BYTES_PER_SAMPLE = 4 

def linear_adaptive_sampler(sequence, threshold):
    """
    Collects a sample if the change exceeds the threshold.
    Returns: (indices, values) for reconstruction.
    """
    collected_indices = [0]
    collected_values = [sequence[0]]
    
    last_val = sequence[0]
    
    for i in range(1, len(sequence)):
        curr_val = sequence[i]
        if abs(curr_val - last_val) > threshold:
            collected_indices.append(i)
            collected_values.append(curr_val)
            last_val = curr_val
            
    return np.array(collected_indices), np.array(collected_values)

def calculate_energy(sampled_count, message_bytes):
    return (sampled_count * ENERGY_PER_SAMPLE_mJ) + \
           (message_bytes * ENERGY_PER_BYTE_mJ)

def calculate_mae(original_seq, indices, values):
    """
    Reconstructs signal via linear interpolation and calculates MAE.
    """
    # Create the full time axis
    full_indices = np.arange(len(original_seq))
    
    # Linear Interpolation
    reconstructed_seq = np.interp(full_indices, indices, values)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(original_seq - reconstructed_seq))
    return mae

def determine_dynamic_threshold(X_data, target_rate=0.5):
    all_diffs = []
    for seq in X_data:
        diffs = np.abs(np.diff(seq))
        all_diffs.extend(diffs)
    return np.percentile(all_diffs, (1.0 - target_rate) * 100)

def run_sensor():
    print("[SENSOR] Starting simulation...")
    
    # 1. Load Data
    # Handle both local dev path and Docker path
    if os.path.exists("/app/sim/data"):
        data_path = "/app/sim/data"
    else:
        data_path = os.path.join("sim", "data")
        
    X, y = load_real_epilepsy_dataset(data_path)
    
    # 2. Setup Policy
    threshold = determine_dynamic_threshold(X, target_rate=0.5)
    print(f"[SENSOR] Threshold: {threshold:.4f}")

    # Outputs to share with attacker
    metadata_log = []
    
    # Internal metrics
    total_energy = 0.0
    total_mae = 0.0

    for i, seq in enumerate(X):
        # A. Run Sampling
        indices, samples = linear_adaptive_sampler(seq, threshold)
        sampled_count = len(samples)
        
        # --- DEFENSE LOGIC (This is where you edit for Phase 4) ---
        # Mode: Insecure Baseline
        message_bytes = sampled_count * BYTES_PER_SAMPLE
        
        # B. Calculate Metrics
        seq_energy = calculate_energy(sampled_count, message_bytes)
        seq_mae = calculate_mae(seq, indices, samples)
        
        total_energy += seq_energy
        total_mae += seq_mae
        
        # C. Log "Leaked" Metadata
        # The attacker sees: Message Size (bytes)
        # The attacker needs: True Label (for training/validation)
        metadata_log.append({
            "message_bytes": int(message_bytes),
            "label": int(y[i]) 
        })

    # 3. Aggregate Results
    avg_energy = total_energy / len(X)
    avg_mae = total_mae / len(X)
    
    print(f"[SENSOR] Avg Energy: {avg_energy:.4f} mJ")
    print(f"[SENSOR] Avg MAE:    {avg_mae:.4f}")

    # 4. Write Output
    # In Docker: writes to shared volume /metrics
    # Local: writes to build/metrics
    if os.path.exists("/metrics"):
        output_dir = "/metrics"
    else:
        output_dir = "build/metrics"
        os.makedirs(output_dir, exist_ok=True)

    # Payload contains the log for the attacker AND the sensor's performance metrics
    payload = {
        "sensor_metrics": {
            "mae": avg_mae,
            "energy": avg_energy
        },
        "traffic_log": metadata_log
    }

    out_file = os.path.join(output_dir, "sensor_output.json")
    with open(out_file, "w") as f:
        json.dump(payload, f)
        
    print(f"[SENSOR] Data written to {out_file}")

if __name__ == "__main__":
    run_sensor()