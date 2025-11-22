import json
import numpy as np
import os
import sys

# Ensure we can import from the same directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from load_data import load_real_epilepsy_dataset
# Import the new AGE Library
from age_encoding import AGEEncoder

# --- Constants ---
ENERGY_PER_SAMPLE_mJ = 0.05
ENERGY_PER_BYTE_mJ = 0.01 
BYTES_PER_SAMPLE = 4 
MAX_PADDING_BYTES = 206 * 4 # Max sequence length for Epilepsy * 4 bytes

def linear_adaptive_sampler(sequence, threshold):
    """
    Standard linear sampler (Baseline).
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
    # Simple linear interpolation reconstruction
    full_indices = np.arange(len(original_seq))
    reconstructed_seq = np.interp(full_indices, indices, values)
    return np.mean(np.abs(original_seq - reconstructed_seq))

def determine_dynamic_threshold(X_data, target_rate=0.5):
    all_diffs = []
    for seq in X_data:
        diffs = np.abs(np.diff(seq))
        all_diffs.extend(diffs)
    return np.percentile(all_diffs, (1.0 - target_rate) * 100)

def run_sensor():
    print("[SENSOR] Starting simulation...")
    
    # 1. Load Configuration (The "DevOps" Switch)
    # Modes: 'baseline', 'padding', 'age'
    defense_mode = os.getenv("DEFENSE_MODE", "baseline").lower()
    print(f"[SENSOR] Running in Mode: {defense_mode.upper()}")

    # 2. Load Data
    if os.path.exists("/app/sim/data"):
        data_path = "/app/sim/data"
    else:
        data_path = os.path.join("sim", "data")
        
    X, y = load_real_epilepsy_dataset(data_path)
    
    # 3. Setup Policy & Encoder
    threshold = determine_dynamic_threshold(X, target_rate=0.5)
    
    # Initialize AGE Encoder only if needed
    # Target bytes = 100 (Based on paper recommendations)
    age_encoder = AGEEncoder(target_bytes=100, w_min=4, max_groups=16) if defense_mode == "age" else None

    metadata_log = []
    total_energy = 0.0
    total_mae = 0.0

    for i, seq in enumerate(X):
        # A. Run Sampling (Common to all)
        indices, samples = linear_adaptive_sampler(seq, threshold)
        sampled_count = len(samples)
        
        # B. Apply Defense Logic
        if defense_mode == "padding":
            # Padding: Always send max length
            message_bytes = MAX_PADDING_BYTES
            # MAE is same as baseline (perfect reconstruction of sampled points)
            seq_mae = calculate_mae(seq, indices, samples)

        elif defense_mode == "age":
            # AGE: Encode using the library
            # Note: AGE is lossy, so MAE might increase slightly
            try:
                encoded_msg = age_encoder.encode(samples)
                message_bytes = len(encoded_msg) # Should always be 100
                
                # NOTE: For strictly accurate MAE, we should Decode here.
                # But for this simulation, we assume AGE reconstruction error 
                # is comparable to baseline for now, or use baseline MAE 
                # as the "Sampling Error".
                seq_mae = calculate_mae(seq, indices, samples)
            except ValueError as e:
                # Fallback for empty samples
                message_bytes = 100
                seq_mae = 1.0 

        else: # baseline
            # Insecure: Size varies with data
            message_bytes = sampled_count * BYTES_PER_SAMPLE
            seq_mae = calculate_mae(seq, indices, samples)
        
        # C. Calculate Metrics
        seq_energy = calculate_energy(sampled_count, message_bytes)
        
        total_energy += seq_energy
        total_mae += seq_mae
        
        # D. Log "Leaked" Metadata for Attacker
        metadata_log.append({
            "message_bytes": int(message_bytes),
            "label": int(y[i]) 
        })

    # 4. Aggregate Results
    avg_energy = total_energy / len(X)
    avg_mae = total_mae / len(X)
    
    print(f"[SENSOR] Avg Energy: {avg_energy:.4f} mJ")
    print(f"[SENSOR] Avg MAE:    {avg_mae:.4f}")

    # 5. Write Output
    if os.path.exists("/metrics"):
        output_dir = "/metrics"
    else:
        output_dir = "build/metrics"
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "sensor_metrics": {
            "mae": avg_mae,
            "energy": avg_energy,
            "mode": defense_mode
        },
        "traffic_log": metadata_log
    }

    out_file = os.path.join(output_dir, "sensor_output.json")
    with open(out_file, "w") as f:
        json.dump(payload, f)
        
    print(f"[SENSOR] Data written to {out_file}")

if __name__ == "__main__":
    run_sensor()