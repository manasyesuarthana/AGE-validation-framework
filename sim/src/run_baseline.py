import json
import numpy as np
import os
import sys

# Import the new loader
# (Assuming this script is run from the repo root as `python sim/src/run_baseline.py`)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sim.src.load_data import load_real_epilepsy_dataset

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

# --- Constants for Energy Model ---
ENERGY_PER_SAMPLE_mJ = 0.05
ENERGY_PER_BYTE_mJ = 0.01 
BYTES_PER_SAMPLE = 4 

def linear_adaptive_sampler(sequence, threshold):
    """
    Implements the Linear Adaptive Sampling policy[cite: 457].
    Collects a sample if the change from the previous value exceeds the threshold.
    """
    collected_samples = []
    
    # Always take the first sample
    last_val = sequence[0]
    collected_samples.append(last_val)
    
    for i in range(1, len(sequence)):
        curr_val = sequence[i]
        # Check difference
        if abs(curr_val - last_val) > threshold:
            collected_samples.append(curr_val)
            last_val = curr_val # Update reference
            
    return collected_samples

def calculate_energy(sampled_data_length, message_byte_length):
    """Simple linear energy model."""
    collection_energy = sampled_data_length * ENERGY_PER_SAMPLE_mJ
    communication_energy = message_byte_length * ENERGY_PER_BYTE_mJ
    return collection_energy + communication_energy

def determine_dynamic_threshold(X_data, target_rate=0.5):
    """
    Calculates a threshold that achieves an approx target collection rate 
    across the dataset. This ensures the simulation isn't trivial.
    """
    # Calculate all absolute differences in the dataset
    all_diffs = []
    for seq in X_data:
        diffs = np.abs(np.diff(seq))
        all_diffs.extend(diffs)
    
    # Use percentile to find threshold. 
    # If we want to KEEP 50% of samples, we need the threshold to be 
    # at the 50th percentile of differences (roughly).
    # Higher percentile = Higher threshold = Fewer samples kept.
    threshold = np.percentile(all_diffs, (1.0 - target_rate) * 100)
    return threshold

def simulate_attack():
    print("Starting simulation with REAL Epilepsy Data...")
    
    # 1. Load Real Data
    # Ensure data files are at sim/data/ relative to repo root
    data_path = os.path.join("sim", "data") 
    X, y = load_real_epilepsy_dataset(data_path)
    
    # 2. Determine Threshold
    # We target a 50% sampling rate to create high variance between 
    # "active" (Seizure/Run) and "inactive" (Sit/Walk) classes.
    threshold = determine_dynamic_threshold(X, target_rate=0.5)
    print(f"Calculated Adaptive Threshold: {threshold:.4f}")

    message_lengths = []
    true_labels = []
    total_energy_mJ = 0.0 
    
    print("Running Adaptive Sampler...")
    for i, seq in enumerate(X):
        sampled_data = linear_adaptive_sampler(seq, threshold)
        
        sampled_length = len(sampled_data)
        
        # --- DEFENSE LOGIC SELECTION ---
        # Current Mode: Insecure Baseline
        message_bytes = sampled_length * BYTES_PER_SAMPLE
        
        # For Padding (Phase 4):
        # message_bytes = 206 * BYTES_PER_SAMPLE # Max length of Epilepsy seq
        
        # For AGE-Lite (Phase 4):
        # message_bytes = 100 # Target fixed size
        
        # Energy Calculation
        sequence_energy = calculate_energy(sampled_length, message_bytes)
        total_energy_mJ += sequence_energy

        message_lengths.append(sampled_length)
        true_labels.append(y[i])
        
    # Reshape for attack model
    X_attack = np.array(message_lengths).reshape(-1, 1)
    y_attack = np.array(true_labels)
    
    # 3. Attack Classifier (Ensemble of Decision Trees)
    # Matches paper methodology 
    base_estimator = DecisionTreeClassifier(max_depth=2)
    classifier = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,
        random_state=42
    )
    
    # 4. Cross Validation
    print("Running 5-fold cross-validation on attack model...")
    # Using 5-fold CV as per paper 
    cv_results = cross_validate(
        classifier, X_attack, y_attack, cv=5, 
        scoring=['accuracy', 'normalized_mutual_info_score']
    )
    
    avg_accuracy = np.mean(cv_results['test_accuracy'])
    avg_nmi = np.mean(cv_results['test_normalized_mutual_info_score'])
    avg_energy_per_seq = total_energy_mJ / len(X)

    print("--- Final Results ---")
    print(f"Attack Accuracy: {avg_accuracy:.4f}")
    print(f"Normalized Mutual Info (NMI): {avg_nmi:.4f}")
    print(f"Avg Energy per Sequence: {avg_energy_per_seq:.4f} mJ")

    # Output for CI/CD pipeline
    metrics = {
        "attacker_accuracy": avg_accuracy,
        "nmi": avg_nmi,
        "baseline_mae": 0.0, # Placeholder
        "simulated_energy": avg_energy_per_seq
    }

    if os.path.exists("/metrics"):
        output_dir = "/metrics"
    else:
        output_dir = "build/metrics"
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "metrics.json")
    
    with open(output_path, "w") as f:
        json.dump(metrics, f)
        
    print(f"Metrics successfully written to {output_path}")

if __name__ == "__main__":
    simulate_attack()