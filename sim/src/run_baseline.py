import json
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

ENERGY_PER_SAMPLE_mJ = 0.05
ENERGY_PER_BYTE_mJ = 0.01 
BYTES_PER_SAMPLE = 4      

# Placeholder for user's data loading and sampler code ---
def load_epilepsy_data():
    # User implements this: e.g., pd.read_csv('sim/data/epilepsy.csv')
    # Returns (all_sequences, all_labels)
    print("Data loading mock...")
    # Mock 100 sequences, 50 of class 0, 50 of class 1
    X = [np.random.rand(100) for _ in range(100)]
    y = [0]*50 + [1]*50
    return X, y

def linear_adaptive_sampler(sequence, threshold=0.1):
    # User implements this:
    # e.g., sample if |val[i] - val[i-1]| > threshold
    if np.mean(sequence) > 0.5: # Mock logic for variable length
        return sequence[:70] # "High activity" -> 70 samples
    else:
        return sequence[:30] # "Low activity" -> 30 samples

# End Placeholder

def calculate_energy(sampled_data_length, message_byte_length):
    collection_energy = sampled_data_length * ENERGY_PER_SAMPLE_mJ
    communication_energy = message_byte_length * ENERGY_PER_BYTE_mJ
    return collection_energy + communication_energy


def simulate_attack():
    print("Starting simulation...")
    X, y = load_epilepsy_data()
    
    message_lengths = []
    true_labels = []
    
    # 1. Initialize an accumulator for energy
    total_energy_mJ = 0.0 
    
    for i, seq in enumerate(X):
        sampled_data = linear_adaptive_sampler(seq)
        
        # --- [START] Energy Calculation Logic Integration ---
        
        sampled_length = len(sampled_data)
                
        # Option A: Insecure Baseline (Dynamic size)
        # The message size is exactly proportional to the data sampled.
        message_bytes = sampled_length * BYTES_PER_SAMPLE
        
        # Option B: Padding Defense (Fixed Max size)
        # message_bytes = 2000 # MAX_POSSIBLE_BYTES
        
        # Option C: AGE-Lite Defense (Fixed Target size)
        # message_bytes = 500 # TARGET_MB_BYTES
        
        # Calculate energy for this specific sequence
        sequence_energy = calculate_energy(sampled_length, message_bytes)
        
        # Add to the running total
        total_energy_mJ += sequence_energy
        
        # --- [END] Energy Calculation Logic Integration ---

        message_lengths.append(sampled_length)
        true_labels.append(y[i])
        
    # Reshape for scikit-learn
    X_attack = np.array(message_lengths).reshape(-1, 1)
    y_attack = np.array(true_labels)
    
    # Define the attack classifier
    base_estimator = DecisionTreeClassifier(max_depth=2)
    classifier = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,
        random_state=42
    )
    
    # Validate the attack
    print("Running 5-fold cross-validation on attack model...")
    cv_results = cross_validate(
        classifier, X_attack, y_attack, cv=5, 
        scoring=['accuracy', 'normalized_mutual_info_score']
    )
    
    avg_accuracy = np.mean(cv_results['test_accuracy'])
    avg_nmi = np.mean(cv_results['test_normalized_mutual_info_score'])
    
    # Calculate Average Energy per Sequence
    avg_energy_per_seq = total_energy_mJ / len(X)

    print(f"Attack Accuracy: {avg_accuracy:.4f}")
    print(f"Normalized Mutual Info (NMI): {avg_nmi:.4f}")
    print(f"Avg Energy per Sequence: {avg_energy_per_seq:.4f} mJ")

    # Output for CI/CD pipeline
    metrics = {
        "attacker_accuracy": avg_accuracy,
        "nmi": avg_nmi,
        "baseline_mae": 0.0, # Placeholder for Phase 4
        "simulated_energy": avg_energy_per_seq # <--- Updated with real calculation
    }
    
    
    # Write to a file for the "Assert" step
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    
    print("Metrics successfully written to metrics.json")

if __name__ == "__main__":
    simulate_attack()
