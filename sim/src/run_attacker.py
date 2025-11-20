import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

def run_attacker():
    print("[ATTACKER] Starting interception...")

    # 1. Locate Input Data
    # Docker: /data_in (ReadOnly Volume)
    # Local: build/metrics
    if os.path.exists("/data_in/sensor_output.json"):
        input_path = "/data_in/sensor_output.json"
        output_dir = "/data_out"
    else:
        input_path = "build/metrics/sensor_output.json"
        output_dir = "build/metrics"

    if not os.path.exists(input_path):
        print(f"[ATTACKER] CRITICAL: No input file found at {input_path}")
        sys.exit(1)

    with open(input_path, "r") as f:
        payload = json.load(f)

    traffic_log = payload["traffic_log"]
    sensor_metrics = payload["sensor_metrics"]

    # 2. Extract Features (Message Sizes) and Labels
    X_attack = []
    y_attack = []

    for entry in traffic_log:
        # The side-channel is the message size
        X_attack.append(entry["message_bytes"])
        y_attack.append(entry["label"])

    X_attack = np.array(X_attack).reshape(-1, 1)
    y_attack = np.array(y_attack)

    print(f"[ATTACKER] Intercepted {len(X_attack)} messages.")

    # 3. Train Attack Model
    base_estimator = DecisionTreeClassifier(max_depth=2)
    classifier = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,
        random_state=42
    )

    print("[ATTACKER] Running 5-fold cross-validation...")
    cv_results = cross_validate(
        classifier, X_attack, y_attack, cv=5, 
        scoring=['accuracy', 'normalized_mutual_info_score']
    )
    
    avg_acc = np.mean(cv_results['test_accuracy'])
    avg_nmi = np.mean(cv_results['test_normalized_mutual_info_score'])

    print(f"[ATTACKER] Attack Accuracy: {avg_acc:.4f}")
    print(f"[ATTACKER] NMI Score:       {avg_nmi:.4f}")

    # 4. Combine Metrics for Final Report
    # We combine the attacker's findings (Security) with the sensor's reported stats (Performance)
    final_metrics = {
        "nmi": avg_nmi,
        "attacker_accuracy": avg_acc,
        "baseline_mae": sensor_metrics["mae"],
        "simulated_energy": sensor_metrics["energy"]
    }

    # 5. Write Final Report
    # Check for container permission output or local fallback
    if not os.path.exists(output_dir):
         os.makedirs(output_dir, exist_ok=True)
         
    out_file = os.path.join(output_dir, "metrics.json")
    with open(out_file, "w") as f:
        json.dump(final_metrics, f)

    print(f"[ATTACKER] Final report written to {out_file}")

if __name__ == "__main__":
    run_attacker()