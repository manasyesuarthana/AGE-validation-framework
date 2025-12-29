# AGE & HOPE Security & Performance Validation Framework

This repository contains the code and CI/CD framework for a project validating the security and performance of privacy-preserving algorithms for edge devices, specifically focusing on Adaptive Group Encoding (AGE) and the Hybrid Optimization for Privacy and Energy (HOPE) framework.

## 1. Project Thesis

### The Problem: Side-Channel Attacks on Edge Sensors

Low-power sensors, common in healthcare and IoT, often use **adaptive sampling** to save energy—sampling more when data is interesting and less when it's not. However, when this data is sent in encrypted batches, two critical side-channel vulnerabilities emerge:

1.  **Message Size Side-Channel:** The *size of the encrypted message* often correlates directly with the sampling rate. A passive attacker can analyze these message sizes to infer sensitive information about the underlying data, even without breaking the encryption. This attack can be highly effective, in some cases revealing events like an epileptic seizure with high accuracy.

2.  **Timing Side-Channel (FATS Attack):** While periodic transmissions (sending data at fixed intervals) can prevent the message size side-channel, switching to more energy-efficient event-triggered transmissions (sending only when data is available) creates a **Timing Side-Channel**. An attacker can infer the nature of an event by observing the *frequency* and *inter-arrival time* of packets. A naive transition from periodic to event-triggered transmission due to low battery also creates a "Privacy Cliff," explicitly signaling a vulnerable state.

### The Solution: Adaptive Group Encoding (AGE) and Hybrid Optimization for Privacy and Energy (HOPE)

This project aims to build a validation framework for two novel defense mechanisms:

*   **Adaptive Group Encoding (AGE):** This defense mitigates the **message size side-channel**. The core strategy of AGE is to ensure all data batches are encoded into **fixed-length messages**, breaking the link between message size and the information it contains. Unlike simple padding, which is too energy-intensive, AGE uses an efficient, lossy encoding process (quantization and pruning), operating as a "drop-in" module with negligible computational overhead and significant energy savings.

*   **Hybrid Optimization for Privacy and Energy (HOPE):** This framework addresses the **timing side-channel** by gracefully managing the trade-off between privacy and energy as device battery depletes. Instead of a binary switch, HOPE treats privacy as a "luxury resource" that can be dynamically adjusted. Its key features include:
    *   **Decoupled State Architecture:** Separating data plane from control plane for low-power MCUs.
    *   **Dynamic Volatility Thresholding:** Adapting the "Panic Threshold" (when to start degrading privacy) based on data sensitivity (e.g., maintaining higher privacy during critical events like seizures).
    *   **The "Twilight" Sigmoid Transition:** As battery levels fall, HOPE uses a **sigmoid function** to determine the *probability* of sending a dummy packet, introducing controlled, probabilistic irregularity to obscure the timing channel from attackers.

This repository implements a DevSecOps pipeline to build, test, and validate the security and performance claims of *both* the AGE algorithm and the HOPE framework.

## 2. Framework Architecture

The validation framework is designed as a "bifurcated pipeline" that simulates passive side-channel attacks for both message size and timing:

*   **`sensor` Container:** This component runs the data sampling logic (e.g., using the AGE algorithm, a baseline, or the HOPE framework) and generates messages. It simulates the behavior of a resource-constrained edge device.
*   **`attacker` Container:** This component acts as a passive observer. It attempts to classify the original data based on message size metadata (for AGE validation) or infer battery levels from packet timing metadata (for HOPE validation).
*   **CI/CD Orchestrator (GitHub Actions):** A "heavy" pipeline that builds the containers, orchestrates the simulation, and runs automated security and performance assertions. It fails the build if a defense mechanism does not meet its predefined security goals (e.g., low attack accuracy for AGE, significant lifetime extension and high attack regression error for HOPE).

## 3. How to Run

*Prerequisites and setup instructions will be added here.*

---
*This `README.md` is temporary. Final documentation will include detailed setup instructions, experimental results, and analysis.*

## Citation

This framework was developed to validate the principles described in the following paper. If you use this code in your research, please consider citing the original work:

```bibtex
@inproceedings{Kannan2022ProtectingAS,
  title={{Protecting Adaptive Sampling from Information Leakage on Low-Power Sensors}},
  author={Tejas Kannan and Henry Hoffmann},
  booktitle={{Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '22)}},
  year={2022},
  pages={15},
  publisher={ACM},
  address={Lausanne, Switzerland},
  url={https://doi.org/10.1145/3503222.3507775}
}
```
