# AGE Security & Performance Validation Framework

This repository contains the code and CI/CD framework for a project validating the security and performance of privacy-preserving algorithms for edge devices.

## 1. Project Thesis

### The Problem: Side-Channel Attacks on Edge Sensors

Low-power sensors, common in healthcare and IoT, often use **adaptive sampling** to save energy—sampling more when data is interesting and less when it's not. However, when this data is sent in encrypted batches, the *size of the encrypted message* often correlates directly with the sampling rate. This creates a critical **side-channel vulnerability**. A passive attacker can analyze these message sizes to infer sensitive information about the underlying data, even without breaking the encryption. This attack has been shown to be highly effective, in some cases revealing the occurrence of an epileptic seizure with 100% accuracy.

### The Solution: Adaptive Group Encoding (AGE)

This project aims to build a validation framework for **Adaptive Group Encoding (AGE)**, a novel defense designed to mitigate this specific side-channel attack. The core strategy of AGE is to ensure all data batches are encoded into **fixed-length messages**, breaking the link between message size and the information it contains.

Unlike simple padding, which is too energy-intensive for constrained devices, AGE uses an efficient, lossy encoding process involving quantization and pruning. It operates as a "drop-in" module between the sensor's sampling and encryption steps, adding negligible computational overhead while significantly saving energy by reducing the amount of data transmitted wirelessly.

This repository implements a DevSecOps pipeline to build, test, and validate the security and performance claims of the AGE algorithm.

## 2. Framework Architecture

The validation framework is designed as a "bifurcated pipeline" that simulates a passive side-channel attack:

*   **`sensor` Container:** This component runs the data sampling logic (e.g., using the AGE algorithm or a baseline) and generates messages. It simulates the behavior of a resource-constrained edge device.
*   **`attacker` Container:** This component acts as a passive observer. It only has access to the metadata of the messages (e.g., message length) and attempts to classify the original data based on this leakage.
*   **CI/CD Orchestrator (GitHub Actions):** A "heavy" pipeline that builds the containers, orchestrates the simulation, and runs automated security and performance assertions. It fails the build if a defense mechanism does not meet its predefined security goals (e.g., low attack accuracy).

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
