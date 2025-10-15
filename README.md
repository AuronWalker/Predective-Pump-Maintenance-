# 🛠️ Predictive Pump Maintenance  
**Author:** Auron Walker  
**Date:** October 2025  

---

## Overview  
This project presents the design and implementation of a **predictive maintenance system** for water pumps. By analysing electrical characteristics such as **voltage, current, and vibration**, the system can detect faults and communicate them using **LED indicators**.  

A **Random Forest machine learning model** is deployed on an **ESP32 microcontroller** to identify the type of fault based on sensor data. The goal is to make **future repairs faster, cheaper, and more sustainable**, while demonstrating the potential of **TinyML** in engineering applications.

---

## System Design  

### Design Overview  
- **Microcontroller:** ESP32  
- **Sensors:** Voltage, Current, and Accelerometer  
- **ML Model:** Random Forest (trained in Python, deployed via `microMLgen`)  
- **Output:** Fault indication through labeled LEDs  

The ESP32 processes real-time sensor data and predicts the pump’s condition locally. Designed for **low power**, **outdoor operation**, and **reliability**, it suits solar-powered fountain pumps or similar small-scale systems.

---

## Data  

### Source  
- **Dataset:** *Fieldlab Pump-Motor Dataset*  
- **Size:** 90 GB  
- **Sampling Rate:** 20 kHz (1 sample every 0.05 ms)  
- **Features:** Voltage, Current, and Vibration  
- **Fault Types:**  
  - Healthy operation  
  - Impeller damage  
  - Cavitation (discharge/suction side)  
  - Unbalanced motor/pump  

### Preprocessing  
1. Combined multi-phase voltage, current, and vibration signals.  
2. Converted AC data to **RMS values** for DC compatibility.  
3. Computed statistical features (over 50 ms window, 1000 samples):  
   - **Mean** 
   - **Variance**

These features form the model’s input to distinguish between normal and faulty conditions.

---

## Machine Learning Model  

### Model Type  
- **Algorithm:** Random Forest Classifier  
- **Reason for Selection:**  
  - High interpretability  
  - Fast inference resulting in a low computational demand

### Model Parameters  

| Parameter | Value | Description | Rationale (Microcontroller Focus) |
|------------|--------|-------------|----------------------------------|
| `n_estimators` | 30 | Number of trees in the forest | Reduces inference time and RAM usage |
| `criterion` | `gini` | Splitting function | Computation-efficient vs entropy |
| `min_samples_split` | 5 | Minimum samples per split | Prevents deep, slow trees |
| `max_depth` | 8 | Maximum tree depth | Ensures fast prediction |
| `random_state` | 42 | Random seed | Reproducible results |

Typical Random Forests use 50–200 trees, but this lightweight configuration ensures real-time inference on embedded hardware.

---

## Hardware Integration  

- The trained model (Python/scikit-learn) is converted to **C** using `microMLgen`.  
- The resulting header file is integrated into an **Arduino sketch** for the ESP32.  

This approach eliminates dependence on external computation or internet connectivity, enabling **offline fault detection**.