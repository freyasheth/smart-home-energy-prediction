# Deep Learning-Based Smart Home Energy Consumption Forecasting System

## Overview

This project presents a comprehensive, multi-model deep learning framework designed to forecast household electrical energy consumption. Inspired by the research study *"CNN-LSTM based deep learning application on Jetson Nano: Estimating electrical energy consumption for future smart homes"* by Gozuoglu et al., this repository extends the foundational concepts to explore sequential, parallel, and attention-based neural network architectures.

Developed across interactive Google Colab notebooks, the system integrates robust data preprocessing pipelines, multiple datasets, and cutting-edge model interpretability using **SHAP** (SHapley Additive exPlanations) not only to predict energy use but also to understand the *why* behind the predictions.

---

## Research Context & Conceptual Architecture

The referenced research focuses on building a demand-side smart home energy management system using IoT edge devices and deep learning models. 

**The conceptual hardware/data flow of this ecosystem includes:**
1. **Data Collection:** IoT sensors (e.g., ESP-32, NodeMCU) collect real-time AC energy metrics.
2. **Transmission:** Data is transmitted via the lightweight MQTT protocol.
3. **Storage:** Stored in a centralized database (e.g., MariaDB on a Home Assistant server).
4. **AI Processing:** Time-series data is sent to an AI processor (like the NVIDIA Jetson Nano) to execute CNN-LSTM models.
5. **Actionable Output:** Predictions are generated to profile consumption behavior and optimize future smart grids.

This repository handles the **AI Processing** core of this pipeline, taking historical time-series data and mapping it to future consumption behavior.

---

## Datasets

To ensure the models generalize well and can handle different feature spaces, the project trains separate pipelines on two distinct datasets:

### 1. Kaggle: Smart Home Energy Consumption
* **Focus:** Feature-rich, categorical processing across multiple households.
* **Features:** Outdoor Temperature (°C), Household Size, Appliance Type, Season.
* **Target:** Energy Consumption (kWh).
* **Processing:** Sequences generated with a lookback window of 10. Handled via `ColumnTransformer` (MinMaxScaler for numerical, OneHotEncoder for categorical).
* https://www.kaggle.com/datasets/mexwell/smart-home-energy-consumption

### 2. UCI Individual Household Electric Power Consumption
* **Focus:** Deep architecture search, temporal baseline testing, and Explainable AI (XAI).
* **Features:** Univariate/Multivariate active power metrics.
* **Processing:** Sequences generated with a lookback window of 24 (representing 1 day of hourly data).
* https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

---

## Data Preprocessing Pipeline

Handling time-series data requires strict protocols to prevent data leakage. Our pipeline enforces the following steps:

1. **Temporal Structuring:** `Date` and `Time` columns are combined into a single `Datetime` index, and the dataset is sorted chronologically.
2. **Strict Train/Test Splitting:** A temporal split (80% train / 20% test) is performed **BEFORE** scaling and sequence generation to ensure zero future data leakage.
3. **Scaling & Encoding:** Numerical features are normalized using `MinMaxScaler`, while categorical features undergo one-hot encoding.
4. **Sliding Window Sequencing:** The `create_sequences(data, target_index, lookback)` function transforms the continuous time-series into supervised learning windows (e.g., using the past 24 hours to predict the 25th).

---

## Model Architectures

We built and evaluated four distinct deep learning architectures to compare feature extraction and temporal sequencing capabilities:

### 1. Paper Baseline (CNN-LSTM)
* **Structure:** 5 Conv1D layers (filters: 64, 128, 256, 128, 64) followed by 6 LSTM layers and a Dense linear output.
* **Purpose:** Replicates the reference paper. The CNN layers extract localized temporal patterns (spikes in energy use), while the deep LSTM stack captures long-term dependencies.

### 2. Parallel CNN-LSTM
* **Structure:** A multi-branch architecture. Branch 1 is a deep CNN; Branch 2 is an LSTM. The outputs of both branches are concatenated before passing to Dense layers.
* **Advantage:** Simultaneous, independent learning of short-term anomalies and long-term trends, often reducing the bottleneck of purely sequential processing.

### 3. CNN-GRU
* **Structure:** Conv1D layers followed by GRU (Gated Recurrent Unit) layers.
* **Advantage:** GRUs solve the vanishing gradient problem similarly to LSTMs but with fewer parameters, leading to faster convergence and reduced computational overhead.

### 4. Attention-Based LSTM
* **Structure:** An LSTM network augmented with a custom Self-Attention mechanism.
* **Advantage:** Instead of treating all historical data equally, the attention weights highlight exactly *which* specific past timesteps (e.g., a specific hour yesterday) contribute most heavily to the current prediction.

---

## Explainable AI (SHAP Integration)

Deep learning models are often "black boxes." To ensure transparency, we integrated **SHAP KernelExplainer** to map feature importance across both the variables and the time dimension.

**Implementation Workflow:**
1. **Dimensionality Reduction:** Because SHAP expects 2D inputs for its KernelExplainer, the 3D sequences `(samples, timesteps, features)` are flattened to 2D `(samples, timesteps × features)`.
2. **Prediction Wrapper:** A custom function reshapes the flattened data back to 3D on the fly during model inference.
3. **Background Sampling:** A representative background dataset (50 samples) is used to establish baseline expectations, drastically speeding up computation.
4. **Summary Plots:** Generates visualizations showing the impact of specific features at specific timesteps (e.g., `Outdoor_Temp_t18`) on the final energy prediction.

---

## Training Configuration & Metrics

* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)
* **Epochs:** 50 (practical Google Colab limits) to 600 (research-level replication).
* **Batch Size:** 16
* **Callbacks:** `ModelCheckpoint` (saving the best model based on validation loss), `EarlyStopping`, and `ReduceLROnPlateau`.

**Evaluation Metrics:**
All models are evaluated on the inverse-transformed predictions to reflect actual physical units (kWh/Watts):
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)

## Final Model Comparison Table

| Architecture | MSE | RMSE | MAE | MAPE (%) | R-Squared | Train Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Parallel_CNN_LSTM** | 0.237833 | 0.487681 | 0.338824 | 45.187862 | 0.554238 | 130.342547 |
| **CNN_GRU** | 0.239686 | 0.489577 | 0.338875 | 43.701388 | 0.550765 | 292.808396 |
| **Paper_Baseline_CNN_LSTM** | 0.241592 | 0.491520 | 0.338260 | 45.521084 | 0.547191 | 409.238405 |
| **Attention_LSTM** | 0.243688 | 0.493648 | 0.340515 | 44.096460 | 0.543264 | 331.092268 |

---

## Repository Artifacts & Saved Models

Running the pipelines will generate the following deployment-ready artifacts:
* `saved_models/Paper_Baseline_CNN_LSTM.keras`
* `saved_models/Parallel_CNN_LSTM.keras`
* `saved_models/CNN_GRU.keras`
* `saved_models/Attention_LSTM.keras`
* `preprocessor.pkl` (Fitted ColumnTransformer)
* `y_scaler.pkl` (Fitted target scaler)
* `history.json` (Training loss histories for plotting)

---

## Execution Instructions (Google Colab)

1. **Environment Setup:** Ensure you are running a Python 3 environment. Install the required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn shap scikit-learn tensorflow kagglehub joblib
   ```
2. **Dataset Access:** The Kaggle notebook utilizes `kagglehub.dataset_download("mexwell/smart-home-energy-consumption")` to dynamically pull the data. Ensure your Colab environment has internet access enabled.
3. **Run Sequentially:** Execute the cells in order. Note that the temporal sequence generation requires significant RAM depending on the lookback window.
4. **SHAP Computation:** *Note: The SHAP KernelExplainer is computationally expensive.* If testing the notebook, keep the `bg_idx` (background size) and `X_explain_flat` sizes small (e.g., 20-50 samples) to prevent runtime timeouts.

## Results
<img width="926" height="523" alt="image" src="https://github.com/user-attachments/assets/573e31a5-ae34-42c7-8652-983f3869605f" />
<img width="1143" height="470" alt="image" src="https://github.com/user-attachments/assets/0d4259bf-2c01-4c51-a73c-35356d769a58" />
<img width="704" height="393" alt="image" src="https://github.com/user-attachments/assets/0f7aed66-17b5-4a34-9914-4b7a91ac932d" />
<img width="718" height="374" alt="image" src="https://github.com/user-attachments/assets/0a42a28f-4b08-4e9b-a79c-11a695785dec" />
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/126ed2f8-7ee4-4a92-bd77-b91932ff4bbb" />
<img width="952" height="489" alt="image" src="https://github.com/user-attachments/assets/18435859-afd6-4df3-aeb8-f1b4441c1e74" />
<img width="961" height="489" alt="image" src="https://github.com/user-attachments/assets/4421023f-5fb3-4dd4-b93b-bb5518262ebb" />




