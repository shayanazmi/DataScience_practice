# **Practical 6: Electrical Load Forecasting -- MLP Regression Lab**

A comprehensive deep learning laboratory demonstrating how to design, implement, train, and evaluate a **Multi-Layer Perceptron (MLP)** built with **TensorFlow 2.x / Keras** to forecast hourly national grid demand in India. The project systematically benchmarks combinations of three activation functions and three loss functions across a 3x3 experimental grid to optimize continuous target regression.

**Author:** Shayan Azmi  
**Course:** ANN & Deep Learning (Sem 5)  
**Date:** August 2026

---

## 📌 Executive Summary & Empirical Results

We conduct a quantitative empirical benchmark comparing two baseline models and nine MLP configurations trained on 32,709 hourly observations (Jan 2019 – Sep 2022) and evaluated on a held-out test partition of 7,010 chronological samples (Jul 2023 – Apr 2024).

### Baseline Benchmarks vs. Winning MLP

To justify the addition of neural network complexity, we establish a performance floor using a Mean Predictor and an Ordinary Least Squares (OLS) Linear Regression model:

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Predictor** | 32,520.61 | 36,900.71 | 16.93% | -2.0983 | Baseline Floor |
| **Linear Regression** | 14,568.66 | 17,480.75 | 7.86% | 0.3047 | Linear Baseline |
| **MLP (Sigmoid + MSE)** | **9,990.46** | **12,704.54** | **5.51%** | **0.6327** | **Winning Model** |

* The winning MLP reduces the Mean Absolute Error by **31.4%** relative to the linear regression model.
* The average forecasting percentage error of **5.51%** lies well within acceptable operational planning tolerances.

---

### 3x3 Grid Search Quantitative Comparison

The table below consolidates the test-set performance metrics across all nine combinations of activation functions (`ReLU`, `Sigmoid`, `Tanh`) and loss functions (`MSE`, `MAE`, `MAPE`):

| Activation | Loss Function | MAE (MW) | RMSE (MW) | MAPE (%) | $R^2$ Score | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sigmoid** | **MSE** | **9,990.46** | **12,704.54** | **5.51%** | **0.6327** | **Winner (1st)** |
| **Sigmoid** | **MAE** | 10,733.20 | 13,402.39 | 5.91% | 0.5913 | Runner-Up (2nd) |
| **ReLU** | **MSE** | 13,777.36 | 17,132.50 | 7.46% | 0.3321 | 3rd |
| **ReLU** | **MAE** | 15,730.96 | 19,192.92 | 8.49% | 0.1618 | 4th |
| **Tanh** | **MAE** | 20,083.16 | 23,403.36 | 10.66% | -0.2463 | 5th |
| **Tanh** | **MSE** | 26,216.77 | 29,453.88 | 13.98% | -0.9740 | 6th |
| **Tanh** | **MAPE** | 31,506.04 | 35,875.01 | 16.43% | -1.9285 | 7th |
| **ReLU** | **MAPE** | 32,125.63 | 36,515.66 | 16.72% | -2.0340 | 8th |
| **Sigmoid** | **MAPE** | 33,748.35 | 38,160.09 | 17.56% | -2.3134 | 9th |

> ⚠️ **Watchpoint:** All configurations using the `MAPE` loss function failed to achieve a positive $R^2$ score. The relative percentage gradient updates led to numerical instability during optimization, causing the weights to converge to poor local minima.

---

## 🧠 Core Engineering & Data Hygiene Principles

### 1. Leakage Prevention (Regional Guard)
The dataset contains regional demand columns alongside the national total. An initial audit confirmed that the maximum absolute difference between the national demand and the sum of the regional demands is just **0.02 MW** (a direct linear combination). Retaining regional columns as inputs would introduce severe target leakage, so they are excluded, leaving the model to rely solely on calendar features.

### 2. Time-Aware Chronological Splitting
Random shuffling of time-series data leaks future information into training. We implement a chronological split (70% Train / 15% Val / 15% Test) to test the model on unseen future periods. The test target distribution exhibits a **+11.8% mean shift** (+19,000 MW) relative to the training set, forcing the model to extrapolate demand under long-term growth.

### 3. Cyclic Feature Engineering
Traditional integer hour and month representations introduce artificial boundaries (e.g., hour 23 and hour 0 appear distant but are adjacent). We apply trigonometric transformations to map hour, day of the week, and month onto a continuous unit circle, preserving temporal proximity for the neural network.

---

## 🏗 MLP Model Architecture Blueprint

The network uses a tapered feed-forward MLP configuration (`64 -> 32 -> 16` units) to progressively compress representations before predicting the single continuous target.

```
Input Layer (29 Scaled Features)
       │
       ▼
Dense Hidden Layer 1 (64 Neurons + Sigmoid Activation)  [1,920 parameters]
       │
       ▼
Dense Hidden Layer 2 (32 Neurons + Sigmoid Activation)  [2,080 parameters]
       │
       ▼
Dense Hidden Layer 3 (16 Neurons + Sigmoid Activation)  [528 parameters]
       │
       ▼
Output Dense Layer (1 Neuron + Linear Activation)       [17 parameters]
```

### Parameter Breakdown Table

$$\text{Dense Parameters} = (D_{\text{in}} \times D_{\text{out}}) + D_{\text{out}}$$

| Layer Name | Type | Input Dim | Output Shape | Parameter Formula | Param Count | Activation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`input_layer`** | Input | — | $(29,)$ | Non-parametric | 0 | None |
| **`hidden_1_sigmoid`** | Dense | 29 | $(64,)$ | $(29 \times 64) + 64$ | 1,920 | `Sigmoid` |
| **`hidden_2_sigmoid`** | Dense | 64 | $(32,)$ | $(64 \times 32) + 32$ | 2,080 | `Sigmoid` |
| **`hidden_3_sigmoid`** | Dense | 32 | $(16,)$ | $(32 \times 16) + 16$ | 528 | `Sigmoid` |
| **`output_layer`** | Dense | 16 | $(1,)$ | $(16 \times 1) + 1$ | 17 | `Linear` |
| **Total Trainable** | | | | | **4,545** | |

---

## 📐 Mathematical Foundations

### 1. Cyclic Coordinate Projection
To preserve temporal continuity, calendar indices are transformed into sine and cosine coordinates:
$$x_{\text{sin}} = \sin\left(\frac{2\pi \cdot x}{T}\right) \qquad x_{\text{cos}} = \cos\left(\frac{2\pi \cdot x}{T}\right)$$
where $T = 24$ for hours, $T = 7$ for days of the week, and $T = 12$ for months.

### 2. Feature Scaling (Z-Score Normalization)
Inputs are scaled using training set parameters to prevent feature magnitude imbalance from dominating gradients:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

### 3. Activation Functions
* **ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$
* **Sigmoid:** $f(x) = \frac{1}{1 + e^{-x}}$
* **Tanh (Hyperbolic Tangent):** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

### 4. Loss Functions & Optimization Penalties
* **Mean Squared Error (MSE):** $\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$
* **Mean Absolute Error (MAE):** $\mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$
* **Mean Absolute Percentage Error (MAPE):** $\mathcal{L}_{\text{MAPE}} = \frac{100\%}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right|$

---

## ⚙️ Requirements & Environment Setup

Run the following command to install the required libraries, including `openpyxl` which is required to parse Excel spreadsheets:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn openpyxl
```

---

## 📂 File Structure

| File Path | Description |
| :--- | :--- |
| [`p6.ipynb`](./p6.ipynb) | Primary executed notebook with preserved outputs, residual diagnostics, and the dynamic report card. |
| [`hourlyLoadDataIndia.xlsx`](./hourlyLoadDataIndia.xlsx) | Primary dataset featuring national and regional hourly electrical loads (Jan 2019 – Apr 2024). |
| [`monthly_temp.xlsx`](./monthly_temp.xlsx) | Secondary dataset containing monthly maximum temperature indicators (Jan 2019 – Dec 2021). |
| [`README.md`](./README.md) | Standard directory entry point markdown file. |
| [`practical_6_readme.md`](./practical_6_readme.md) | Duplicate lab documentation matching workspace naming conventions. |

---

## 💡 Key Takeaways

1. **Collinearity and Redundancy:** Standard gradient-based deep learning pipelines easily handle cyclic collinearity, but linear regression requires manual feature selection to avoid singular matrices.
2. **Sigmoid Optimization Success:** Sigmoid's bounded range of $(0, 1)$ closely matched the standard-scaled target, stabilizing weight updates and preventing gradient explosions seen in Tanh and ReLU.
3. **Extrapolation Limits:** No pure calendar-based model can fully predict unprecedented peak-demand demand growth (above 220,000 MW). Resolving this limitation requires incorporating weather indicators or autoregressive lags.
4. **Explainability Check:** Permutation feature importance confirmed that the model relies most heavily on the `year` feature (capturing baseline growth) and the cyclic `hour_sin`/`hour_cos` features (capturing the daily diurnal demand cycle).

---
**Author:** Shayan Azmi  
**Repository:** [DataScience_practice](https://github.com/shayanazmi/DataScience_practice)
