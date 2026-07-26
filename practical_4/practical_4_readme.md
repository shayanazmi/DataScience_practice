# Practical 4 - Implement Keras Multi-Layer Perceptron (MLP) for Multiclass Classification

This repository contains the implementation of a **Multi-Layer Perceptron (MLP)** built with **TensorFlow / Keras** for multiclass classification on the classic **Iris Flower Dataset**.

---

## Overview

While previous practicals focused on binary classification (such as logical AND gates with a single output neuron and Sigmoid activation), **Practical 4** expands neural network fundamentals to **multiclass classification**:

- **Target Problem**: Classifying Iris flowers into one of three species (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) based on four sepal and petal measurements.
- **Key Concepts Covered**:
  - Exploratory Data Analysis (EDA) & dataset validation.
  - Categorical target transformation via `LabelEncoder` and One-Hot Encoding (`to_categorical`).
  - Stratified sampling (`stratify=y_encoded`) and feature standardization via `StandardScaler`.
  - Multiclass neural network design using `Softmax` output activation and `Categorical Cross-Entropy` loss.
  - Training dynamics monitoring via real-time validation splits and loss/accuracy plots.
  - Out-of-sample evaluation using Confusion Matrices and detailed Classification Reports ($F_1$-score, Precision, Recall).

---

## Objectives

1. **Architecture Scaling**: Learn how to design an MLP with multiple hidden layers ($4 \to 10 \to 8 \to 3$) for multi-feature input and multi-class output.
2. **Binary vs. Multiclass Dynamics**: Understand the mathematical shift from binary targets ($1$ output neuron, `sigmoid`, `binary_crossentropy`) to multi-class targets ($K$ output neurons, `softmax`, `categorical_crossentropy`).
3. **Loss Surface Conditioning**: Apply feature scaling (`StandardScaler`) to prevent ill-conditioned Hessian loss surfaces and accelerate mini-batch gradient descent convergence.
4. **Generalization & Diagnostics**: Monitor training curves (loss and accuracy over 100 epochs) to detect overfitting/underfitting, and perform error analysis on unseen test data.

---

## Model Architecture

```
Input Layer (4 features: Sepal & Petal length/width)
       │
       ▼
Dense Hidden Layer 1 (4 → 10 neurons, ReLU activation)   [50 parameters]
       │
       ▼
Dense Hidden Layer 2 (10 → 8 neurons, ReLU activation)   [88 parameters]
       │
       ▼
Dense Output Layer (8 → 3 neurons, Softmax activation)   [27 parameters]
```

### Parameter Breakdown

$$\text{Params} = (n_{\text{in}} \times n_{\text{out}}) + n_{\text{out}}$$

| Layer | Type | Input Dim ($n_{\text{in}}$) | Output Dim ($n_{\text{out}}$) | Param Formula | Total Params | Activation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense 1** | Hidden | 4 | 10 | $(4 \times 10) + 10$ | 50 | `ReLU` |
| **Dense 2** | Hidden | 8 | 8 | $(10 \times 8) + 8$ | 88 | `ReLU` |
| **Dense 3** | Output | 8 | 3 | $(8 \times 3) + 3$ | 27 | `Softmax` |
| **Total** | | | | | **165** | |

---

## Mathematical Foundations

### 1. Feature Standardization (`StandardScaler`)
Input features are rescaled using training set mean ($\mu_{\text{train}}$) and standard deviation ($\sigma_{\text{train}}$):

$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

* **Optimization Impact**: Rescales elliptical loss contours to be isotropic (spherical), enabling gradient descent algorithms like Adam to take optimal, direct steps toward the minimum without oscillating.

### 2. Target Representation (One-Hot Encoding)
Discrete class labels $y \in \{0, 1, 2\}$ are mapped to orthogonal unit vectors in $\mathbb{R}^3$:

$$\text{Setosa (0)} \to \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \text{Versicolor (1)} \to \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad \text{Virginica (2)} \to \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

* **Theoretical Justification**: Prevents the network from imposing an artificial ordinal distance (e.g., assuming class 2 is "twice as large" as class 1). Pairs of orthogonal vectors maintain equal Euclidean distance ($\|e_i - e_j\|_2 = \sqrt{2} \quad \forall i \neq j$).

### 3. Softmax Activation Function
Converts raw logit outputs $z_k$ into normalized class probabilities $\hat{y}_k \in [0, 1]$:

$$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{such that} \quad \sum_{k=1}^{K} \hat{y}_k = 1$$

### 4. Categorical Cross-Entropy Loss
Measures divergence between true one-hot target vectors $Y$ and predicted distributions $\hat{Y}$:

$$L(\theta) = -\sum_{k=1}^{K} y_k \log(\hat{y}_k) = -\log(\hat{y}_{\text{true class}})$$

---

## Requirements & Dependencies

- Python 3.8+
- TensorFlow / Keras 2.x+
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib

Install dependencies via `pip`:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

---

## Notebook Workflow Structure

| Section | Topic | Description |
| :--- | :--- | :--- |
| **1. Importing Libraries** | Setup | Loads `numpy`, `pandas`, `matplotlib`, `sklearn`, and `tensorflow.keras`. |
| **2. Load Dataset** | Data Acquisition | Imports Iris dataset directly into a Pandas DataFrame. |
| **3. Exploratory Data Analysis** | EDA | Inspects `df.shape`, column data types, missing values, and verifies class balance ($50$ per class). |
| **4. Data Preprocessing** | Encoding | Extracts feature matrix $X$ and maps string targets to one-hot encoded vectors using `LabelEncoder` + `to_categorical`. |
| **5. Data Partitioning & Scaling** | Preprocessing | Performs 80/20 train/test split with `stratify=y_encoded` and applies `StandardScaler` to features. |
| **6. Model Architecture & Compilation** | Model Design | Defines sequential MLP ($4 \to 10 \to 8 \to 3$) compiled with `Adam` optimizer and `categorical_crossentropy` loss. |
| **7. Model Training** | Training | Trains for 100 epochs (batch size = 8) with a 20% validation split to monitor learning curves. |
| **8. Convergence Diagnostics** | Diagnostics | Plots training vs. validation loss and accuracy curves to analyze bias/variance tradeoff. |
| **9. Test Set Evaluation** | Generalization | Evaluates loss and accuracy on unseen holdout test set ($X_{\text{test}}, Y_{\text{test}}$). |
| **10. Confusion Matrix & Metrics** | Error Analysis | Computes `confusion_matrix` and per-class `classification_report` (Precision, Recall, $F_1$-score). |

---

## ▶ How to Run

1. Open `lab4-keras-mlp-for-multiclass-classification.ipynb` in **Jupyter Notebook**, **JupyterLab**, or **Google Colab**.
2. Execute all cells in sequential order (**Runtime → Run all** in Colab, or **Cell → Run All** in Jupyter).
3. Observe epoch loss outputs, training/validation plots, and test evaluation metrics.

---

## Results & Experimental Evaluation

### Test Set Metrics (Holdout $N = 30$)

- **Test Loss**: `0.1218`
- **Test Accuracy**: `93.33%`

### Classification Report

```text
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       0.90      0.90      0.90        10
 Iris-virginica       0.90      0.90      0.90        10

       accuracy                           0.93        30
      macro avg       0.93      0.93      0.93        30
   weighted avg       0.93      0.93      0.93        30
```

### Error Analysis
- **Iris-setosa**: Perfect separation ($1.00$ Precision, Recall, and $F_1$-score) due to distinct, non-overlapping feature distributions in 4D measurement space.
- **Iris-versicolor vs. Iris-virginica**: Minor decision boundary overlap ($90\%$ Precision/Recall) resulting in 1 misclassified instance between Versicolor and Virginica, highlighting natural feature proximity in boundary regions.

---

## Key Takeaways

1. **Multiclass Paradigm Shift**: Moving from binary to multiclass classification requires switching the output layer activation to `Softmax` and loss function to `Categorical Cross-Entropy`.
2. **Feature Normalization is Essential**: Unscaled inputs create unbalanced gradient updates across features; `StandardScaler` ensures uniform learning rates and faster model convergence.
3. **Stratified Splitting Preserves Class Distribution**: Using `stratify=y_encoded` ensures training and test splits retain the original dataset's class ratio ($1:1:1$), preventing sampling bias.
4. **Beyond Overall Accuracy**: Per-class precision, recall, $F_1$-score, and confusion matrices provide actionable diagnostic insights into where the model succeeds or experiences feature overlap.

---

## Author

Created as part of the **ANN & Deep Learning Lab** practical series demonstrating Multi-Layer Perceptron (MLP) multi-class classification using Keras/TensorFlow.
