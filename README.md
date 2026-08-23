# Data Science & Deep Learning Practice Projects

A comprehensive collection of data science, exploratory data analysis (EDA), machine learning, and artificial neural network (ANN) lab practicals.

**Author:** Shayan Azmi  
**GitHub Repository:** [https://github.com/shayanazmi/DataScience_practice](https://github.com/shayanazmi/DataScience_practice)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=flat&logo=keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2C5E3B?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-37976B?style=flat&logo=python&logoColor=white)

---

## Projects & Notebooks Overview

### Data Science & Machine Learning Projects

| Project / Notebook | Description | Key Tech / Models |
| :--- | :--- | :--- |
| **Customer Churn Prediction** (`customer-churn-prediction.ipynb`) | Classification model predicting customer churn probability. | Scikit-Learn, Logistic Regression, Random Forest |
| **Insurance Sales EDA & Model** (`customer-insurance-sales-eda-and-performance-model.ipynb`) | Exploratory data analysis and predictive performance modeling for insurance sales. | Pandas, Seaborn, Feature Engineering |
| **Engine Health Diagnostics** (`engine-health-prediction.ipynb`) | Predictive maintenance modeling for engine sensor health metrics. | Machine Learning, Time Series / Regression |
| **Flight Ticket Propensity** (`flight-ticket-propensity-device-segmented-xgboost.ipynb`) | Device-segmented XGBoost model predicting flight ticket booking propensity. | XGBoost, Device Segmentation, Data Pipelines |
| **Supply Chain Analytics** (`supply-chain.ipynb`) | Optimization, inventory tracking, and supply chain metrics analysis. | Data Analytics, Matplotlib, Pandas |

---

### ANN & Deep Learning Lab Practicals

| Practical | Project Title | Description | Framework / Stack |
| :--- | :--- | :--- | :--- |
| [**Practical 1**](./practical_1/practical_1_readme.md) | **Perceptron Simulation** | Single-layer perceptron built from scratch using NumPy learning AND gate logic. | Python 3, NumPy |
| [**Practical 2**](./practical_2/practical_2_readme.md) | **Single Neuron Model ANN** | Artificial Neural Network (4 hidden neurons + ReLU + Sigmoid) training AND gate logic. | Keras, TensorFlow, NumPy |
| [**Practical 3**](./practical_3/practical_3_readme.md) | **Feedforward Neural Network (FNN)** | Multi-layer Feedforward Neural Network using PyTorch (`nn.Module`, Adam, BCELoss). | PyTorch, Python 3 |
| [**Practical 4**](./practical_4/practical_4_readme.md) | **Keras MLP Multiclass Classification** | Multi-Layer Perceptron (4 to 10 to 8 to 3) classifying Iris flower species using Softmax & Categorical Cross-Entropy. | Keras, TensorFlow, Scikit-Learn |
| [**Practical 5**](./practical_5/practical_5_readme.md) | **CNN Image Classification & Grad-CAM** | Convolutional Neural Network (Conv2D, MaxPool, BatchNorm, Dropout, GAP) classifying Cats vs. Dogs with Grad-CAM & Feature Maps (80.50% Acc, 0.8843 ROC-AUC). | TensorFlow / Keras, Scikit-Learn, PIL, imagehash, visualkeras |
| [**Practical 6**](./practical_6/practical_6_readme.md) | **Electrical Load Forecasting (MLP)** | Tapered Multi-Layer Perceptron (29 to 64 to 32 to 16 to 1) forecasting hourly national grid demand (9,990.46 MW MAE, 5.51% MAPE, 0.6327 R²). Grid search comparing activations (ReLU, Sigmoid, Tanh) and losses (MSE, MAE, MAPE). | TensorFlow / Keras, Scikit-Learn, Pandas, NumPy |
| [**Practical 7**](./practical_7/README.md) | **Human Activity Recognition (MLP)** | Compact Multi-Layer Perceptron (561 to 64 to 64 to 6) classifying smartphone sensor telemetry across 6 activities (**95.05% Test Acc, 0.9498 Macro F1, 87.98% Sitting Recall**). Subject-independent group validation, dynamic learning rate annealing, early stopping with checkpoint restoration, and a 3x3 hyperparameter grid sweep on Model 4 (ReLU + SGD Momentum). | TensorFlow / Keras, Scikit-Learn, Pandas, NumPy, Seaborn |

---

## Tech Stack & Tools

- **Machine Learning & Data Science:** Python 3.x, Pandas, NumPy, Scikit-Learn, XGBoost
- **Deep Learning Frameworks:** PyTorch, TensorFlow 2.x / Keras
- **Computer Vision & Explainable AI:** OpenCV / PIL, `imagehash` (pHash), `visualkeras`, Grad-CAM (Class Activation Mapping)
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebooks, Google Colab, Kaggle GPU Environments

---

## Engineering Guidelines & Custom Skills

This repository uses custom agent skills researched, engineered, and authored by Shayan Azmi to standardize experimental rigor, evidence-based reasoning, and viva-defensible reporting:

- [**Data Science Lab Harness**](./.agents/skills/data-science-lab-harness/): Master closed-loop framework (`INSPECT → UNDERSTAND → PLAN → ACT → OBSERVE → VERIFY → REFINE → REPEAT`) for evidence-driven, viva-defensible data science and machine learning notebooks.
- [**Notebook Engineering Style**](./.agents/skills/notebook-engineering-style/): Engineering standards for modular, dataset-aware, PEP 8-compliant, and mathematically clean Python machine learning notebooks.
- [**Evidence-Based Observations**](./.agents/skills/evidence-based-observations/): 3-part formula for writing post-cell observations grounded in exact numerical outputs and visual plots with direct causal mechanisms (zero analogies, zero fluff).
- [**Master Skills Index**](./.agents/skills/README.md): Master documentation and directory catalog for all custom skills.

---

## Author & Contact

- **Author:** Shayan Azmi
- **GitHub:** [@shayanazmi](https://github.com/shayanazmi)
- **Repository:** [DataScience_practice](https://github.com/shayanazmi/DataScience_practice)
