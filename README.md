# Data Science & Deep Learning Practice Projects

A comprehensive collection of data science, exploratory data analysis (EDA), machine learning, and artificial neural network (ANN) lab practicals.

**Author:** Shayan Azmi  
**GitHub Repository:** [https://github.com/shayanazmi/DataScience_practice](https://github.com/shayanazmi/DataScience_practice)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=flat&logo=keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2C5E3B?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-008080?style=flat&logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-F50057?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-111111?style=flat&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-37976B?style=flat&logo=python&logoColor=white)

---

## Projects & Notebooks Overview

### Data Science & Machine Learning Projects

| Project / Notebook | Description | Key Tech / Models |
|-------------------|-------------|-------------------|
| **Customer Churn Prediction** (`customer-churn-prediction.ipynb`) | Classification model predicting customer churn probability. | Scikit-Learn, Logistic Regression, Random Forest |
| **Insurance Sales EDA & Model** (`customer-insurance-sales-eda-and-performance-model.ipynb`) | Exploratory data analysis and predictive performance modeling for insurance sales. | Pandas, Seaborn, Feature Engineering |
| **Engine Health Diagnostics** (`engine-health-prediction.ipynb`) | Predictive maintenance modeling for engine sensor health metrics. | Machine Learning, Time Series / Regression |
| **Flight Ticket Propensity** (`flight-ticket-propensity-device-segmented-xgboost.ipynb`) | Device-segmented XGBoost model predicting flight ticket booking propensity. | XGBoost, Device Segmentation, Data Pipelines |
| **Supply Chain Analytics** (`supply-chain.ipynb`) | Optimization, inventory tracking, and supply chain metrics analysis. | Data Analytics, Matplotlib, Pandas |

---

### ANN & Deep Learning Lab Practicals

| Practical | Project Title | Description | Framework / Stack |
|-----------|---------------|-------------|-------------------|
| [**Practical 1**](./practical_1/practical_1_readme.md) | **Perceptron Simulation** | Single-layer perceptron built from scratch using NumPy learning AND gate logic. | Python 3, NumPy |
| [**Practical 2**](./practical_2/practical_2_readme.md) | **Single Neuron Model ANN** | Artificial Neural Network (4 hidden neurons + ReLU + Sigmoid) training AND gate logic. | Keras, TensorFlow, NumPy |
| [**Practical 3**](./practical_3/practical_3_readme.md) | **Feedforward Neural Network (FNN)** | Multi-layer Feedforward Neural Network using PyTorch (`nn.Module`, Adam, BCELoss). | PyTorch, Python 3 |
| [**Practical 4**](./practical_4/practical_4_readme.md) | **Keras MLP Multiclass Classification** | Multi-Layer Perceptron ($4 \to 10 \to 8 \to 3$) classifying Iris flower species using Softmax & Categorical Cross-Entropy. | Keras, TensorFlow, Scikit-Learn |
| [**Practical 5**](./practical_5/practical_5_readme.md) | **CNN Image Classification & Grad-CAM** | Convolutional Neural Network (Conv2D, MaxPool, BatchNorm, Dropout, GAP) classifying Cats vs. Dogs with Grad-CAM & Feature Maps (**80.50% Acc, 0.8843 ROC-AUC**). | TensorFlow / Keras, Scikit-Learn, PIL, imagehash, visualkeras |
| [**Practical 6**](./practical_6/practical_6_readme.md) | **Electrical Load Forecasting (MLP)** | Tapered Multi-Layer Perceptron ($29 \to 64 \to 32 \to 16 \to 1$) forecasting hourly national grid demand (**9,990.46 MW MAE, 5.51% MAPE, 0.6327 R²**). Grid search comparing activations (ReLU, Sigmoid, Tanh) and losses (MSE, MAE, MAPE). | TensorFlow / Keras, Scikit-Learn, Pandas, NumPy |

---

## Tech Stack & Tools

- **Machine Learning & Data Science:** Python 3.x, Pandas, NumPy, Scikit-Learn, XGBoost
- **Deep Learning Frameworks:** PyTorch, TensorFlow 2.x / Keras
- **Computer Vision & Explainable AI:** OpenCV / PIL, `imagehash` (pHash), `visualkeras` (3D volume rendering), Grad-CAM (Class Activation Mapping)
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebooks, Google Colab, Kaggle GPU Environments

---

## 🛠 Guidelines & Assistant Customizations

These documents define the style guidelines and instruction patterns used to format notebooks and write dataset-driven observations throughout this repository:

- [**Notebook Writing Guidelines**](./notbookwriting-skill.md): Comprehensive guidelines for structuring, explanation density, and mathematical honesty in machine learning notebooks.
- [**Observation Cells Guidelines**](./observationskills.md): Structured rules for inserting data-driven, post-cell explanation blocks following visualizations and metrics.

---

## Author & Contact

- **Author:** Shayan Azmi
- **GitHub:** [@shayanazmi](https://github.com/shayanazmi)
- **Repository:** [DataScience_practice](https://github.com/shayanazmi/DataScience_practice)
