<div align="center">

# Data Science & Deep Learning Lab

**Production-grade machine learning pipelines, deep neural network architectures, and evidence-based AI research.**

[![Star Repo](https://img.shields.io/badge/⭐_Star-Repository-0969da?style=flat-square)](https://github.com/shayanazmi/DataScience_practice)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-2ea44f?style=flat-square)](https://github.com/shayanazmi/DataScience_practice/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Open in Colab](https://img.shields.io/badge/Colab-Ready-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/shayanazmi/DataScience_practice)

[Explore Practicals](#-deep-learning-practicals-leaderboard) • [Industry Projects](#-industry-machine-learning-projects) • [Crown Jewels](#-featured-crown-jewels) • [Agent Skills](#-autonomous-ai-research-harness) • [Quickstart](#-reproduction--quickstart)

</div>

---

## ⚡ Repository at a Glance

```
├── 🧠 7 Deep Learning Practicals (Perceptrons, PyTorch FNN, Keras MLP, CNN + Grad-CAM XAI, HAR Sensors)
├── 📊 5 Industry ML Pipelines (Device-Segmented XGBoost, Predictive Maintenance, Churn, Insurance EDA)
├── 🔬 Autonomous Research Harness (.agents/skills closed-loop experimental engineering)
└── 🚀 1-Click Reproducible (Zero-setup Google Colab launchers for all notebooks)
```

---

## 💎 Featured Crown Jewels

### 1. Computer Vision & Explainable AI (Grad-CAM) — Practical 5
> **Binary Image Classification with Explainability Heatmaps** | **80.50% Test Accuracy** | **0.8843 ROC-AUC**

A Convolutional Neural Network with Global Average Pooling (GAP), dynamic Dropout, and visual interpretability using **Gradient-weighted Class Activation Mapping (Grad-CAM)** and layer-wise feature map extractions.

```mermaid
flowchart LR
    subgraph Input
        A[RGB Image 150x150]
    end
    subgraph Feature_Extractor["Feature Extractor (CNN)"]
        B[Conv2D 32 + MaxPool] --> C[Conv2D 64 + MaxPool]
        C --> D[Conv2D 128 + MaxPool]
        D --> E[Conv2D 128 + MaxPool]
    end
    subgraph Classifier["Classification Head"]
        E --> F[Global Average Pooling]
        F --> G[Dense 512 + Dropout 0.5]
        G --> H[Sigmoid Output: Cat vs Dog]
    end
    subgraph XAI["Explainable AI (XAI)"]
        E -.->|Gradients of Conv_4| I[Grad-CAM Heatmap]
        I --> J[Superimposed Explainability Overlay]
    end
    A --> B
```

- **Highlights:** Perceptual image hashing (`imagehash`), batch normalization, data augmentation, visual verification of neural activation loci.
- **Deep Dive:** [`practical_5/practical_5_readme.md`](./practical_5/practical_5_readme.md) • [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/practical_5/cnn_implementation_keras.ipynb)

---

### 2. Smartphone Sensor Telemetry & HAR (MLP) — Practical 7
> **Human Activity Recognition from 561-feature Sensor Telemetry** | **95.05% Test Accuracy** | **0.9498 Macro F1**

Multi-layer Perceptron classifying 6 distinct human physical postures from accelerometer and gyroscope telemetry. Evaluated with strict **Subject-Independent Group Stratification** to prevent spatial-temporal data leakage across human participants.

```mermaid
flowchart TD
    subgraph Sensor_Input["561-D Tri-Axial Sensor Telemetry"]
        S[tBodyAcc, tGravityAcc, tBodyGyro, fBodyAcc, Energy, Entropy]
    end
    subgraph Model_Architecture["Deep MLP Topology (Model 4)"]
        M1[Dense 64 + ReLU + Dropout 0.3]
        M2[Dense 64 + ReLU + Dropout 0.3]
        M3[Dense 32 + ReLU]
        Out[Dense 6 Softmax Output]
        S --> M1 --> M2 --> M3 --> Out
    end
    subgraph Validation_Strategy["Subject-Independent Split"]
        V[GroupKFold / Subject Partitioning] -.->|Zero User Leakage| M1
    end
    subgraph Metrics["Key Class Performance"]
        Out --> C1[LAYING: 100.0% Recall]
        Out --> C2[WALKING: 98.99% Recall]
        Out --> C3[SITTING: 87.98% Recall]
    end
```

- **Highlights:** Dynamic Learning Rate Annealing (`ReduceLROnPlateau`), Early Stopping checkpoint restoration, 3×3 optimizer hyperparameter grid sweep (Adam vs SGD Momentum).
- **Deep Dive:** [`practical_7/README.md`](./practical_7/README.md) • [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/practical_7/practical_7_keras_mlp_multiclass.ipynb)

---

### 3. Industrial End-to-End ML Pipelines
Real-world operational intelligence notebooks built with production-grade data pipelines:

| Project | Key Focus & Technical Novelty | Architecture / Stack | Launch |
| :--- | :--- | :--- | :---: |
| **Flight Propensity Modeling** | Device-segmented conversion prediction on imbalanced booking telemetry. | Segmented XGBoost, Scikit-Learn | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/flight-ticket-propensity-device-segmented-xgboost.ipynb) |
| **Engine Health Diagnostics** | Multi-sensor predictive maintenance and remaining useful life (RUL) modeling. | Sensor Signal Regression, Scikit-Learn | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/engine-health-prediction.ipynb) |
| **Supply Chain Analytics** | Multi-echelon inventory optimization, lead time risk scoring, and demand profiling. | Pandas, Seaborn, Optimization Analytics | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/supply-chain.ipynb) |

---

## 🏆 Deep Learning Practicals Leaderboard

Comprehensive practicals covering fundamental perceptrons to advanced deep learning architectures:

| # | Practical | Description & Methodology | Architecture / Framework | Benchmark Metrics | Run |
| :-: | :--- | :--- | :--- | :--- | :-: |
| **P1** | [**Perceptron Simulation**](./practical_1/practical_1_readme.md) | Single-layer Perceptron built from scratch with NumPy learning logical AND gate decision boundaries. | NumPy, Python 3 | 100% Convergence (Loss $\to$ 0) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/practical_1/practical_1.ipynb) |
| **P2** | [**Single Neuron ANN**](./practical_2/practical_2_readme.md) | Feedforward ANN with 4 hidden neurons, ReLU activation, and Sigmoid output training binary logic gates. | Keras, TensorFlow | 100% Training Accuracy | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/practical_2/practical_2.ipynb) |
| **P3** | [**PyTorch FNN**](./practical_3/practical_3_readme.md) | Modular PyTorch `nn.Module` Feedforward Neural Network using Adam optimizer and `BCELoss`. | PyTorch 2.x, Python 3 | 100% Binary Accuracy | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/practical_3/practical_3.ipynb) |
| **P4** | [**Iris Multiclass MLP**](./practical_4/practical_4_readme.md) | Deep MLP ($4 \to 10 \to 8 \to 3$) for multi-class classification using Softmax and Categorical Cross-Entropy. | Keras, Scikit-Learn | 96.67% Test Accuracy | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/practical_4/practical_4.ipynb) |
| **P5** | [**CNN & Grad-CAM XAI**](./practical_5/practical_5_readme.md) | 4-Stage Conv2D + GAP architecture on Cats vs Dogs with visual interpretability and activation maps. | Keras, TensorFlow, Grad-CAM | **80.50% Acc, 0.8843 ROC-AUC** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/practical_5/cnn_implementation_keras.ipynb) |
| **P6** | [**Grid Load Forecasting**](./practical_6/practical_6_readme.md) | Tapered regression MLP ($29 \to 64 \to 32 \to 16 \to 1$) predicting national grid load with multi-activation sweeps. | Keras, Scikit-Learn, Pandas | **9,990 MW MAE, 5.51% MAPE** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/practical_6/p6.ipynb) |
| **P7** | [**Smartphone HAR MLP**](./practical_7/README.md) | Deep sensory MLP on 561-D telemetry across 6 activities with Subject-Independent cross-validation. | Keras, TensorFlow, Seaborn | **95.05% Acc, 0.9498 Macro F1** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/practical_7/practical_7_keras_mlp_multiclass.ipynb) |

---

## 📊 Industry Machine Learning Projects

| Notebook | Domain | Key Machine Learning Techniques | Key Libraries | Run |
| :--- | :--- | :--- | :--- | :-: |
| [`customer-churn-prediction.ipynb`](./customer-churn-prediction.ipynb) | FinTech / SaaS | Class imbalance handling, Random Forests, Gradient Boosting, ROC curves | `scikit-learn`, `pandas`, `seaborn` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/customer-churn-prediction.ipynb) |
| [`customer-insurance-sales-eda-and-performance-model.ipynb`](./customer-insurance-sales-eda-and-performance-model.ipynb) | InsurTech | Statistical EDA, missing data imputation, conversion probability scoring | `pandas`, `matplotlib`, `scipy` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/customer-insurance-sales-eda-and-performance-model.ipynb) |
| [`engine-health-prediction.ipynb`](./engine-health-prediction.ipynb) | IoT / Manufacturing | Sensor time-series telemetry analysis, degradation threshold modeling | `scikit-learn`, `numpy` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/engine-health-prediction.ipynb) |
| [`flight-ticket-propensity-device-segmented-xgboost.ipynb`](./flight-ticket-propensity-device-segmented-xgboost.ipynb) | E-Commerce / Travel | Device-segmented feature engineering, hyperparameter tuning, XGBoost | `xgboost`, `scikit-learn`, `pandas` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/flight-ticket-propensity-device-segmented-xgboost.ipynb) |
| [`supply-chain.ipynb`](./supply-chain.ipynb) | Logistics & Ops | Lead-time variance estimation, stockout probability, distribution metrics | `pandas`, `matplotlib`, `seaborn` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shayanazmi/DataScience_practice/blob/main/supply-chain.ipynb) |

---

## 🤖 Autonomous AI Research Harness

All laboratory practicals and machine learning pipelines in this repository adhere to custom agent engineering skills researched and authored to enforce **strict experimental rigor, zero data leakage, and viva-defensible results**:

```mermaid
flowchart TD
    I[1. INSPECT<br/><i>Schema & Null Audits</i>] --> U[2. UNDERSTAND<br/><i>Math & Leakage Risk</i>]
    U --> P[3. PLAN<br/><i>Hypothesis & Baselines</i>]
    P --> A[4. ACT<br/><i>Vectorized Execution</i>]
    A --> O[5. OBSERVE<br/><i>Exact Numerical Evidence</i>]
    O --> V[6. VERIFY<br/><i>Shape & Invariant Proofs</i>]
    V --> R[7. REFINE<br/><i>Hyperparameter Tuning</i>]
    R --> REP[8. REPEAT / DEFENSIBLE<br/><i>Viva-Ready Conclusions</i>]
```

- [`data-science-lab-harness`](./.agents/skills/data-science-lab-harness/): The 8-stage closed-loop execution protocol for evidence-grounded machine learning workflows.
- [`notebook-engineering-style`](./.agents/skills/notebook-engineering-style/): Engineering standards for PEP 8 compliance, deterministic seeding, and modular data science pipelines.
- [`evidence-based-observations`](./.agents/skills/evidence-based-observations/): Strict 3-part formula tying cell conclusions directly to exact numerical outputs and causal mechanisms (zero fluff, zero analogies).
- [`Master Skills Catalog`](./.agents/skills/README.md): Index of all custom agent skill packages.

---

## 🚀 Reproduction & Quickstart

### Option 1: 1-Click Interactive Cloud Execution
Click any **Open In Colab** badge next to the respective practical or project above to launch an interactive GPU/CPU environment instantly.

### Option 2: Local Reproduction
```bash
# 1. Clone the repository
git clone https://github.com/shayanazmi/DataScience_practice.git
cd DataScience_practice

# 2. Create and activate a clean Python environment
conda create -n ml-lab python=3.10 -y
conda activate ml-lab

# 3. Install dependencies
pip install -r practical_7/requirements.txt  # Or install core ML/DL packages:
pip install torch torchvision tensorflow scikit-learn xgboost pandas numpy matplotlib seaborn jupyterlab

# 4. Launch JupyterLab
jupyter lab
```

---

## ⭐ Show Your Support

If you find these notebooks, deep learning architectures, or custom AI agent skills helpful, please consider **starring the repository** to support open research!

<div align="center">

[![Star on GitHub](https://img.shields.io/badge/⭐_Star_This_Repository-shayanazmi%2FDataScience__practice-0969da?style=for-the-badge&logo=github)](https://github.com/shayanazmi/DataScience_practice)

</div>

---

## 📌 Citation & Attribution

If you use these architectures, notebooks, or custom AI agent skills in your research, coursework, or projects, please provide credit:

```bibtex
@misc{azmi2026datascience,
  author = {Shayan Azmi},
  title = {Data Science & Deep Learning Lab: Production-Grade ML Pipelines & Architectures},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shayanazmi/DataScience_practice}}
}
```

---

## 👤 Author & Connect

- **Author:** Shayan Azmi
- **GitHub:** [@shayanazmi](https://github.com/shayanazmi)
- **Repository:** [shayanazmi/DataScience_practice](https://github.com/shayanazmi/DataScience_practice)

---

<div align="center">

<sub>Crafted with experimental rigor, evidence-based engineering, and deep learning.</sub>

</div>
