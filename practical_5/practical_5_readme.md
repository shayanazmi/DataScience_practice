# **Practical 5: Implement a CNN Model using Keras for Dogs & Cats Classification**

A comprehensive, production-grade computer vision laboratory demonstrating how to design, implement, train, evaluate, and explain **Convolutional Neural Networks (CNNs)** built with **TensorFlow 2.x / Keras** for binary image classification on the **Microsoft Cats vs. Dogs Dataset**.

---

## 📌 Executive Summary & Empirical Results

We conduct a quantitative empirical benchmark across three custom CNN architectures trained from scratch on 4,000 standardized $150 \times 150 \times 3$ RGB images (80% Train / 20% Validation split):

### Quantitative Performance Comparison

| Metric | Base Model | Tuned CNN | GAP CNN | Best Performer |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 76.00% | **80.50%** | 67.88% | **Tuned CNN (+4.50%)** |
| **Precision** | 79.21% | **85.25%** | 67.33% | **Tuned CNN (+6.04%)** |
| **Recall** | 72.71% | **75.36%** | 73.67% | **Tuned CNN (+2.65%)** |
| **F1-Score** | 0.7582 | **0.8000** | 0.7036 | **Tuned CNN (+0.0418)** |
| **ROC-AUC** | 0.8328 | **0.8843** | 0.7488 | **Tuned CNN (+0.0515)** |

* **Winner**: The **Tuned CNN** achieved **80.50% Accuracy** and **0.8843 ROC-AUC** with an outstanding **85.25% Precision**, demonstrating superior generalization and variance reduction.

---

## 🧠 Core Pedagogical & Engineering Guidelines

All notebook cells and lab documentation adhere strictly to the following hybrid execution guidelines:

### 1. Concept Rules (The Theory)
- **First Principles First:** Raw array manipulations and spatial geometry are explained before mentioning library abstractions.
- **The Human Analogy:** Complex matrix operations mapped to physical actions (e.g., comparing a convolution filter to sliding a magnifying glass across a photograph vs. shredding it into 67,500 strips for an MLP).
- **The Dimension Trace:** Geometric array transformations explicitly documented at every step:
  `Input Shape: [Batch, 150, 150, 3] ──► Conv2D(32): [Batch, 150, 150, 32] ──► MaxPool(2x2): [Batch, 75, 75, 32]`
- **The 3-Sentence Cap:** Paragraphs strictly limited to 3 sentences maximum for high scannability.

### 2. Code Rules (The Mechanics & S.P.R. Layout)
- **Code Previews & Intent Statements:** Code blocks open with a ````python ... ```` preview box and a single intent sentence stating the programmatic goal.
- **The S.P.R. Layout Engine:**
  - **Syntax:** The abstract blueprint or signature of the function.
  - **Parameter Breakdown:** Crucial inputs and justifications for parameter choices.
  - **Return Value:** Specific object class or data structure returned post-execution.
- **Isolate Magic Numbers:** Every arbitrary constant explicitly justified.

### 3. Output Rules (Reading the Results)
- **The Macro Summary:** Direct statement summarizing final visualization or metric outcome (referencing exact numbers: **80.50% Acc, 0.8843 ROC-AUC, 85.25% Precision**).
- **Component Deconstruction:** Axis, label, confusion matrix quadrant, and visual flag identification.
- **Spot the Outliers:** Explicit identification of edge cases, anomalies, and decision threshold behavior ($p=0.50$).

### 4. Motivation Rules (The "Why")
- **The Problem-First Approach:** Problems stated before introducing architectural layers.
- **Expose the Trade-Offs:** Engineering compromises explicitly highlighted.
- **Deploy Blockquote "Watchpoints":**
  > ⚠️ **Watchpoint:** `GlobalAveragePooling2D` reduces bottleneck parameter count from 21.2M down to 65.7K, but requires lower initial learning rates ($5 \times 10^{-4}$) to prevent spatial averaging from washing out local features.

---

## 🏗 Model Architecture Blueprint (Tuned CNN Winner)

```
Input Layer (150 × 150 × 3 RGB Tensor)
       │
       ▼
Enhanced Data Augmentation (Flip, Rotation, Zoom, Translation, Brightness, Contrast)
       │
       ▼
Rescaling (1 / 255.0) -> [0.0000, 1.0000]
       │
       ▼
Block 1: Conv2D (32, 3x3, L2=5e-5) ──► BatchNorm ──► ReLU ──► MaxPool2D (2x2)  [1,024 params]
       │
       ▼
Block 2: Conv2D (64, 3x3, L2=5e-5) ──► BatchNorm ──► ReLU ──► MaxPool2D (2x2)  [18,752 params]
       │
       ▼
Block 3: Conv2D (128, 3x3, L2=5e-5) ──► BatchNorm ──► ReLU ──► MaxPool2D (2x2) [74,368 params]
       │
       ▼
Block 4: Conv2D (256, 3x3, L2=5e-5) ──► BatchNorm ──► ReLU ──► MaxPool2D (2x2) [296,192 params]
       │
       ▼
Flatten Layer (Spatial 3D tensor -> 20,736 1D vector)                           [0 params]
       │
       ▼
Dense Classifier Head (256 neurons, L2=5e-5) ──► BatchNorm ──► ReLU ──► Dropout(0.4) [5,309,696 params]
       │
       ▼
Output Layer (1 neuron, Sigmoid Activation)                                    [257 params]
```

### Parameter Breakdown Table (Tuned CNN)

$$\text{Conv2D Params} = (k_w \times k_h \times C_{\text{in}} + 1) \times C_{\text{out}}$$

| Layer | Type | Input Dim | Output Shape | Param Formula | Param Count | Activation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`conv2d_0`** | Conv2D | 3 | $(150, 150, 32)$ | $(3 \times 3 \times 3 + 1) \times 32$ | 896 | Linear |
| **`batch_norm_0`** | BatchNorm | 32 | $(150, 150, 32)$ | $32 \times 4$ | 128 | `ReLU` |
| **`maxpool_0`** | MaxPool2D | 32 | $(75, 75, 32)$ | Non-parametric | 0 | None |
| **`conv2d_1`** | Conv2D | 32 | $(75, 75, 64)$ | $(3 \times 3 \times 32 + 1) \times 64$ | 18,496 | Linear |
| **`batch_norm_1`** | BatchNorm | 64 | $(75, 75, 64)$ | $64 \times 4$ | 256 | `ReLU` |
| **`maxpool_1`** | MaxPool2D | 64 | $(37, 37, 64)$ | Non-parametric | 0 | None |
| **`conv2d_2`** | Conv2D | 64 | $(37, 37, 128)$ | $(3 \times 3 \times 64 + 1) \times 128$ | 73,856 | Linear |
| **`batch_norm_2`** | BatchNorm | 128 | $(37, 37, 128)$ | $128 \times 4$ | 512 | `ReLU` |
| **`maxpool_2`** | MaxPool2D | 128 | $(18, 18, 128)$ | Non-parametric | 0 | None |
| **`conv2d_3`** | Conv2D | 128 | $(18, 18, 256)$ | $(3 \times 3 \times 128 + 1) \times 256$ | 295,168 | Linear |
| **`batch_norm_3`** | BatchNorm | 256 | $(18, 18, 256)$ | $256 \times 4$ | 1,024 | `ReLU` |
| **`maxpool_3`** | MaxPool2D | 256 | $(9, 9, 256)$ | Non-parametric | 0 | None |
| **`flatten`** | Flatten | $(9, 9, 256)$ | $(20736,)$ | Non-parametric | 0 | None |
| **`dense_head`** | Dense | $20,736$ | $(256,)$ | $(20,736 \times 256) + 256$ | 5,308,672 | `ReLU` |
| **`dropout`** | Dropout(0.4) | 256 | $(256,)$ | Non-parametric | 0 | None |
| **`output_layer`**| Dense | 256 | $(1,)$ | $(256 \times 1) + 1$ | 257 | `Sigmoid` |
| **Total** | | | | | **5,700,289** | |

---

## 📐 Mathematical Foundations

### 1. 2D Convolution Operation
$$S(i, j) = (\mathbf{X} * \mathbf{K})(i, j) = \sum_{m} \sum_{n} \sum_{c} \mathbf{X}(i+m, j+n, c) \mathbf{K}(m, n, c) + b$$

### 2. Feature Rescaling Normalization
$$\hat{x} = \frac{x}{255.0} \qquad x \in [0, 255] \implies \hat{x} \in [0.0, 1.0]$$

### 3. Binary Cross-Entropy Loss
$$L(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### 4. Batch Normalization
$$\hat{x}^{(k)} = \frac{x^{(k)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \qquad y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$

### 5. Global Average Pooling (GAP)
$$\text{GAP}(A^k) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} A_{ij}^k$$

### 6. Grad-CAM Class Activation Mapping
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\partial y^c}{\partial A_{ij}^k} \qquad L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

---

## ⚙️ Requirements & Environment Setup

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn pillow imagehash visualkeras
```

---

## 📂 File Structure & Execution

| File Path | Description |
| :--- | :--- |
| [`cnn_implementation_keras.ipynb`](file:///Users/shayanazmi/college/college/sem5/ann_&_deeplearning/practical_5/cnn_implementation_keras.ipynb) | Primary executed notebook with preserved cell outputs and clean academic Markdown explanations. |
| [`practical_5_readme.md`](file:///Users/shayanazmi/college/college/sem5/ann_&_deeplearning/practical_5/practical_5_readme.md) | Comprehensive lab documentation. |

---

## 💡 Key Takeaways

1. **Spatial Representation**: Conv2D layers preserve spatial 2D topology, avoiding parameter explosion inherent to MLPs.
2. **Data Pipeline Asynchrony**: `tf.data` with `.cache()` and `.prefetch(AUTOTUNE)` prevents GPU compute starvation.
3. **Data Hygiene**: Purging corrupt JPEG byte headers and filtering near-duplicates via `pHash` guarantees runtime stability and prevents train-val data leakage.
4. **Regularization Synergy**: Combining `BatchNormalization`, L2 weight decay ($5 \times 10^{-5}$), `Dropout(0.4)`, and `ReduceLROnPlateau` eliminated overfitting and boosted validation accuracy to **80.50%** (ROC-AUC **0.8843**).
5. **Explainable AI Verification**: Grad-CAM heatmaps confirm the network focuses on anatomical facial features (ears, eyes, snout) rather than background noise.
