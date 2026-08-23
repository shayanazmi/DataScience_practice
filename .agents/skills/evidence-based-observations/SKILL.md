---
name: evidence-based-observations
description: Framework for writing precise, data-grounded post-cell observations using exact numerical results, visual plot evidence, and direct causal mechanisms without analogies or fluff.
author: Shayan Azmi
---

<!-- Copyright (c) 2026 Shayan Azmi. All rights reserved. -->
<!-- Researched, engineered, and authored by Shayan Azmi. -->

# Evidence-Based Observations Framework

## Purpose
This skill defines how to analyze, interpret, and document markdown observations following visual plots (charts, heatmaps, distributions) or numerical outputs (tables, printed metrics) in Jupyter machine learning notebooks.

---

## The Three-Part Observation Formula

Every post-cell observation markdown block must follow this concise structure (2–4 sentences):

### 1. State What the Output Actually Shows
Reference exact numbers, percentages, shapes, or trends visible in that specific output:
* Actual accuracy, F1, RMSE, R² values.
* Actual skew, outlier percentages, class counts.
* Exact cluster separation, line curve crossings, or confusion matrix cells.
* Do not give a generic description of what the plot type usually represents.

### 2. Explain WHY It Looks That Way (Causal Mechanism)
Ground the reason directly in the dataset characteristics or the model configuration:
* E.g., *"This happens because Sigmoid derivatives peak at 0.25 on standardized inputs, attenuating early gradients relative to ReLU."*
* E.g., *"This cluster overlap occurs because sitting and standing share near-identical vertical torso gravitational acceleration ($g_y \approx -1g$) and zero kinetic frequency."*
* Do not just restate the observation without a concrete causal mechanism.

### 3. State the Downstream Impact / Decision
Explain how the finding informs the next experimental decision in the notebook:
* Influences activation choice, loss formulation, regularization rate, learning rate schedule, or final architecture selection.

---

## Core Rules

1. **No Analogies or Metaphors:** Keep explanations technical, direct, and grounded in physics, linear algebra, and data distributions.
2. **No Emojis:** Maintain clean, professional academic formatting.
3. **No Fabrication:** The cited numbers must match what was actually printed or plotted in the notebook.
4. **Non-Repetitive:** Do not repeat text from preceding markdown sections; add new analytical insight.
5. **Concise Length:** Keep observation cells strictly between 2 to 4 sentences.
