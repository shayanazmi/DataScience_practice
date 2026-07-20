# Practical 3 - Implement a Feedforward Neural Network Using PyTorch

This project demonstrates how a **Feedforward Neural Network (FNN)** is built from scratch using **PyTorch** to learn the **AND logic gate** — a classic binary classification problem in neural network education.

## 📌 Overview

The AND gate is one of the simplest logical functions:

| Input 1 | Input 2 | Output (AND) |
|---------|---------|--------------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

This notebook defines a feedforward neural network class using PyTorch's `nn.Module`, trains it on all four AND gate input combinations, and demonstrates how the model learns to correctly classify each input after 100 epochs of training.

## 🎯 Objective

- Understand the architecture of a feedforward neural network with input, hidden, and output layers.
- Implement forward pass, loss computation, and backpropagation using PyTorch.
- Observe how the Adam optimizer adjusts weights and biases to minimize Binary Cross-Entropy Loss.
- Compare raw predicted probabilities to final classified outputs using a 0.5 threshold.

## 🧠 Model Architecture

```
Input (2 features)
      │
      ▼
Linear Layer (2 → 4 neurons, ReLU activation)   ← Hidden layer
      │
      ▼
Linear Layer (4 → 1 neuron, Sigmoid activation) ← Output layer
```

- **Hidden layer**: 4 fully connected neurons with ReLU activation, introducing non-linearity.
- **Output layer**: 1 neuron with Sigmoid activation, squashing the output into a `[0, 1]` probability for binary classification.

## 🛠️ Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/)

Install dependencies:

```bash
pip install torch
```

## 📂 Notebook Structure

| Section | Description |
|---|---|
| 1. Importing Libraries | Loads PyTorch and its submodules (`nn`, `optim`). |
| 2. Defining the Dataset | Creates the AND gate input tensor `X` and label tensor `y`. |
| 3. Building the Model | Defines a `FeedforwardNN` class extending `nn.Module` with two linear layers. |
| 4. Compiling the Model | Sets the loss function (`BCELoss`) and optimizer (`Adam`, lr=0.01). |
| 5. Training the Model | Trains for 100 epochs using forward pass, loss computation, and backpropagation. |
| 6. Predictions | Evaluates the trained model and prints probability and class for each input. |

## ▶️ How to Run

1. Open the notebook `Practical_3_Feedforward_Neural_Network.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run all cells sequentially (**Runtime → Run all** in Colab, or **Cell → Run All** in Jupyter).
3. Observe the printed predictions after training.

## ✅ Expected Result

After training for 100 epochs, the model should correctly classify all four inputs according to the AND truth table:

| Input | Expected Output | Predicted Probability | Predicted Class |
|-------|-----------------|----------------------|-----------------|
| `[0.0, 0.0]` | 0 | ~0.2082 | 0 |
| `[0.0, 1.0]` | 0 | ~0.2371 | 0 |
| `[1.0, 0.0]` | 0 | ~0.3124 | 0 |
| `[1.0, 1.0]` | 1 | ~0.7344 | 1 |

> **Note:** Predicted probabilities may vary slightly between runs due to random weight initialization. The final class (0 or 1) is determined by a threshold of 0.5.

## 📖 Key Takeaways

- A feedforward neural network passes data in one direction — from input to output — with no cycles or loops.
- PyTorch's `nn.Module` makes it easy to define custom network architectures using object-oriented programming.
- `BCELoss` (Binary Cross-Entropy Loss) is the standard loss function for binary classification tasks.
- The Adam optimizer adapts the learning rate for each parameter, enabling faster and more stable convergence compared to plain SGD.
- ReLU in the hidden layer introduces non-linearity, allowing the network to learn complex decision boundaries.
- Sigmoid in the output layer maps the final value to a probability between 0 and 1, suitable for binary classification.


## Author

Created as part of ANN and Deep Learning Lab practical demonstrating the implementation of a Feedforward Neural Network using PyTorch.
