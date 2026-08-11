# Practical 2 - Implement a Single Neuron Model Manually Using Python

This project demonstrates how a minimal artificial neural network — a single hidden layer with 4 neurons feeding into one output neuron — can learn the **AND logic gate** from data, using TensorFlow/Keras and NumPy.

## Overview

The AND gate is one of the simplest logical functions:

| Input 1 | Input 2 | Output (AND) |
|---------|---------|--------------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

This notebook builds a tiny neural network from scratch (using Keras' `Sequential` API), trains it on all four input combinations, and shows how its predictions evolve from random guesses (before training) to accurate AND-gate outputs (after training).

## Objective

- Understand how a neuron/network maps inputs to outputs using weights, bias, and an activation function.
- Observe the effect of training (gradient descent via the Adam optimizer) on prediction accuracy.
- Compare untrained vs. trained model behavior on a simple, well-understood logical function.

## Model Architecture

<img width="1180" height="798" alt="image" src="https://github.com/user-attachments/assets/992a1e32-0f5e-43be-942a-23656ae03c0a" />



- **Hidden layer**: 4 fully connected neurons with ReLU activation, introducing non-linearity.
- **Output layer**: 1 neuron with Sigmoid activation, squashing the output into a `[0, 1]` probability for binary classification.

## Requirements

- Python 3.x
- [TensorFlow](https://www.tensorflow.org/) (Keras API)
- [NumPy](https://numpy.org/)

Install dependencies:

```bash
pip install tensorflow numpy
```

## Notebook Structure

| Section | Description |
|---|---|
| 1. Importing Libraries | Loads TensorFlow and NumPy. |
| 2. Defining the Dataset | Creates the AND gate input matrix `X` and label vector `y`. |
| 3. Building the Model | Defines a `Sequential` model with a hidden and an output layer. |
| 4. Compiling the Model | Sets the optimizer (`adam`), loss (`binary_crossentropy`), and tracked metric (`accuracy`). |
| 5. Predictions Before Training | Shows the model's (essentially random) output prior to any learning. |
| 6. Training the Model | Trains for 10 epochs using `model.fit()`. |
| 7. Predictions After Training | Re-evaluates the model and prints the learned AND-gate predictions. |
| 8. Result | Summarizes and interprets the before/after comparison. |

## ▶ How to Run

1. Open the notebook `Practical_2_Single_Neuron_Model_ANN_Demo.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run all cells sequentially (**Runtime → Run all** in Colab, or **Cell → Run All** in Jupyter).
3. Observe the printed predictions before and after training.

## Expected Result

After training for enough epochs, the model should correctly classify all four inputs according to the AND truth table:

| Input | Expected Output | Predicted Output (trained) |
|-------|------------------|------------------------------|
| `[0, 0]` | 0 | 0 |
| `[0, 1]` | 0 | 0 |
| `[1, 0]` | 0 | 0 |
| `[1, 1]` | 1 | 1 |

> **Note:** With only 10 epochs, the model's loss may still be relatively high and predictions may not have fully converged. Increasing the number of epochs generally improves convergence toward the correct AND-gate behavior.

## Key Takeaways

- Even a very small neural network can learn simple logical functions through gradient-based optimization.
- Before training, predictions are close to random (~0.5 probability) because weights start at their initial (untrained) values.
- Training iteratively adjusts weights and biases to minimize the loss, gradually aligning predictions with the true labels.
- ReLU in the hidden layer and Sigmoid in the output layer are common choices for introducing non-linearity and producing probability-like outputs, respectively.


## Author

**Author:** Shayan Azmi  
Created as part of ANN and Deep Learning Lab practical demonstrating the implementation of a Single Neuron Model ANN.
