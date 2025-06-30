# MyTorch: A Minimal Deep Learning Library from Scratch

> Course: Principles of Computational Intelligence – Amirkabir University of Technology

> Semester: Spring 2024

This academic project demonstrates a hands-on implementation of a lightweight deep learning framework named MyTorch, built entirely from scratch using Python and NumPy. The goal of the project is to gain a deeper understanding of how modern neural networks operate by replicating the core functionality of PyTorch at a low level.

## Overview
MyTorch is designed to replicate the fundamental building blocks of a deep learning framework, including:
- Tensor operations with automatic differentiation
- Neural network layers: Fully Connected, Convolutional, Max/Average Pooling
- Common activation functions: ReLU, Sigmoid, Leaky ReLU, Softmax, etc.
- Loss functions: Mean Squared Error (MSE), Cross-Entropy (CE)
- Optimizer: Stochastic Gradient Descent (SGD)
- End-to-end model training and evaluation on the MNIST dataset using both MLP and CNN architectures

## Modules Implemented
**Tensor Module (tensor.py)**
- **Core class for numerical operations** and data handling using NumPy arrays
- Backbone of all computations in MyTorch

**Layer Module**
- **FullyConnected** (linear.py): Implements dense layers with weight/bias updates
- **Conv2D** (conv2d.py): Supports multi-channel 2D convolution
- **MaxPool2D / AvgPool2D**: Contains pooling layers for dimension reduction

**Activation Module**
- Implemented non-linearities: step, **sigmoid**, **relu**, **softmax**

**Loss Module**
- **MeanSquaredError**
- **CrossEntropyLoss**

**Optimizer Module**
- **SGD**: Updates parameters based on gradients and learning rate

## Tasks Completed
**Task 1: Sanity Check with Linear Regression**
- Implemented a simple single-layer linear model and trained it to match known coefficients

**Task 2: MLP on MNIST**
- Built a Multi-Layer Perceptron with ReLU activation
- Achieved **94% accuracy** on MNIST test set using only MyTorch

**Task 3: CNN on MNIST**
- Built a Convolutional Neural Network with MaxPooling and ReLU
- Achieved **93% accuracy** on MNIST test set using only the framework

## How to Run

Clone the repository:

    git clone https://github.com/AAEA132/MyTorch.git

Unzip the light version of MNIST dataset from **MNIST_light.zip** into the parent directory.

Check the training notebooks:

    MNIST-cnn.ipynb    # For MLP on MNIST
    MNIST-cnn.ipynb    # For CNN on MNIST

Run the notebooks with your own architecture and see the results!

## License
This project is licensed under the MIT License. You are free to use it in academic, personal, or research projects.
