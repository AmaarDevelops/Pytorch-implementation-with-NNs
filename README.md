
# 🤖 PyTorch Deep Learning Fundamentals and Applications

This project serves as a comprehensive introduction and practical guide to building and training neural networks using the **PyTorch** framework. It progresses from foundational tensor operations and manual gradient calculations to implementing and testing two standard models: a **Fully Connected Network (FCN) on MNIST** and a **Convolutional Neural Network (CNN) on CIFAR-10**.

## 🚀 Project Structure & Progression

The single script covers four major sections, demonstrating the evolution of a PyTorch workflow:

1.  **Tensor & Autograd Fundamentals:** Basic tensor arithmetic, views, and the core concept of automatic differentiation (`requires_grad=True` and `.backward()`).
2.  **Linear Regression:** Implementation of linear regression using both a manual gradient descent loop and PyTorch's modular components (`nn.Module`, `nn.MSELoss`, `torch.optim.SGD`).
3.  **Fully Connected Network (FCN):** Training an $\text{Input}(784) \rightarrow \text{Hidden}(500) \rightarrow \text{Output}(10)$ network on the **MNIST** handwriting dataset.
4.  **Convolutional Neural Network (CNN):** Training a simple, multi-layer CNN on the **CIFAR-10** image dataset.

---

## ⚙️ Setup and Prerequisites

### 1. Requirements

This project requires PyTorch and its associated vision libraries. Install them using `pip`:

``bash

pip install torch torchvision numpy matplotlib

2. Dataset Download
The script automatically downloads both the MNIST and CIFAR-10 datasets to a local ./data directory during the first run.

3. Execution
Save the entire code block into a file named pytorch_deep_learning.py.

Bash:
python pytorch_deep_learning.py

🧠 Model Architectures & Results

1. MNIST FCN
This model uses the high-level nn.Module to classify 28×28 grayscale images.

Architecture: Linear(784→500)→ReLU→Linear(500→10)

Loss: nn.CrossEntropyLoss()

Optimizer: torch.optim.Adam

Training Time: 2 Epochs

Expected Accuracy: ≈94%−96%

2. CIFAR-10 CNN
This model handles 32×32 color images (3 channels) and is built using a sequence of Convolutional and Pooling layers.

Architecture:

Conv(3→32)

Pool

Conv(32→64)

Pool

Conv(64→64)

Flatten→Dense(64×4×4→64)→Dense(64→10)

Preprocessing: ToTensor and Normalize to the range [−1,1].

Loss: nn.CrossEntropyLoss()

Optimizer: torch.optim.Adam

Training Time: 10 Epochs

Expected Accuracy: ≈60%−70% (A simple CNN architecture on CIFAR-10)

🛠️ Key PyTorch Concepts Demonstrated

Concept	PyTorch Implementation

Model Definition	Subclassing nn.Module

Training Loop	Explicit steps: loss.backward(), optimizer.step(), optimizer.zero_grad()

Data Handling	Using torchvision.datasets and torch.utils.data.DataLoader

Inference	Using with torch.no_grad(): to disable gradient tracking during testing
