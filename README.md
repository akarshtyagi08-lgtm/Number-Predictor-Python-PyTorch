# 🔢 Handwritten Digit Recognition (MNIST) with PyTorch

A lightweight, high-performance Deep Learning pipeline implemented in PyTorch to train, evaluate, and test a Multi-Layer Perceptron (MLP) on the classic MNIST dataset of handwritten digits (0–9).

---

## 📌 Overview & Performance

This project demonstrates an end-to-end computer vision workflow in PyTorch: from automatic data acquisition and preprocessing pipelines, to custom neural network architecture design, GPU/CPU hardware acceleration, backpropagation, model serialization, and visual inference.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

### 🎯 Key Highlights
* **Test Accuracy:** Achieved **`97.29%`** accuracy on 10,000 unseen test images.
* **Rapid Convergence:** Loss dropped from **327.20** down to **54.60** within just **5 training epochs**.
* **Adaptive Hardware Routing:** Automatically executes on NVIDIA CUDA GPUs if available, with automatic fallback to CPU.
* **Serialized State Dict:** Exports portable learned weights to `mnist_model.pth`.

---

## 🛠️ Tech Stack & Dependencies

* **`torch` & `torch.nn`** – Defines tensor operations, layer parameters, forward computational graphs, and loss criteria.
* **`torch.optim`** – Gradient descent optimization using the Adam algorithm.
* **`torchvision` & `torchvision.transforms`** – Automatic downloading, splitting, and normalization of the MNIST dataset.
* **`torch.utils.data.DataLoader`** – Efficient multi-threaded batching, memory pinning, and dataset shuffling.
* **`matplotlib.pyplot`** – Visual inspection of single-sample predictions with grayscale rendering.

---

## 🧠 How It Works: Step-by-Step Architecture

### 1. Data Pipeline & Flattening
* The raw MNIST dataset contains $28 \times 28$ single-channel (grayscale) images with pixel values in the range $[0, 255]$.
* `transforms.ToTensor()` scales pixel intensities to float values in the range $[0.0, 1.0]$.
* Inside the training loop, each batch tensor of shape `(64, 1, 28, 28)` is dynamically flattened into a 1D vector of shape `(64, 784)` using `.view(image.size(0), -1)`.

### 2. Neural Network Design (`DigitModel`)
The model is constructed as a 3-layer fully connected feedforward network:

```text
Input Layer (784 features)
       │
       ▼
Linear Layer 1 ──► [784 ➔ 128]
       │
       ▼
Activation 1   ──► ReLU (introduces non-linearity)
       │
       ▼
Linear Layer 2 ──► [128 ➔ 64]
       │
       ▼
Activation 2   ──► ReLU
       │
       ▼
Output Layer   ──► [64 ➔ 10 logits]

 * Non-Linear Activation: Rectified Linear Units (\text{ReLU}(x) = \max(0, x)) enable the network to learn non-linear decision boundaries between ambiguous digit strokes.
 * Output Logits: The final layer outputs 10 unnormalized class scores corresponding to digits 0 through 9.
3. Optimization & Training Dynamics
 * Loss Computation (CrossEntropyLoss): Combines nn.LogSoftmax and nn.NLLLoss into a single, numerically stable loss step.
 * Optimization (Adam): Utilizes adaptive learning rates and momentum with a learning rate of \alpha = 0.001.
 * Zeroing Gradients: optimizer.zero_grad() prevents gradient accumulation across batches before running loss.backward().
📂 Repository Structure
Number-Prediction/
├── number_detection.py    # Complete pipeline: ETL, Model, Train, Test, Inference
├── mnist_model.pth        # Saved PyTorch model weights (state_dict)
└── README.md              # Detailed project documentation

⚙️ Installation & Usage
1. Clone Repository
git clone [https://github.com/YOUR_USERNAME/Number-Prediction.git](https://github.com/YOUR_USERNAME/Number-Prediction.git)
cd Number-Prediction

2. Install Required Packages
pip install torch torchvision matplotlib

3. Run Pipeline
python number_detection.py

📈 Training Logs & Benchmark
Using device: cpu (or cuda)
Epoch 1/5 | Loss 327.2035
Epoch 2/5 | Loss 132.9703
Epoch 3/5 | Loss 90.1618
Epoch 4/5 | Loss 69.2706
Epoch 5/5 | Loss 54.6007

Total Accuracy: 97.29%
Model saved successfully as 'mnist_model.pth'

User picked image index: 0
Actual Label: 7
Model Prediction: 7

🔮 Inference & Weight Loading
To load the trained weights into any separate script or API endpoint without retraining:
import torch

# Initialize architecture
model = DigitModel()

# Load saved weights
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Run prediction
with torch.no_grad():
    prediction = model(image_tensor.view(1, -1)).argmax(dim=1).item()
    print("Predicted Digit:", prediction)


