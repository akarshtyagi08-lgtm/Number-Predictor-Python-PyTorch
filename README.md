# 🔢 Handwritten Digit Recognition (MNIST)

A complete PyTorch pipeline to train, evaluate, and test an Artificial Neural Network on handwritten digits (0–9) using the MNIST dataset.

---

## ⚡ Key Highlights

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

* **Achieved Test Accuracy:** **`97.29%`** across 10,000 unseen test samples 🎯
* **Fast Convergence:** Reaches strong performance in just **5 epochs**
* **Device Adaptive:** Automatically routes compute to CUDA GPU when available or falls back to CPU
* **Interactive Inference:** Visualizes a single test sample with Matplotlib alongside true vs. predicted labels

---

## 🛠️ Libraries Used

* **PyTorch (`torch`, `torch.nn`, `torch.optim`)** – Building neural network layers, managing backward passes, and gradient optimization.
* **Torchvision (`torchvision.datasets`, `torchvision.transforms`)** – Downloading and transforming MNIST image tensors.
* **Matplotlib (`matplotlib.pyplot`)** – Plotting grayscale digit images with predicted labels.

---

## 🧠 Network Architecture

```text
Input (28x28 = 784) ──► Linear(784, 128) ──► ReLU ──► Linear(128, 64) ──► ReLU ──► Linear(64, 10) ──► Predictions (0-9)

 * Loss Function: nn.CrossEntropyLoss()
 * Optimizer: optim.Adam (Learning Rate = 0.001)
 * Batch Size: 64
📂 Repository Structure
├── number_detection.py    # Training loop, evaluation, and test visualization
├── mnist_model.pth        # Saved PyTorch model state_dict weights
└── README.md              # Project documentation

🚀 How to Run
 * Clone the repository:
   git clone [https://github.com/YOUR_USERNAME/Number-Prediction.git](https://github.com/YOUR_USERNAME/Number-Prediction.git)
cd Number-Prediction

 * Install required packages:
   pip install torch torchvision matplotlib

 * Train and evaluate:
   python number_detection.py

📈 Training Summary
Epoch 1/5 | Loss 327.2035
Epoch 2/5 | Loss 132.9703
Epoch 3/5 | Loss 90.1618
Epoch 4/5 | Loss 69.2706
Epoch 5/5 | Loss 54.6007

Total Accuracy: 97.29%
Model saved successfully as 'mnist_model.pth'


