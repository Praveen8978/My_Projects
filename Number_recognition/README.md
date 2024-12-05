
# Neural Network using PyTorch on CUDA for MNIST Classification

This project demonstrates how to build and train a feedforward neural network (NN) using PyTorch. The model is trained on the MNIST dataset, which consists of handwritten digits. The network achieves an accuracy of **97.76%** on the test data using CUDA for faster computations.

## Key Features
- **Feedforward Neural Network Architecture**: The model consists of an input layer, one hidden layer with 256 neurons, and an output layer with 10 neurons (representing the 10 digits).
- **CUDA Support**: The training and evaluation are done on the GPU (CUDA-enabled) for faster performance.
- **MNIST Dataset**: The dataset is loaded directly from PyTorch's `torchvision.datasets`, and the model is trained and evaluated on this dataset.
- **Achieved Accuracy**: The model achieves an accuracy of **97.76%** on the test set.

## Requirements
- Python 3.6 or later
- PyTorch (with CUDA support)
- torchvision
- numpy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pytorch-mnist-feedforward.git
   cd pytorch-mnist-feedforward
   ```

2. Install the required dependencies:
   ```bash
   pip install torch torchvision numpy
   ```

3. If you have a CUDA-compatible GPU, ensure that PyTorch is configured to use CUDA. Otherwise, the code will run on the CPU.

This script will:
- Load the MNIST dataset.
- Train the feedforward neural network model.
- Display the training loss and accuracy during training.

## Model Architecture
The architecture of the model is as follows:
- Input Layer: 784 neurons (28x28 pixels in MNIST images)
- Hidden Layer: 256 neurons with ReLU activation
- Output Layer: 10 neurons (corresponding to digits 0-9)
- The model uses CrossEntropyLoss for the classification task and the Adam optimizer.

## References
- Original code for the model structure: [PyTorch Tutorial - Feedforward Neural Network](https://github.com/patrickloeber/pytorchTutorial/blob/master/13_feedforward.py)

