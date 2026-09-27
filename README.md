# Lab 4.2 — RNN Basics: Forward and Backward Through Time

## 📌 Project Overview

This lab demonstrates the fundamentals of Recurrent Neural Networks (RNNs) using PyTorch. It covers building a simple RNN cell from scratch, understanding hidden-state updates, and training an RNN model to predict sine-wave values.

The project introduces how RNNs process sequential data and how neural networks learn through backpropagation.

## 🎯 Objectives

* Understand the basic architecture of a Recurrent Neural Network.

* Implement a simple RNN cell from scratch using PyTorch.

* Understand hidden states and sequential data processing.

* Generate a sine-wave dataset for prediction.

* Train an RNN model for next-value prediction.

* Calculate Mean Squared Error (MSE) loss.

* Visualize training loss and compare actual and predicted values.

* Understand the fundamentals of Backpropagation Through Time (BPTT).

## 🛠️ Technologies Used

* Python

* PyTorch

* Matplotlib

## 📂 Project Structure

```
Lab-4.2-RNN-Basics/
│
├── Lab_4_2_RNN_Basics.py
└── README.md
```

## 🧠 Lab Tasks

### 1. RNN Cell From Scratch

A custom RNN cell is implemented using PyTorch's `nn.Module`.

The cell receives an input vector and a previous hidden state, then calculates a new hidden state using the following equation:

$$
h\_t = \\tanh(W\_xx\_t + W\_hh\_{t-1} + b)  
$$

**Main components:**

* `Wx`: Input-to-hidden transformation.

* `Wh`: Hidden-to-hidden transformation.

* `bias`: Learnable bias parameter.

* `torch.tanh()`: Activation function.

* `h`: Hidden state that carries information across time steps.

The cell processes a sequence of five time steps and prints the hidden-state shape at each step.

### 2. Sine-Wave Prediction Using RNN

A small RNN model is trained to predict the next value in a sine wave.

#### Dataset Generation

A sine-wave dataset is generated using PyTorch.

* Total data points: 1,000

* Sequence length: 20

* Input features: 1

* Target: Next sine-wave value

Each input sequence contains 20 consecutive values, and the corresponding target is the following value.

#### Model Architecture

The model consists of:

1. An RNN layer with 20 hidden units.

2. A fully connected linear layer.

3. A single output representing the predicted next value.

#### Training Configuration

| Parameter       | Value              |
| --------------- | ------------------ |
| Hidden units    | 20                 |
| Sequence length | 20                 |
| Optimizer       | Adam               |
| Learning rate   | 0.01               |
| Loss function   | Mean Squared Error |
| Epochs          | 50                 |

#### Training Process

The model is trained using the following steps:

1. Pass input sequences through the RNN.

2. Extract the final hidden output.

3. Generate the next-value prediction.

4. Calculate MSE loss.

5. Clear previous gradients.

6. Perform backpropagation.

7. Update model parameters using Adam.

## 🔄 Forward and Backward Through Time

### Forward Pass

The RNN processes input data sequentially. At each time step, it updates its hidden state using the current input and the previous hidden state.

The hidden state allows the model to retain information from earlier time steps.

### Backpropagation Through Time (BPTT)

Backpropagation Through Time is the training process used by recurrent neural networks to calculate gradients across sequential time steps.

During training:

* The RNN processes the input sequence.

* The model calculates its prediction error.

* Gradients flow backward through the computational graph.

* The model updates its learnable parameters.

In this project, `loss.backward()` performs automatic gradient calculation through the unrolled RNN sequence.

## 📊 Visualizations

The project generates two graphs:

### 1. Training Loss

Displays the Mean Squared Error (MSE) loss over 50 training epochs.

The graph helps observe how the prediction error changes during training.

### 2. Actual vs Predicted Values

Compares the actual sine-wave values with the values predicted by the trained RNN model.

The graph displays the first 200 samples for easier visualization.

## 🚀 Installation and Execution

### Step 1: Install Python

Install Python 3.10 or later.

### Step 2: Install Required Libraries

Open a terminal and run:

```
pip install torch matplotlib
```

### Step 3: Run the Project

Save the Python code as `Lab_4_2_RNN_Basics.py` and execute:

```
python Lab_4_2_RNN_Basics.py
```

The program prints hidden-state shapes and training losses, then displays the visualization graphs.

## 📚 Key Learnings

* Understanding recurrent neural network architecture.

* Implementing a basic RNN cell from scratch.

* Understanding hidden-state updates.

* Processing sequential data.

* Training an RNN using PyTorch.

* Understanding automatic differentiation and BPTT.

* Applying neural networks to time-series prediction.

* Visualizing model training and predictions.

## ⚠️ Notes

* The model uses a simple sine-wave dataset for educational purposes.

* The entire dataset is used for training.

* The project does not include a separate validation or test dataset.

* Prediction accuracy depends on the model architecture and training configuration.

* The RNN cell and the sine-prediction model are separate demonstrations.

## 👨‍💻 Author

**Muhammad Suffiyan Rafi**

GitHub: [SufyanCh632](https://github.com/SufyanCh632)

## 📄 License

This project is intended for educational and learning purposes.
