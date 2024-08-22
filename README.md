# neural_network
---
neural network: [In python](/)
---
Certainly! Below is a detailed and user-friendly documentation for both `neural_network.c` and `run_inference.c`, including explanations of each function, how to run the programs, and other essential information.

---

# Neural Network and Run Inference Documentation

## Overview

This project consists of two programs:

1. **`neural_network.c`** - Trains a deep neural network model and saves it to a file.
2. **`run_inference.c`** - Loads the trained model from a file and performs inference on new data.

### Project Structure

- `neural_network.c` - Contains code for defining, training, and saving a neural network model.
- `run_inference.c` - Contains code for loading the trained model and performing inference.

## `neural_network.c`

### Purpose

This program trains a deep neural network on a dataset and saves the model's weights and biases to a file named `trained_model.dat`.

### Functions

#### 1. 
```c
void initialize_weights(double* weights, int size)
```

- **Purpose**: Initializes the weights of the neural network with random values.
- **Parameters**:
  - `weights`: Array to store initialized weights.
  - `size`: Number of weights to initialize.
- **Called**: Once per layer during the training setup.

#### 2. 
```c
void forward_pass(double* input, double* weights_hidden1, double* weights_hidden2, double* weights_hidden3, double* weights_output, double* biases_hidden1, double* biases_hidden2, double* biases_hidden3, double* biases_output, double* hidden1_output, double* hidden2_output, double* hidden3_output, double* final_output)
```

- **Purpose**: Computes the forward pass through the network to generate outputs.
- **Parameters**:
  - `input`: Input data for the network.
  - `weights_*`: Weights for each layer.
  - `biases_*`: Biases for each layer.
  - `hidden*_output`: Arrays to store outputs of hidden layers.
  - `final_output`: Array to store the final output.
- **Called**: Once per training sample and during inference.

#### 3. 
```c
double cross_entropy_loss(double* predicted, double* actual, int size)
```

- **Purpose**: Calculates the cross-entropy loss between the predicted and actual values.
- **Parameters**:
  - `predicted`: Predicted output probabilities.
  - `actual`: True target values.
  - `size`: Number of output classes.
- **Returns**: Cross-entropy loss value.
- **Called**: Once per training sample.

#### 4. 
```c
void backpropagate(double* input, double* hidden1_output, double* hidden2_output, double* hidden3_output, double* final_output, double* target, double* weights_hidden1, double* weights_hidden2, double* weights_hidden3, double* weights_output, double* biases_hidden1, double* biases_hidden2, double* biases_hidden3, double* biases_output, double learning_rate)
```

- **Purpose**: Performs backpropagation to update the weights and biases of the network.
- **Parameters**:
  - `input`, `hidden*_output`, `final_output`, `target`: Data used in training.
  - `weights_*`, `biases_*`: Weights and biases to update.
  - `learning_rate`: Rate at which weights are updated.
- **Called**: Once per training sample.

#### 5. 
```c
void train(double inputs[INPUT_SIZE][INPUT_SIZE], double targets[INPUT_SIZE][OUTPUT_SIZE], int num_samples)
```

- **Purpose**: Manages the training process of the neural network.
- **Parameters**:
  - `inputs`: Array of training inputs.
  - `targets`: Array of training targets.
  - `num_samples`: Number of training samples.
- **Called**: Once, to start the training process.

### Variables

- **Weights and Biases Arrays**: Defined globally within `train` function and used for each layer of the network. These values can be changed based on the network's architecture.

### Execution

1. **Compile**:
   ```sh
   gcc -o neural_network neural_network.c -lm
   ```

2. **Run**:
   ```sh
   ./neural_network
   ```

   This will train the network and save the model to `trained_model.dat`.

## `run_inference.c`

### Purpose

This program loads the trained model from `trained_model.dat` and performs inference on new input data.

### Functions

#### 1. 
```c
void load_model(double* weights_hidden1, double* weights_hidden2, double* weights_hidden3, double* weights_output, double* biases_hidden1, double* biases_hidden2, double* biases_hidden3, double* biases_output)
```

- **Purpose**: Loads the trained weights and biases from a file.
- **Parameters**:
  - `weights_*`, `biases_*`: Arrays to store the loaded model parameters.
- **Called**: Once to load the model parameters.

#### 2. 
```c
void forward_pass(double* input, double* weights_hidden1, double* weights_hidden2, double* weights_hidden3, double* weights_output, double* biases_hidden1, double* biases_hidden2, double* biases_hidden3, double* biases_output, double* hidden1_output, double* hidden2_output, double* hidden3_output, double* final_output)
```

- **Purpose**: Performs the forward pass to generate predictions from new input data.
- **Parameters**:
  - `input`, `weights_*`, `biases_*`, `hidden*_output`, `final_output`: As described in `neural_network.c`.
- **Called**: Once per inference.

### Variables

- **Weights and Biases Arrays**: These are loaded from the file and used for making predictions. They should match the sizes used during training.

### Execution

1. **Compile**:
   ```sh
   gcc -o run_inference run_inference.c -lm
   ```

2. **Run**:
   ```sh
   ./run_inference
   ```

   This will load the trained model and perform inference on a predefined test input.

## Summary

- **`neural_network.c`** trains a deep neural network and saves it.
- **`run_inference.c`** loads the saved model and performs predictions.
- Modify the `test_input` in `run_inference.c` as needed to test different scenarios.
- Ensure that the model architecture (layer sizes) and input data match between training and inference.
---
