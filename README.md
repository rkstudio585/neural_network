# neural_network
---
neural network: [In python](/)
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
Certainly! Here's a step-by-step guide on how to train the neural network model using `neural_network.c`.

---

# Step-by-Step Guide to Train the Neural Network Model

## Overview

This guide will walk you through the process of training a neural network model using the `neural_network.c` program. You will learn how to compile and run the training program, and how to understand the training process.

## Prerequisites

1. **Development Environment**:
   - A C compiler (e.g., `gcc`).
   - Basic command-line tools.
   - The math library (`-lm` flag for linking).

2. **Source Code**:
   - Ensure you have `neural_network.c` source code.

## 1. Prepare the Source Code

Ensure that you have the `neural_network.c` file ready. This file contains the code for defining, training, and saving the neural network model.

## 2. Compile the Program

Open a terminal and navigate to the directory where `neural_network.c` is located. Compile the source code using the following command:

```sh
gcc -o neural_network neural_network.c -lm
```

This command compiles `neural_network.c` into an executable named `neural_network`. The `-lm` flag links the math library, which is necessary for mathematical functions like `exp` and `pow`.

## 3. Review Training Configuration

### Key Parameters

- **Learning Rate (`INITIAL_LEARNING_RATE`)**:
- Determines the step size during weight updates.
- **Epochs (`EPOCHS`)**:
- Number of times the entire dataset is passed through the network.
- **Early Stopping Patience (`EARLY_STOPPING_PATIENCE`)**:
- Number of epochs to wait for improvement before stopping training.
- **Learning Rate Decay (`LEARNING_RATE_DECAY`)**:
- Rate at which the learning rate decreases over time.
- **Minimum Learning Rate (`MIN_LEARNING_RATE`)**:
- Lower bound for the learning rate.

These parameters are defined in the `neural_network.c` file. You can adjust them based on your training needs.

### Training Data

In `neural_network.c`, the `inputs` and `targets` arrays represent the training data. For a more realistic scenario, replace these arrays with your own dataset.

## 4. Run the Training Program

Execute the compiled program to start training the neural network:

```sh
./neural_network
```

### What Happens During Training

1. **Initialization**:
   - Weights and biases are initialized with random values.

2. **Training Loop**:
   - **Forward Pass**:
   - Computes the output of the network.
   - **Loss Calculation**:
   - Measures the error between predicted and actual values.
   - **Backpropagation**:
   - Updates weights and biases based on the error.

3. **Loss Monitoring**:
   - The program prints loss values and checks for improvements. If there’s no improvement for a set number of epochs, it stops early.

4. **Model Saving**:
   - After training, the model's weights and biases are saved to `trained_model.dat`.

## 5. Verify Training Output

After training completes, the program prints the following:

- **Best Loss**: The lowest loss achieved during training.
- **Loss Improvement**: Percentage improvement from the previous best loss.
- **No Improvement Count**: Number of epochs without improvement.

The trained model is saved to `trained_model.dat`. Ensure that this file is created in the same directory.

## 6. Troubleshooting

- **Compilation Errors**:
- Ensure all dependencies (e.g., math library) are correctly linked.
- **Runtime Errors**:
- Check for errors in the training data and ensure it matches the expected input format.

## Example Customization

For a different dataset or network architecture:

1. **Update `inputs` and `targets`**: Replace with your dataset.
2. **Modify Layer Sizes**: Change `HIDDEN_SIZE_1`, `HIDDEN_SIZE_2`, `HIDDEN_SIZE_3` to match your architecture.

Ensure that these modifications are reflected consistently throughout the code.

## Conclusion

Following these steps will train your neural network model and save it for future use. For detailed adjustments or specific configurations, refer to the parameters and data setup in the `neural_network.c` source code.

---

### Example 1: Basic Training with Default Configuration

This example demonstrates the basic setup with default parameters and a simple dataset.

#### `neural_network.c` (Partial)

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Constants
#define INPUT_SIZE 10
#define HIDDEN_SIZE_1 128
#define HIDDEN_SIZE_2 64
#define HIDDEN_SIZE_3 32
#define OUTPUT_SIZE 10
#define EPOCHS 100
#define INITIAL_LEARNING_RATE 0.01
#define EARLY_STOPPING_PATIENCE 10
#define LEARNING_RATE_DECAY 0.995
#define MIN_LEARNING_RATE 0.0001

// Sample training data
double inputs[INPUT_SIZE][INPUT_SIZE] = { /* Your input data here */ };
double targets[INPUT_SIZE][OUTPUT_SIZE] = { /* Your target data here */ };

// Other functions...

int main() {
    // Initialize weights, biases, and other variables
    // Train the network
    train(inputs, targets, INPUT_SIZE);
    return 0;
}
```

### Example 2: Custom Training Data

Replace the default training data with your custom dataset.

#### Custom Training Data

```c
// Define your own training data
double inputs[5][10] = {
    {0.5, 0.2, 0.1, 0.4, 0.3, 0.7, 0.8, 0.9, 0.6, 0.2},
    {0.1, 0.4, 0.7, 0.3, 0.2, 0.8, 0.5, 0.6, 0.9, 0.4},
    {0.3, 0.6, 0.2, 0.5, 0.7, 0.4, 0.8, 0.9, 0.1, 0.2},
    {0.9, 0.1, 0.3, 0.6, 0.5, 0.7, 0.2, 0.4, 0.8, 0.3},
    {0.4, 0.8, 0.6, 0.1, 0.7, 0.5, 0.9, 0.2, 0.3, 0.7}
};

double targets[5][10] = {
    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0}
};
```

### Example 3: Adjusting Training Parameters

Change learning rate, epochs, and other parameters to fit different scenarios.

#### Updated Parameters

```c
#define INPUT_SIZE 10
#define HIDDEN_SIZE_1 128
#define HIDDEN_SIZE_2 64
#define HIDDEN_SIZE_3 32
#define OUTPUT_SIZE 10
#define EPOCHS 200                  // Increased number of epochs
#define INITIAL_LEARNING_RATE 0.005 // Lower learning rate
#define EARLY_STOPPING_PATIENCE 15  // Increased patience
#define LEARNING_RATE_DECAY 0.98    // Slower decay
#define MIN_LEARNING_RATE 0.00001   // Lower minimum learning rate
```

### Example 4: Custom Activation Function

Replace the activation function with a custom one if needed.

#### Custom Activation Function

```c
double custom_activation(double x) {
    return x / (1 + fabs(x)); // Example of a custom activation function
}

void forward_pass(double* input, double* weights_hidden1, double* weights_hidden2, double* weights_hidden3,
                  double* weights_output, double* biases_hidden1, double* biases_hidden2, double* biases_hidden3,
                  double* biases_output, double* hidden1_output, double* hidden2_output, double* hidden3_output,
                  double* final_output) {
    // Hidden Layer 1
    for (int j = 0; j < HIDDEN_SIZE_1; j++) {
        hidden1_output[j] = biases_hidden1[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            hidden1_output[j] += input[i] * weights_hidden1[i * HIDDEN_SIZE_1 + j];
        }
        hidden1_output[j] = custom_activation(hidden1_output[j]); // Use custom activation
    }
    // Continue with other layers...
}
```

### Example 5: Saving and Loading Model

Ensure proper model saving and loading.

#### Saving Model

```c
void save_model(double* weights_hidden1, double* weights_hidden2, double* weights_hidden3,
                double* weights_output, double* biases_hidden1, double* biases_hidden2,
                double* biases_hidden3, double* biases_output) {
    FILE *file = fopen("trained_model.dat", "wb");
    if (file != NULL) {
        fwrite(weights_hidden1, sizeof(double), INPUT_SIZE * HIDDEN_SIZE_1, file);
        fwrite(weights_hidden2, sizeof(double), HIDDEN_SIZE_1 * HIDDEN_SIZE_2, file);
        fwrite(weights_hidden3, sizeof(double), HIDDEN_SIZE_2 * HIDDEN_SIZE_3, file);
        fwrite(weights_output, sizeof(double), HIDDEN_SIZE_3 * OUTPUT_SIZE, file);
        fwrite(biases_hidden1, sizeof(double), HIDDEN_SIZE_1, file);
        fwrite(biases_hidden2, sizeof(double), HIDDEN_SIZE_2, file);
        fwrite(biases_hidden3, sizeof(double), HIDDEN_SIZE_3, file);
        fwrite(biases_output, sizeof(double), OUTPUT_SIZE, file);
        fclose(file);
        printf("Model saved to 'trained_model.dat'.\n");
    } else {
        printf("Error saving model.\n");
    }
}
```

#### Loading Model

```c
void load_model(double* weights_hidden1, double* weights_hidden2, double* weights_hidden3,
                double* weights_output, double* biases_hidden1, double* biases_hidden2,
                double* biases_hidden3, double* biases_output) {
    FILE *file = fopen("trained_model.dat", "rb");
    if (file != NULL) {
        fread(weights_hidden1, sizeof(double), INPUT_SIZE * HIDDEN_SIZE_1, file);
        fread(weights_hidden2, sizeof(double), HIDDEN_SIZE_1 * HIDDEN_SIZE_2, file);
        fread(weights_hidden3, sizeof(double), HIDDEN_SIZE_2 * HIDDEN_SIZE_3, file);
        fread(weights_output, sizeof(double), HIDDEN_SIZE_3 * OUTPUT_SIZE, file);
        fread(biases_hidden1, sizeof(double), HIDDEN_SIZE_1, file);
        fread(biases_hidden2, sizeof(double), HIDDEN_SIZE_2, file);
        fread(biases_hidden3, sizeof(double), HIDDEN_SIZE_3, file);
        fread(biases_output, sizeof(double), OUTPUT_SIZE, file);
        fclose(file);
        printf("Model loaded from 'trained_model.dat'.\n");
    } else {
        printf("Error loading model.\n");
        exit(1);
    }
}
```

## Conclusion
These examples should help you set up, customize, and run the neural network training process effectively. Adjust the parameters, data, and functions according to your specific needs and the architecture of your neural networks 

---
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
