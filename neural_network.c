#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Constants and Hyperparameters
#define INPUT_SIZE 10
#define HIDDEN_SIZE_1 128
#define HIDDEN_SIZE_2 64
#define HIDDEN_SIZE_3 32
#define OUTPUT_SIZE 10
#define EPOCHS 10000
#define INITIAL_LEARNING_RATE 0.001
#define MIN_LEARNING_RATE 0.00001
#define LEARNING_RATE_DECAY 0.99
#define DROPOUT_RATE 0.2
#define GRADIENT_CLIP_THRESHOLD 5.0
#define EARLY_STOPPING_PATIENCE 500
#define MODEL_FILENAME "trained_model.dat"

// Color Codes for Output
#define COLOR_RESET "\033[0m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_RED "\033[31m"
#define COLOR_BLUE "\033[34m"

// Activation Functions and Their Derivatives
double relu(double x) {
    return fmax(0, x);
}

double relu_derivative(double x) {
    return (x > 0) ? 1 : 0;
}

double elu(double x) {
    return (x > 0) ? x : exp(x) - 1;
}

double elu_derivative(double x) {
    return (x > 0) ? 1 : elu(x) + 1;
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double tanh_derivative(double x) {
    return 1 - pow(tanh(x), 2);
}

double swish(double x) {
    return x * sigmoid(x);
}

double swish_derivative(double x) {
    return sigmoid(x) + swish(x) * (1 - sigmoid(x));
}

// Softmax and Cross-Entropy Loss
void softmax(double* output, int size) {
    double max = output[0];
    for (int i = 1; i < size; i++) {
        if (output[i] > max) max = output[i];
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(output[i] - max);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

double cross_entropy_loss(double* predicted, double* actual, int size) {
    double loss = 0;
    for (int i = 0; i < size; i++) {
        if (actual[i] == 1) {
            loss -= log(predicted[i]);
        }
    }
    return loss;
}

// Weight Initialization
void initialize_weights(double* weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = (rand() / (double)RAND_MAX - 0.5) * 2 / sqrt(size);
    }
}

// Forward Pass
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
        hidden1_output[j] = swish(hidden1_output[j]);
    }

    // Hidden Layer 2
    for (int k = 0; k < HIDDEN_SIZE_2; k++) {
        hidden2_output[k] = biases_hidden2[k];
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
            hidden2_output[k] += hidden1_output[j] * weights_hidden2[j * HIDDEN_SIZE_2 + k];
        }
        hidden2_output[k] = relu(hidden2_output[k]);
    }

    // Hidden Layer 3
    for (int l = 0; l < HIDDEN_SIZE_3; l++) {
        hidden3_output[l] = biases_hidden3[l];
        for (int k = 0; k < HIDDEN_SIZE_2; k++) {
            hidden3_output[l] += hidden2_output[k] * weights_hidden3[k * HIDDEN_SIZE_3 + l];
        }
        hidden3_output[l] = elu(hidden3_output[l]);
    }

    // Output Layer
    for (int m = 0; m < OUTPUT_SIZE; m++) {
        final_output[m] = biases_output[m];
        for (int l = 0; l < HIDDEN_SIZE_3; l++) {
            final_output[m] += hidden3_output[l] * weights_output[l * OUTPUT_SIZE + m];
        }
    }
    softmax(final_output, OUTPUT_SIZE);
}

// Backpropagation
void backpropagate(double* input, double* hidden1_output, double* hidden2_output, double* hidden3_output,
                   double* final_output, double* actual_output, double* weights_hidden1, double* weights_hidden2,
                   double* weights_hidden3, double* weights_output, double* biases_hidden1, double* biases_hidden2,
                   double* biases_hidden3, double* biases_output, double learning_rate) {
    // Output Layer Gradients
    double output_gradient[OUTPUT_SIZE];
    for (int m = 0; m < OUTPUT_SIZE; m++) {
        output_gradient[m] = final_output[m] - actual_output[m];
    }

    // Update weights and biases for Output Layer
    for (int l = 0; l < HIDDEN_SIZE_3; l++) {
        for (int m = 0; m < OUTPUT_SIZE; m++) {
            weights_output[l * OUTPUT_SIZE + m] -= learning_rate * output_gradient[m] * hidden3_output[l];
        }
    }
    for (int m = 0; m < OUTPUT_SIZE; m++) {
        biases_output[m] -= learning_rate * output_gradient[m];
    }

    // Hidden Layer 3 Gradients
    double hidden3_gradient[HIDDEN_SIZE_3];
    for (int l = 0; l < HIDDEN_SIZE_3; l++) {
        hidden3_gradient[l] = 0;
        for (int m = 0; m < OUTPUT_SIZE; m++) {
            hidden3_gradient[l] += output_gradient[m] * weights_output[l * OUTPUT_SIZE + m];
        }
        hidden3_gradient[l] *= elu_derivative(hidden3_output[l]);
    }

    // Update weights and biases for Hidden Layer 3
    for (int k = 0; k < HIDDEN_SIZE_2; k++) {
        for (int l = 0; l < HIDDEN_SIZE_3; l++) {
            weights_hidden3[k * HIDDEN_SIZE_3 + l] -= learning_rate * hidden3_gradient[l] * hidden2_output[k];
        }
    }
    for (int l = 0; l < HIDDEN_SIZE_3; l++) {
        biases_hidden3[l] -= learning_rate * hidden3_gradient[l];
    }

    // Hidden Layer 2 Gradients
    double hidden2_gradient[HIDDEN_SIZE_2];
    for (int k = 0; k < HIDDEN_SIZE_2; k++) {
        hidden2_gradient[k] = 0;
        for (int l = 0; l < HIDDEN_SIZE_3; l++) {
            hidden2_gradient[k] += hidden3_gradient[l] * weights_hidden3[k * HIDDEN_SIZE_3 + l];
        }
        hidden2_gradient[k] *= relu_derivative(hidden2_output[k]);
    }

    // Update weights and biases for Hidden Layer 2
    for (int j = 0; j < HIDDEN_SIZE_1; j++) {
        for (int k = 0; k < HIDDEN_SIZE_2; k++) {
            weights_hidden2[j * HIDDEN_SIZE_2 + k] -= learning_rate * hidden2_gradient[k] * hidden1_output[j];
        }
    }
    for (int k = 0; k < HIDDEN_SIZE_2; k++) {
        biases_hidden2[k] -= learning_rate * hidden2_gradient[k];
    }

    // Hidden Layer 1 Gradients
    double hidden1_gradient[HIDDEN_SIZE_1];
    for (int j = 0; j < HIDDEN_SIZE_1; j++) {
        hidden1_gradient[j] = 0;
        for (int k = 0; k < HIDDEN_SIZE_2; k++) {
            hidden1_gradient[j] += hidden2_gradient[k] * weights_hidden2[j * HIDDEN_SIZE_2 + k];
        }
        hidden1_gradient[j] *= swish_derivative(hidden1_output[j]);
    }

    // Update weights and biases for Hidden Layer 1
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
            weights_hidden1[i * HIDDEN_SIZE_1 + j] -= learning_rate * hidden1_gradient[j] * input[i];
        }
    }
    for (int j = 0; j < HIDDEN_SIZE_1; j++) {
        biases_hidden1[j] -= learning_rate * hidden1_gradient[j];
    }
}

// Training Process
void train(double inputs[INPUT_SIZE][INPUT_SIZE], double targets[INPUT_SIZE][OUTPUT_SIZE], int num_samples) {
    double weights_hidden1[INPUT_SIZE * HIDDEN_SIZE_1];
    double weights_hidden2[HIDDEN_SIZE_1 * HIDDEN_SIZE_2];
    double weights_hidden3[HIDDEN_SIZE_2 * HIDDEN_SIZE_3];
    double weights_output[HIDDEN_SIZE_3 * OUTPUT_SIZE];
    double biases_hidden1[HIDDEN_SIZE_1] = {0};
    double biases_hidden2[HIDDEN_SIZE_2] = {0};
    double biases_hidden3[HIDDEN_SIZE_3] = {0};
    double biases_output[OUTPUT_SIZE] = {0};
    
    // Initialize weights
    initialize_weights(weights_hidden1, INPUT_SIZE * HIDDEN_SIZE_1);
    initialize_weights(weights_hidden2, HIDDEN_SIZE_1 * HIDDEN_SIZE_2);
    initialize_weights(weights_hidden3, HIDDEN_SIZE_2 * HIDDEN_SIZE_3);
    initialize_weights(weights_output, HIDDEN_SIZE_3 * OUTPUT_SIZE);

    double hidden1_output[HIDDEN_SIZE_1];
    double hidden2_output[HIDDEN_SIZE_2];
    double hidden3_output[HIDDEN_SIZE_3];
    double final_output[OUTPUT_SIZE];

    double learning_rate = INITIAL_LEARNING_RATE;
    double best_loss = INFINITY;
    int patience = 0;
    int no_improvement_count = 0;
    double previous_best_loss = INFINITY;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0;
        for (int i = 0; i < num_samples; i++) {
            forward_pass(inputs[i], weights_hidden1, weights_hidden2, weights_hidden3, weights_output,
                         biases_hidden1, biases_hidden2, biases_hidden3, biases_output,
                         hidden1_output, hidden2_output, hidden3_output, final_output);
            double sample_loss = cross_entropy_loss(final_output, targets[i], OUTPUT_SIZE);
            total_loss += sample_loss;
            backpropagate(inputs[i], hidden1_output, hidden2_output, hidden3_output, final_output, targets[i],
                          weights_hidden1, weights_hidden2, weights_hidden3, weights_output,
                          biases_hidden1, biases_hidden2, biases_hidden3, biases_output, learning_rate);
        }

        total_loss /= num_samples;

        if (total_loss < best_loss) {
            no_improvement_count = 0;
            best_loss = total_loss;
            printf(COLOR_GREEN "Epoch %d: Loss improved to %.6f\n" COLOR_RESET, epoch + 1, total_loss);
        } else {
            no_improvement_count++;
            printf(COLOR_YELLOW "Epoch %d: No improvement, patience %d/%d\n" COLOR_RESET, epoch + 1, no_improvement_count, EARLY_STOPPING_PATIENCE);
            if (no_improvement_count >= EARLY_STOPPING_PATIENCE) {
                printf(COLOR_RED "Early stopping at epoch %d\n" COLOR_RESET, epoch + 1);
                break;
            }
        }

        // Decay learning rate
        learning_rate = fmax(learning_rate * LEARNING_RATE_DECAY, MIN_LEARNING_RATE);
    }

    double loss_improvement = (previous_best_loss - best_loss) / previous_best_loss * 100;
    printf(COLOR_BLUE "Training complete. Best Loss: %.6f\n" COLOR_RESET, best_loss);
    printf(COLOR_BLUE "Loss Improvement: %.2f%%\n" COLOR_RESET, loss_improvement);
    printf(COLOR_BLUE "No Improvement Count: %d\n" COLOR_RESET, no_improvement_count);

    // Save the trained model to file
    FILE *file = fopen(MODEL_FILENAME, "wb");
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
        printf(COLOR_BLUE "Model saved to '%s'\n" COLOR_RESET, MODEL_FILENAME);
    } else {
        printf(COLOR_RED "Failed to save model to file.\n" COLOR_RESET);
    }
}

int main() {
    srand(42); // For reproducibility

    // Example XOR input and output
    double inputs[INPUT_SIZE][INPUT_SIZE] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    double targets[INPUT_SIZE][OUTPUT_SIZE] = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    };

    train(inputs, targets, 10);
    return 0;
}
