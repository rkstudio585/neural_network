#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Constants
#define INPUT_SIZE 10
#define HIDDEN_SIZE_1 128
#define HIDDEN_SIZE_2 64
#define HIDDEN_SIZE_3 32
#define OUTPUT_SIZE 10
#define MODEL_FILENAME "trained_model.dat"

// Color Codes for Output
#define COLOR_RESET "\033[0m"
#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"

// Activation Functions
double relu(double x) {
    return fmax(0, x);
}

double elu(double x) {
    return (x > 0) ? x : exp(x) - 1;
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double swish(double x) {
    return x * sigmoid(x);
}

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

// Load Model from File
void load_model(double* weights_hidden1, double* weights_hidden2, double* weights_hidden3,
                 double* weights_output, double* biases_hidden1, double* biases_hidden2,
                 double* biases_hidden3, double* biases_output) {
    FILE *file = fopen(MODEL_FILENAME, "rb");
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
        printf(COLOR_GREEN "Model loaded successfully from '%s'.\n" COLOR_RESET, MODEL_FILENAME);
    } else {
        printf(COLOR_RED "Failed to load model from '%s'.\n" COLOR_RESET, MODEL_FILENAME);
        exit(1);
    }
}

// Forward Pass for Inference
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

// Main Function for Inference
int main() {
    double weights_hidden1[INPUT_SIZE * HIDDEN_SIZE_1];
    double weights_hidden2[HIDDEN_SIZE_1 * HIDDEN_SIZE_2];
    double weights_hidden3[HIDDEN_SIZE_2 * HIDDEN_SIZE_3];
    double weights_output[HIDDEN_SIZE_3 * OUTPUT_SIZE];
    double biases_hidden1[HIDDEN_SIZE_1];
    double biases_hidden2[HIDDEN_SIZE_2];
    double biases_hidden3[HIDDEN_SIZE_3];
    double biases_output[OUTPUT_SIZE];
    
    load_model(weights_hidden1, weights_hidden2, weights_hidden3,
               weights_output, biases_hidden1, biases_hidden2,
               biases_hidden3, biases_output);

    double hidden1_output[HIDDEN_SIZE_1];
    double hidden2_output[HIDDEN_SIZE_2];
    double hidden3_output[HIDDEN_SIZE_3];
    double final_output[OUTPUT_SIZE];

    // Example input for testing
    double test_input[INPUT_SIZE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}; // Modify as needed

    forward_pass(test_input, weights_hidden1, weights_hidden2, weights_hidden3, weights_output,
                 biases_hidden1, biases_hidden2, biases_hidden3, biases_output,
                 hidden1_output, hidden2_output, hidden3_output, final_output);

    printf("Inference result:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Class %d: %.4f\n", i, final_output[i]);
    }

    return 0;
}
