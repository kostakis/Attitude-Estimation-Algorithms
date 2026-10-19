#include "LSTMNetworkNoAttention.h"
#include "Utils.hpp"
#include "csvreader.hpp"

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>

#include <torch/torch.h>

using torch::Tensor;

void setRandomSeeds(unsigned int seed) {
    // C++
    std::srand(seed);

    // PyTorch/LibTorch
    torch::manual_seed(seed);

    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed(seed);
        torch::cuda::manual_seed_all(seed);
    }
}

int main(int argc, char* argv[]) {
    int numEpochs = 300;  // Default value
    unsigned int randomSeed = 42;  // Default seed for reproducibility

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-epochs" || arg == "--epochs") && i + 1 < argc) {
            numEpochs = std::atoi(argv[i + 1]);
            i++;  // Skip next argument
        } else if ((arg == "-seed" || arg == "--seed") && i + 1 < argc) {
            randomSeed = std::atoi(argv[i + 1]);
            i++;  // Skip next argument
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -epochs N, --epochs N   Number of training epochs (default: 300)" << std::endl;
            std::cout << "  -seed N, --seed N       Random seed for reproducibility (default: 42)" << std::endl;
            std::cout << "  -h, --help              Show this help message" << std::endl;
            return 0;
        } else {
            std::cout << "Wrong argument exiting \n";
            return 0;
        }
    }

    const int lookbackWindow = 50;
    const int windowSize = 30;
    const int NUM_INPUT_FEATURES = 9;   // 9 columns in data: all_combined_reordered.csv
    const int NUM_OUTPUT_FEATURES = 3;  // Predict 3 angles (roll, pitch, yaw)
    const int hiddenStateSize = 128;

    // Set random seeds for reproducibility
    setRandomSeeds(randomSeed);

    std::cout << "LSTM Multi-Step Ahead Prediction" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Lookback window (K): " <<  lookbackWindow << " timesteps" << std::endl;
    std::cout << "  Prediction horizon (N): " << windowSize << " timesteps" << std::endl;
    std::cout << "  Input features: " << NUM_INPUT_FEATURES << std::endl;
    std::cout << "  Output features: " << NUM_OUTPUT_FEATURES << std::endl;
    std::cout << "  Epochs: " << numEpochs << std::endl;
    std::cout << "  Random seed: " << randomSeed << std::endl;
    std::cout << std::endl;

    CsvReader datasetReader("Data/Quadcopter_Datasets/all_combined_reordered.csv");
    datasetReader.read();
    datasetReader.printStats();

    Eigen::MatrixXd dataset = datasetReader.getEigenData();
    int totalSamples = dataset.rows();

    int trainingSamples = totalSamples - windowSize - lookbackWindow +1;  // Remove windowSize from the training sample
    std::cout << "Number of training samples: " << trainingSamples  << " after removing windowSize and loopback" << std::endl;

    // Convert dataset to tensor
    Tensor datasetTensor = LSTMNetwork::eigenToTensor(dataset);
    std::cout << "Converted eigen matrix to Tensor" << std::endl;
    std::cout << "Tensor shape: " << datasetTensor.sizes() << std::endl;

    // Normalize the data ...

    // 1. Calculate mean and standar deviation
    Tensor featureMean = datasetTensor.mean(/*dim=*/0, /*keepdim=*/true);
    Tensor featureStd = datasetTensor.std(/*dim=*/0, /*unbiased=*/true, /*keepdim=*/true);

    // Add a tiny epsilon to std to prevent division by zero in case a column is perfectly constant
    featureStd = featureStd + 1e-8;

    // 2. Scale the entire dataset: (x - mean) / std
    Tensor scaledDatasetTensor = (datasetTensor - featureMean) / featureStd;
    std::cout << "Data standardized successfully." << std::endl;

    // Saved to be used later for evalatuion
    Tensor targetMean = featureMean.slice(/*dim=*/1, /*start=*/0, /*end=*/NUM_OUTPUT_FEATURES);
    Tensor targetStd = featureStd.slice(/*dim=*/1, /*start=*/0, /*end=*/NUM_OUTPUT_FEATURES);

    ////////////////////////


    // Pre-allocate X and y tensors
    auto options = torch::TensorOptions().dtype(torch::kDouble);
    Tensor X = torch::zeros({trainingSamples, lookbackWindow, NUM_INPUT_FEATURES}, options);
    Tensor y = torch::zeros({trainingSamples, windowSize, NUM_OUTPUT_FEATURES}, options);

    // Angles tensor containg the ground truth values
    Tensor anglesTensor = scaledDatasetTensor.slice(1, 0, 3); // 0-1-2 columns have the thuth values
    std::cout << "Angles tensor shape: " << anglesTensor.sizes() << std::endl;
    // [3397, 3]

    // Create sequences
    for (int i = 0; i < trainingSamples; i++) {
        X[i] = scaledDatasetTensor.slice(0, i, i + lookbackWindow);
        
        int predStart = i + lookbackWindow;
        y[i] = anglesTensor.slice(0, predStart, predStart + windowSize);
    }

    // Shuffle all windows before splitting (mix data from all flights)
    std::vector<int> shuffleIndices(trainingSamples);
    for (int i = 0; i < trainingSamples; i++) shuffleIndices[i] = i;

    std::mt19937 shuffleRng(randomSeed);
    std::shuffle(shuffleIndices.begin(), shuffleIndices.end(), shuffleRng);

    // Reorder X and y using shuffled indices
    Tensor X_shuffled = torch::zeros_like(X);
    Tensor y_shuffled = torch::zeros_like(y);
    for (int i = 0; i < trainingSamples; i++) {
        X_shuffled[i] = X[shuffleIndices[i]];
        y_shuffled[i] = y[shuffleIndices[i]];
    }

    // Train/Validation/Test split (80/10/10)
    int numTrain = static_cast<int>(trainingSamples * 0.8);
    int numVal = static_cast<int>(trainingSamples * 0.1);
    int numTest = trainingSamples - numTrain - numVal;

    Tensor X_train = X_shuffled.slice(0, 0, numTrain);
    Tensor y_train = y_shuffled.slice(0, 0, numTrain);
    Tensor X_val = X_shuffled.slice(0, numTrain, numTrain + numVal);
    Tensor y_val = y_shuffled.slice(0, numTrain, numTrain + numVal);
    Tensor X_test = X_shuffled.slice(0, numTrain + numVal, trainingSamples);
    Tensor y_test = y_shuffled.slice(0, numTrain + numVal, trainingSamples);

    std::cout << std::endl;
    std::cout << "Data Shapes" << std::endl;
    std::cout << "X (inputs):  " << X.sizes() << std::endl;
    std::cout << "y (targets): " << y.sizes() << std::endl;
    std::cout << "Windows shuffled before splitting (seed: " << randomSeed << ")" << std::endl;
    std::cout << "Train: " << numTrain << " | Val: " << numVal << " | Test: " << numTest << " (80/10/10 split)" << std::endl;
    std::cout << std::endl;

    // Model Creation
    int outputSize = windowSize * NUM_OUTPUT_FEATURES; // Output = how many timesteps to predict * features
    auto model = std::make_shared<LSTMNetwork>(NUM_INPUT_FEATURES, hiddenStateSize, outputSize);

    std::cout << "Model Created" << std::endl;
    std::cout << "Architecture: " << model->describe() << std::endl;
    std::cout << std::endl;

    // Training parameters
    const int batchSize = 64;
    const double learningRate = 0.001; // Classic value .. TODO Should we tune this ? 

    auto criterion = torch::nn::MSELoss();
    auto optimizer = torch::optim::Adam(model->parameters(), learningRate);

    std::cout << "Training Configuration" << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;
    std::cout << "Epochs: " << numEpochs << std::endl;
    std::cout << "Learning rate: " << learningRate << std::endl;
    std::cout << "Optimizer: Adam" << std::endl;
    std::cout << "Loss function: " << criterion->name() << std::endl;
    std::cout << std::endl;

    std::cout << "Starting Training" << std::endl;
    model->train();  // Set model to training mode

    // Vector holding one index for each training sample
    std::vector<int> indices(numTrain);
    for (int i = 0; i < numTrain; i++) {
        indices[i] = i;
    }

    std::mt19937 rng(randomSeed);

    auto start = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        double epochLoss = 0.0;
        int numBatches = 0;

        // Shuffle indices at the beginning of each epoch
        std::shuffle(indices.begin(), indices.end(), rng);

        // Training per batch
        for (int batchStart = 0; batchStart < numTrain; batchStart += batchSize) {
            int currentBatchSize = std::min(batchSize, numTrain - batchStart); // Just for the last iteration no left overs ...

            // Extract batch using shuffled indices - Gather data
            std::vector<Tensor> batchXList, batchYList;
            for (int i = 0; i < currentBatchSize; i++) {
                int idx = indices[batchStart + i];
                batchXList.push_back(X_train[idx]);
                batchYList.push_back(y_train[idx]);
            }

            Tensor batchX = torch::stack(batchXList);
            Tensor batchY = torch::stack(batchYList);

            // Flatten target: [batch, N, 3] -> [batch, N*3]
            Tensor batchYFlat = batchY.reshape({currentBatchSize, -1});

            // Forward pass
            Tensor predictions = model->forward(batchX);
            Tensor loss = criterion(predictions, batchYFlat);

            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epochLoss += loss.item<double>();
            numBatches++;
        }

        // Print progress with validation loss
        if (epoch % 5 == 0 || epoch == numEpochs - 1) {
            double avgTrainLoss = epochLoss / numBatches;

            // Compute validation loss
            model->eval();
            torch::NoGradGuard val_no_grad;
            Tensor valPred = model->forward(X_val);
            Tensor valYFlat = y_val.reshape({numVal, -1});
            double valLoss = criterion(valPred, valYFlat).item<double>();
            model->train();

            std::cout << "Epoch " << std::setw(3) << epoch
                      << " | Train Loss: " << std::fixed << std::setprecision(6)
                      << avgTrainLoss
                      << " | Val Loss: " << valLoss << std::endl;
        }

        // Save model every 100 epochs
        if ((epoch + 1) % 100 == 0 || epoch == numEpochs - 1) { // make sure to save at last epoch
            std::string modelPath = "lstm_model_epoch_" + std::to_string(epoch + 1) + ".pt";
            torch::save(model, modelPath);
            std::cout << "  -- Saved model: " << modelPath << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "Training Complete" << std::endl;
    std::cout << "Elapsed(s) = " << since<std::chrono::seconds>(start).count() << std::endl;

    // EVALUATION
    model->eval();
    torch::NoGradGuard no_grad;

    evaluateSet(model, X_train, y_train, numTrain, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Training");
    evaluateSet(model, X_val, y_val, numVal, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Validation");
    evaluateSet(model, X_test, y_test, numTest, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Test");
    
    return 0;
}
