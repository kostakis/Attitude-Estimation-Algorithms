#include "Utils.hpp"
#include "csvreader.hpp"
#ifdef USE_NO_ATTENTION
#include "LSTMNetworkNoAttention.h"
#else
#include "LSTMNetwork.h"
#endif

#include <exception>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <algorithm>
#include <vector>

#include <torch/torch.h>

using torch::Tensor;

bool isNumber(const std::string& str) {
    if (str.empty()) return false;
    
    try {
        std::size_t pos;
        std::stoi(str, &pos);
        return pos == str.length(); // ensure entire string was converted
    } catch (const std::invalid_argument&) {
        return false;
    } catch (const std::out_of_range&) {
        return false;
    }
}

int printHelpMessageAndExit() {
    std::cout << "Usage: ./lstmEval.out <model_path> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  [sample_index]       Evaluate single sample (e.g., 0, 1000)" << std::endl;
    std::cout << "  --save-all, -a       Generate predictions for ALL samples and save to CSV" << std::endl;
    std::cout << "  --window N, -w N     Prediction horizon (default: 10)" << std::endl;
    std::cout << "  --help, -h           Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  ./lstmEval.out lstm_model_epoch_1000.pt 0" << std::endl;
    std::cout << "  ./lstmEval.out lstm_model_epoch_1000.pt --save-all" << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    bool saveAll = false;
    std::string modelPath;
    int sampleIndex = 0;
    int windowSize = 10;  // Default, override with --window N

    if(argc == 1) {
        std::cout << "Wrong input use -h to see help" << std::endl;
        return -1;
    }

    // Check if first argument is a help flag
    std::string firstArg = argv[1];
    if (firstArg == "--help" || firstArg == "-h") {
        return printHelpMessageAndExit();
    }

    modelPath = firstArg;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--save-all" || arg == "-a" ) {
            saveAll = true;
        } else if((arg == "--window" || arg == "-w") && i + 1 < argc) {
            windowSize = std::stoi(argv[i + 1]);
            i++;
        } else if(i == 2 && isNumber(arg)) {
            sampleIndex = std::stoi(arg);
        } else {
            std::cout << "Wrong argument use -h to view help" << std::endl;
            return -1;
        }
    }

    // Model Configuration
    const int lookbackWindow = 50;  // Must match training
    const int NUM_INPUT_FEATURES = 9;
    const int NUM_OUTPUT_FEATURES = 3;
    const int outputSize = windowSize * NUM_OUTPUT_FEATURES;
    const int hiddenStateSize = 128;
    const unsigned int randomSeed = 42;  // Must match training for shuffle

    std::cout << "LSTM Model Evaluation" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    if (saveAll) {
        std::cout << "Mode: Generate predictions for ALL samples" << std::endl;
    } else {
        std::cout << "Mode: Single sample evaluation (index: " << sampleIndex << ")" << std::endl;
    }
    std::cout << "Lookback window: " << lookbackWindow << " timesteps" << std::endl;
    std::cout << "Prediction horizon: " << windowSize << " timesteps" << std::endl;
    std::cout << std::endl;

    // Load model
    std::cout << "Loading model..." << std::endl;
    auto model = std::make_shared<LSTMNetwork>(NUM_INPUT_FEATURES, hiddenStateSize, outputSize);

    try {
        torch::load(model, modelPath);
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    // Load CSV
    CsvReader datasetReader("Data/Quadcopter_Datasets/all_combined_reordered.csv");
    datasetReader.read();
    Eigen::MatrixXd dataset = datasetReader.getEigenData();

    int totalSamples = dataset.rows();
    int trainingSamples = totalSamples - windowSize - lookbackWindow + 1;

    // Convert to tensor and normalize - must match training!
    Tensor datasetTensor = LSTMNetwork::eigenToTensor(dataset);

    Tensor featureMean = datasetTensor.mean(/*dim=*/0, /*keepdim=*/true);
    Tensor featureStd = datasetTensor.std(/*dim=*/0, /*unbiased=*/true, /*keepdim=*/true);
    featureStd = featureStd + 1e-8;

    Tensor scaledDatasetTensor = (datasetTensor - featureMean) / featureStd;

    Tensor targetMean = featureMean.slice(/*dim=*/1, /*start=*/0, /*end=*/NUM_OUTPUT_FEATURES);
    Tensor targetStd = featureStd.slice(/*dim=*/1, /*start=*/0, /*end=*/NUM_OUTPUT_FEATURES);

    // Create all windows
    auto options_w = torch::TensorOptions().dtype(torch::kDouble);
    Tensor anglesTensor = scaledDatasetTensor.slice(1, 0, NUM_OUTPUT_FEATURES);
    Tensor X = torch::zeros({trainingSamples, lookbackWindow, NUM_INPUT_FEATURES}, options_w);
    Tensor y = torch::zeros({trainingSamples, windowSize, NUM_OUTPUT_FEATURES}, options_w);

    for (int i = 0; i < trainingSamples; i++) {
        X[i] = scaledDatasetTensor.slice(0, i, i + lookbackWindow);
        int predStart = i + lookbackWindow;
        y[i] = anglesTensor.slice(0, predStart, predStart + windowSize);
    }

    // Shuffle windows
    std::vector<int> shuffleIndices(trainingSamples);
    for (int i = 0; i < trainingSamples; i++) shuffleIndices[i] = i;
    std::mt19937 shuffleRng(randomSeed);
    std::shuffle(shuffleIndices.begin(), shuffleIndices.end(), shuffleRng);

    Tensor X_shuffled = torch::zeros_like(X);
    Tensor y_shuffled = torch::zeros_like(y);
    for (int i = 0; i < trainingSamples; i++) {
        X_shuffled[i] = X[shuffleIndices[i]];
        y_shuffled[i] = y[shuffleIndices[i]];
    }

    // Train/Val/Test
    int numTrain = static_cast<int>(trainingSamples * 0.8);
    int numVal = static_cast<int>(trainingSamples * 0.1);
    int numTest = trainingSamples - numTrain - numVal;

    // Split into train/val/test
    Tensor X_train = X_shuffled.slice(0, 0, numTrain);
    Tensor y_train = y_shuffled.slice(0, 0, numTrain);
    Tensor X_val = X_shuffled.slice(0, numTrain, numTrain + numVal);
    Tensor y_val = y_shuffled.slice(0, numTrain, numTrain + numVal);
    Tensor X_test = X_shuffled.slice(0, numTrain + numVal, trainingSamples);
    Tensor y_test = y_shuffled.slice(0, numTrain + numVal, trainingSamples);

    std::cout << "Data normalized (z-score standardization)" << std::endl;
    std::cout << "Train: " << numTrain << " | Val: " << numVal << " | Test: " << numTest << " (80/10/10 split)" << std::endl;
    std::cout << std::endl;

    // Evaluate all splits
    model->eval();
    {
        torch::NoGradGuard no_grad;
        evaluateSet(model, X_train, y_train, numTrain, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Training");
        evaluateSet(model, X_val, y_val, numVal, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Validation");
        evaluateSet(model, X_test, y_test, numTest, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Test");
    }

    // Handle --save-all mode
    if (saveAll) {
        std::cout << "Generating predictions for " << trainingSamples << " samples..." << std::endl;

        model->eval();
        torch::NoGradGuard no_grad;

        // Open output file
        std::ofstream outFile("Results/lstm_predictions.csv"); // Rename after app ends to proper name ...
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open Results/lstm_predictions.csv for writing" << std::endl;
            return 1;
        }

        // Write header
        outFile << "sample,step_ahead,roll_pred,pitch_pred,yaw_pred,roll_gt,pitch_gt,yaw_gt,set" << std::endl;

        // Process all shuffled samples
        for (int sample = 0; sample < trainingSamples; sample++) {
            // Get prediction from shuffled window
            Tensor input = X_shuffled[sample].unsqueeze(0);
            Tensor output = model->forward(input);
            Tensor prediction = output.view({1, windowSize, NUM_OUTPUT_FEATURES});

            // Denormalize predictions and targets back to radians
            Tensor unscaledPred = ((prediction * targetStd) + targetMean).squeeze(0);
            Tensor unscaledTarget = (y_shuffled[sample] * targetStd) + targetMean;

            // Determine set label
            std::string setLabel;
            if (sample < numTrain) { 
                setLabel = "train";
            } else if (sample < numTrain + numVal) {
                setLabel = "val";
            } else {
                setLabel = "test";
            }

            // Write predictions for each step
            for (int step = 0; step < windowSize; step++) {
                int stepAhead = step + 1;

                auto pred = unscaledPred[step];
                double roll_pred = pred[0].item<double>();
                double pitch_pred = pred[1].item<double>();
                double yaw_pred = pred[2].item<double>();

                auto gt = unscaledTarget[step];
                double roll_gt = gt[0].item<double>();
                double pitch_gt = gt[1].item<double>();
                double yaw_gt = gt[2].item<double>();

                outFile << sample << "," << stepAhead << ","
                       << roll_pred << "," << pitch_pred << "," << yaw_pred << ","
                       << roll_gt << "," << pitch_gt << "," << yaw_gt << ","
                       << setLabel << std::endl;
            }

            // Progress indicator
            if (sample % 500 == 0) {
                std::cout << "  Processed " << sample << "/" << trainingSamples << " samples" << std::endl;
            }
        }

        outFile.close();
        std::cout << "Done! Saved predictions to Results/lstm_predictions.csv" << std::endl;
        std::cout << "Total prediction rows: " << trainingSamples * windowSize << std::endl;
        return 0;
    }

    // Single sample evaluation mode (index into shuffled windows)
    if (sampleIndex >= trainingSamples) {
        std::cerr << "Sample index too large! Max: " << trainingSamples - 1 << std::endl;
        return 1;
    }

    // Determine which set this sample belongs to
    std::string setLabel;
    if (sampleIndex < numTrain) {
        setLabel = "train";
    } else if (sampleIndex < numTrain + numVal) {
        setLabel = "val";
    }
    else {
        setLabel = "test";
    }
    
    std::cout << "Sample " << sampleIndex << " belongs to: " << setLabel << " set" << std::endl;
    std::cout << std::endl;

    // Run prediction on shuffled window
    model->eval();
    torch::NoGradGuard no_grad;

    Tensor input = X_shuffled[sampleIndex].unsqueeze(0);
    Tensor output = model->forward(input);
    Tensor prediction = output.view({1, windowSize, NUM_OUTPUT_FEATURES});

    // Denormalize predictions and targets back to radians
    Tensor unscaledPred = ((prediction * targetStd) + targetMean).squeeze(0);
    Tensor groundTruth = (y_shuffled[sampleIndex] * targetStd) + targetMean;

    // Display results
    std::cout << "=== Prediction (next " << windowSize << " timesteps) ===" << std::endl;
    std::cout << "Shape: [" << windowSize << ", 3] (roll, pitch, yaw) in radians" << std::endl;
    std::cout << unscaledPred << std::endl;
    std::cout << std::endl;

    std::cout << "=== Ground Truth (next " << windowSize << " timesteps) ===" << std::endl;
    std::cout << groundTruth << std::endl;
    std::cout << std::endl;

    // Calculate error on original scale
    Tensor error = unscaledPred - groundTruth;
    Tensor rmse = error.pow(2).mean().sqrt();
    std::cout << "=== Error Metrics ===" << std::endl;
    std::cout << "RMSE: " << std::fixed << std::setprecision(6)
              << rmse.item<double>() << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(rmse.item<double>()) << " degrees" << std::endl;

    return 0;
}
