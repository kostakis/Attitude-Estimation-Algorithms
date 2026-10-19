#ifndef LSTMNETWORK_HPP
#define LSTMNETWORK_HPP

#include <torch/torch.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "Utils.hpp"

using torch::Tensor;

class LSTMNetwork: public torch::nn::Module {
public:
    LSTMNetwork(int inputSize, int hiddenStateSize, int outputSize, double dropoutRate = 0.2) {
        // 1. Standard LSTM layer
        lstm = register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(inputSize, hiddenStateSize).batch_first(true)
        ));

        //2. Attention Projection: Learns a non-linear representation of the hidden states
        attn_linear = register_module("attn_linear", torch::nn::Linear(hiddenStateSize, hiddenStateSize));

        // 3. Attention Scoring: Reduces the projected representation to a scalar score per timestep
        attn_vector = register_module("attn_vector", torch::nn::Linear(hiddenStateSize, 1));

        // 4. Dropout for regularization (active only during training)
        dropout = register_module("dropout", torch::nn::Dropout(dropoutRate));

        // 5. Fully Connected Output Layer
        fc = register_module("fc", torch::nn::Linear(hiddenStateSize, outputSize));

        inputSize_ = inputSize;
        hiddenStateSize_ = hiddenStateSize;
        outputSize_ = outputSize;

        this->to(torch::kDouble);
    }

    std::string describe() const {
        std::ostringstream ss;
        ss << "Input(" << inputSize_ << ") -> LSTM(" << hiddenStateSize_ << ") -> Attention -> FC(" << outputSize_ << ")";
        return ss.str();
    }

    Tensor forward(Tensor X) {
        // Step 1: Pass through LSTM
        auto lstmOutput = lstm->forward(X);
        auto hiddenStates = std::get<0>(lstmOutput); // [batch, seq_len, hidden]

        // Step 2: Apply dropout on LSTM hidden states
        hiddenStates = dropout->forward(hiddenStates);

        // Step 3: Attention - project and score
        auto energy = torch::tanh(attn_linear->forward(hiddenStates));
        auto scores = attn_vector->forward(energy).squeeze(-1); // [batch, seq_len]

        // Step 4: Normalize scores via softmax
        auto weights = torch::softmax(scores, /*dim=*/1);

        // Step 5: Weighted sum -> context vector
        auto context = torch::bmm(weights.unsqueeze(1), hiddenStates).squeeze(1); // [batch, hidden]

        // Step 6: Dropout before FC
        context = dropout->forward(context);

        // Step 7: FC output
        auto out = fc->forward(context); // [batch, output_size]
        return out;
    }

    static Tensor eigenToTensor(const Eigen::MatrixXd& matrixToConvert) {
        int rows = matrixToConvert.rows();
        int cols = matrixToConvert.cols();

        auto options = torch::TensorOptions().dtype(torch::kDouble);
        auto tensor = torch::zeros({rows, cols}, options);

        for (int i = 0; i < rows; i++) {
            std::vector<double> rowData(cols);
            for (int j = 0; j < cols; j++) {
                rowData[j] = matrixToConvert(i, j);
            }

            auto rowTensor = torch::from_blob(rowData.data(), {cols}, options).clone();
            tensor[i] = rowTensor;
        }
        return tensor;
    }

private:
    int inputSize_;
    int hiddenStateSize_;
    int outputSize_;
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear attn_linear{nullptr};
    torch::nn::Linear attn_vector{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear fc{nullptr};
};

// Print evaluation metrics in a formatted way
inline void printMetrics(double rmseValue, double maeValue,
                  const std::vector<double>& rmsePerAngle,
                  const std::vector<std::vector<double>>& rmsePerStep,
                  int windowSize) {
    std::cout << "Overall Metrics:" << std::endl;
    std::cout << "  RMSE (all): " << std::fixed << std::setprecision(6)
              << rmseValue << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(rmseValue) << " deg" << std::endl;
    std::cout << "  MAE  (all): " << std::fixed << std::setprecision(6)
              << maeValue << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(maeValue) << " deg" << std::endl;
    std::cout << std::endl;

    std::cout << "RMSE per angle (all samples, all steps):" << std::endl;
    std::cout << "  Roll  RMSE: " << std::setprecision(6) << rmsePerAngle[0] << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(rmsePerAngle[0])<< " deg " << std::endl;
    std::cout << "  Pitch RMSE: " << std::setprecision(6) << rmsePerAngle[1] << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(rmsePerAngle[1]) << " deg " << std::endl;
    std::cout << "  Yaw   RMSE: " << std::setprecision(6) << rmsePerAngle[2] << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(rmsePerAngle[2]) << " deg" << std::endl;
    std::cout << std::endl;

    std::cout << "RMSE per step" << std::endl;
    std::cout << "Step | Roll (deg) | Pitch (deg) | Yaw (deg)" << std::endl;
    std::cout << "-----+------------+-------------+-----------" << std::endl;
    for (int step = 0; step < windowSize; step++) {
        std::cout << std::setw(4) << (step + 1) << " | "
                  << std::fixed << std::setprecision(6)
                  << std::setw(10) << Utils::convertToDeg(rmsePerStep[step][0]) << " | "
                  << std::setw(11) << Utils::convertToDeg(rmsePerStep[step][1]) << " | "
                  << std::setw(9) << Utils::convertToDeg(rmsePerStep[step][2]) << std::endl;
    }
    std::cout << std::endl;
}

// Evaluate a dataset split and print metrics
inline void evaluateSet(std::shared_ptr<LSTMNetwork> model,
                 const Tensor& X_set, const Tensor& y_set,
                 int numSamples, int windowSize, int numOutputFeatures,
                 const Tensor& targetStd, const Tensor& targetMean,
                 const std::string& setName) {
    std::cout << "=== " << setName << " Metrics (" << numSamples << " samples) ===" << std::endl;

    Tensor preds = model->forward(X_set);
    preds = preds.view({numSamples, windowSize, numOutputFeatures});

    // Denormalize back to radians
    Tensor unscaledPreds = (preds * targetStd) + targetMean;
    Tensor unscaledTargets = (y_set * targetStd) + targetMean;

    Tensor diff = unscaledPreds - unscaledTargets;

    double rmseValue = diff.pow(2).mean().sqrt().item<double>();
    double maeValue = diff.abs().mean().item<double>();

    // RMSE per angle
    std::vector<int64_t> dims = {0, 1};
    Tensor rmsePerAngleTensor = diff.pow(2).mean(dims).sqrt();
    auto rmse_acc = rmsePerAngleTensor.accessor<double, 1>();
    std::vector<double> rmsePerAngle = {rmse_acc[0], rmse_acc[1], rmse_acc[2]};

    // RMSE per step
    std::vector<std::vector<double>> rmsePerStep;
    for (int step = 0; step < windowSize; step++) {
        Tensor stepPred = unscaledPreds.select(1, step);
        Tensor stepTarget = unscaledTargets.select(1, step);
        Tensor rmse = (stepPred - stepTarget).pow(2).mean(0).sqrt();
        auto acc = rmse.accessor<double, 1>();
        rmsePerStep.push_back({acc[0], acc[1], acc[2]});
    }

    printMetrics(rmseValue, maeValue, rmsePerAngle, rmsePerStep, windowSize);
}

#endif // LSTMNETWORK_HPP
