#include "torch/nn/modules/linear.h"
#include <torch/nn/module.h>
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

using torch::Tensor;

class LSTMNetwork: public torch::nn::Module {
public:
    LSTMNetwork(int inputSize, int hiddenStateSize, int outputSize) {
        lstm =  register_module("lstm", torch::nn::LSTM(
          torch::nn::LSTMOptions(inputSize, hiddenStateSize).batch_first(true)
      ));

      fc = register_module("fc", torch::nn::Linear(hiddenStateSize, outputSize));
    }

    Tensor forward(Tensor x) {
        auto lstmOutput = lstm->forward(x);
        auto out = std::get<0>(lstmOutput); // TODO Check what is the second value of the tuple
        out = fc->forward(out);
        return out;
    }

    

private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};
};

int main() {
    Tensor X = torch::tensor(
      {{{0.}, {1.}, {2.}, {3.}, {4.}}},
      torch::TensorOptions().dtype(torch::kFloat32));

    Tensor y = torch::tensor({{{1.}, {2.}, {3.}, {4.}, {5.}}}).to(torch::kFloat32);

    std::cout << "X shape: " << X.sizes() << std::endl;
    std::cout << "X:\n" << X << std::endl;

    std::cout << "\ny shape: " << y.sizes() << std::endl;
    std::cout << "y:\n" << y << std::endl;

    auto model = std::make_shared<LSTMNetwork>(1,10,1);

    torch::Tensor output = model->forward(X);

    std::cout << "\nBefore training:" << std::endl;
    std::cout << "Input shape: " << X.sizes() << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Output:\n" << output << std::endl;

    auto criterion = torch::nn::MSELoss();
    auto optimizer = torch::optim::Adam(model->parameters(), 0.01);

    std::cout << "\nTraining LSTM to learn: output = input + 1" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
      // Forward pass
      torch::Tensor prediction = model->forward(X);
      torch::Tensor loss = criterion(prediction, y);

      // Backward pass
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      // Print loss every 20 epochs
      if (epoch % 20 == 0) {
        std::cout << "Epoch " << std::setw(3) << epoch
                  << " | Loss: " << std::fixed << std::setprecision(4)
                  << loss.item<float>() << std::endl;
      }
    }

    // Testing
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing LSTM:" << std::endl;
    std::cout << "========================================" << std::endl;

    model->eval();
    torch::NoGradGuard no_grad;

    torch::Tensor test_output = model->forward(X);

    std::cout << "Input:  " << X.squeeze() << std::endl;
    std::cout << "Output: " << test_output.squeeze() << std::endl;
    std::cout << "Target: " << y.squeeze() << std::endl;

    std::cout << "\nLSTM learned to add 1 to each number!" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "RNN vs LSTM Architecture:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "RNN:  Hidden state only (short-term memory)" << std::endl;
    std::cout << "LSTM: Cell state + Hidden state + 3 gates" << std::endl;
    std::cout << "      - Forget gate: what to forget" << std::endl;
    std::cout << "      - Input gate:  what to remember" << std::endl;
    std::cout << "      - Output gate: what to output" << std::endl;
    std::cout << "\nLSTM handles long sequences better!" << std::endl;

    return 0;
}
