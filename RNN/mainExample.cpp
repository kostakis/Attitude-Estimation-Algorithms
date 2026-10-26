#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/loss.h"
#include "torch/nn/modules/rnn.h"
#include "torch/optim/adam.h"
#include <torch/torch.h>
#include <torch/types.h>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

class HelloRNN : public torch::nn::Module {
  public:
    HelloRNN();

    torch::Tensor forward(torch::Tensor x);

  private:
    torch::nn::RNN rnn{nullptr};
    torch::nn::Linear fc{nullptr};
};

HelloRNN::HelloRNN() {
    rnn = register_module("rnn", torch::nn::RNN(torch::nn::RNNOptions(1, 10).batch_first(true)));

    fc = register_module("fc", torch::nn::Linear(10, 1));
}

torch::Tensor HelloRNN::forward(torch::Tensor x) {
    auto rnn_output = rnn->forward(x);
    auto out = std::get<0>(rnn_output);
    out = fc->forward(out);
    return out;
}

int main() {
    // Initialize spdlog
    auto console = spdlog::stdout_color_mt("mainExample");

    torch::Tensor X = torch::tensor({{{0.}, {1.}, {2.}, {3.}, {4.}}}, torch::TensorOptions().dtype(torch::kFloat32));

    torch::Tensor y = torch::tensor({{{1.}, {2.}, {3.}, {4.}, {5.}}}).to(torch::kFloat32);

    console->info("X shape: [{}, {}, {}]", X.size(0), X.size(1), X.size(2));
    console->info("X:\n{}", torch::str(X));

    console->info("y shape: [{}, {}, {}]", y.size(0), y.size(1), y.size(2));
    console->info("y:\n{}", torch::str(y));

    auto model = std::make_shared<HelloRNN>();

    torch::Tensor output = model->forward(X);

    console->info("Input shape: [{}, {}, {}]", X.size(0), X.size(1), X.size(2));
    console->info("Output shape: [{}, {}, {}]", output.size(0), output.size(1), output.size(2));
    console->info("Output:\n{}", torch::str(output));

    auto criterion = torch::nn::MSELoss();
    auto optimizer = torch::optim::Adam(model->parameters(), 0.01);

    console->info("Training RNN to learn: output = input + 1");
    console->info("----------------------------------------");

    // Training loop
    for(int epoch = 0; epoch < 100; ++epoch) {
        // Forward pass
        torch::Tensor prediction = model->forward(X);
        torch::Tensor loss = criterion(prediction, y);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // Print loss every 20 epochs
        if(epoch % 20 == 0) {
            console->info("Epoch {:3d} | Loss: {:.4f}", epoch, loss.item<float>());
        }
    }

    // Testing
    console->info("========================================");
    console->info("Testing RNN:");
    console->info("========================================");

    model->eval();
    torch::NoGradGuard no_grad;

    torch::Tensor xNew =
        torch::tensor({{{10.}, {11.}, {12.}, {13.}, {14.}}}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor test_output = model->forward(xNew);

    console->info("Input:  {}", torch::str(xNew.squeeze()));
    console->info("Output: {}", torch::str(test_output.squeeze()));
    // console->info("Target: {}", torch::str(y.squeeze()));

    console->info("RNN learned to add 1 to each number!#NOT");

    return 0;
}
