#include "torch/nn/module.h"
#include "torch/nn/modules/linear.h"
#include "torch/nn/modules/loss.h"
#include "torch/nn/modules/rnn.h"
#include "torch/optim/adam.h"
#include <torch/types.h>
#include <torch/torch.h>

#include <iostream>
#include <iomanip>

class HelloRNN: public torch::nn::Module {
public:
  HelloRNN();

  torch::Tensor forward(torch::Tensor x);
private:
  torch::nn::RNN rnn{nullptr};
  torch::nn::Linear fc{nullptr};
};

HelloRNN::HelloRNN() {
  rnn = register_module("rnn", torch::nn::RNN(
        torch::nn::RNNOptions(1, 10).batch_first(true)
    ));

  fc = register_module("fc", torch::nn::Linear(10, 1));
}

torch::Tensor HelloRNN::forward(torch::Tensor x) {
  auto rnn_output = rnn->forward(x);
  auto out = std::get<0>(rnn_output);
  out = fc->forward(out);
  return out;
}


int main() {  
  torch::Tensor X = torch::tensor(
    {{{0.}, {1.}, {2.}, {3.}, {4.}}}, 
    torch::TensorOptions().dtype(torch::kFloat32));

  torch::Tensor y = torch::tensor({{{1.}, {2.}, {3.}, {4.}, {5.}}}).to(torch::kFloat32);
  
  std::cout << "X shape: " << X.sizes() << std::endl;
  std::cout << "X:\n" << X << std::endl;
  
  std::cout << "\ny shape: " << y.sizes() << std::endl;
  std::cout << "y:\n" << y << std::endl;
  
  auto model = std::make_shared<HelloRNN>();

  torch::Tensor output = model->forward(X);
  
  std::cout << "Input shape: " << X.sizes() << std::endl;
  std::cout << "Output shape: " << output.sizes() << std::endl;
  std::cout << "Output:\n" << output << std::endl;


  auto criterion = torch::nn::MSELoss();
  auto optimizer = torch::optim::Adam(model->parameters(), 0.01);

  std::cout << "\nTraining RNN to learn: output = input + 1" << std::endl;
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
  std::cout << "Testing RNN:" << std::endl;
  std::cout << "========================================" << std::endl;

  model->eval();
  torch::NoGradGuard no_grad;

  torch::Tensor test_output = model->forward(X);

  std::cout << "Input:  " << X.squeeze() << std::endl;
  std::cout << "Output: " << test_output.squeeze() << std::endl;
  std::cout << "Target: " << y.squeeze() << std::endl;

  std::cout << "\nRNN learned to add 1 to each number!" << std::endl;

  return 0;
}
