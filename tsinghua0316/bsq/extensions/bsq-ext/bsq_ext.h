/*test.h*/
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> weiquan_forward(const torch::Tensor& x, const torch::Tensor& s, const float intervals, const float neg, const float pos);

std::vector<torch::Tensor> weiquan_backward(const torch::Tensor& x,const torch::Tensor& s, torch::Tensor& y, torch::Tensor& x_bar,const float intervals, const float neg, const float pos);

torch::Tensor weiquan_init(const torch::Tensor& x, const torch::Tensor& param);

torch::Tensor quant_base_init(const torch::Tensor& param);

torch::Tensor quant_base_forward(const torch::Tensor& x);

torch::Tensor quant_base_backward(const torch::Tensor& x, torch::Tensor& y);

std::vector<torch::Tensor> actquan_forward(const torch::Tensor& x, const torch::Tensor& s,const float neg, const float pos);

torch::Tensor actquan_backward(const torch::Tensor& x, const torch::Tensor& s, torch::Tensor& y, torch::Tensor& x_bar, const float neg, const float pos);

torch::Tensor actquan_init(const torch::Tensor& x, const torch::Tensor& param, bool isFirstBatch);