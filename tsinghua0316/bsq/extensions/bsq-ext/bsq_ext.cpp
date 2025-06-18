/*bsq_ext.cpp*/
#include "bsq_ext.h"
#include <iostream>
#include <iomanip>


std::vector<torch::Tensor> weiquan_forward(const torch::Tensor& x, const torch::Tensor& s, const float intervals, const float neg, const float pos){
    auto x_bar = 2*torch::sigmoid(x)-1.0;
    auto y = torch::round(torch::clamp(x_bar / (s/intervals), neg, pos)) * (s/intervals);
    std::vector<torch::Tensor>ret = {y, x_bar};
    return ret;
}

std::vector<torch::Tensor> weiquan_backward(const torch::Tensor& x,const torch::Tensor& s,torch::Tensor& y, torch::Tensor& x_bar, const float intervals, const float neg, const float pos){
    torch::Tensor grad_s = torch::zeros_like(s);
    auto sig_x = x_bar;
    x_bar = x_bar / (s/intervals);
    x_bar = ( y - (1-(x_bar < neg - 0.5).to(torch::kFloat) - (x_bar > pos+0.5).to(torch::kFloat)) * sig_x) * (y-sig_x) / s;
    if (s.dim() == 1){
        grad_s = x_bar.sum().view(s.sizes())/pow(y.numel(), 0.5);
    }else if(s.dim() == 2){
        grad_s = x_bar.sum(1).view(s.sizes())*s.size(0)/pow(y[0].numel(), 0.5);
    }else if(s.dim() == 4){
        grad_s = x_bar.sum({1,2,3}).view(s.sizes())*s.size(0)/pow(y[0].numel(), 0.5);
    }
    return {0.5-0.5*sig_x*sig_x, grad_s};
}

torch::Tensor weiquan_init(const torch::Tensor& x, const torch::Tensor& param){
    auto x_bar = 2*torch::sigmoid(x) - 1.0;
    x_bar = x_bar.abs().view({x.size(0),-1});
    //auto s = ((std::get<0>(x_bar.max(1)) - std::get<0>(x_bar.min(1)))*0.99).reshape(param.sizes());
    auto s = (2*std::get<0>(x_bar.max(1))* 1.0).reshape(param.sizes());
    auto mask = (s > 0.05).to(torch::kFloat);
    s = s * mask + 0.05 * (1-mask);
    return s;
}

torch::Tensor quant_base_forward(const torch::Tensor& x){
    return  2*torch::sigmoid(x) - 1;
}

torch::Tensor quant_base_backward(const torch::Tensor& x, torch::Tensor& y){
    return 0.5 - 0.5*y*y;
}

torch::Tensor quant_base_init(const torch::Tensor& param){
    // auto s = param.abs().mean({1,2,3}) * 31.8747549;
    // s = s.view({param.size(0),1,1,1});
    // auto p = torch::clamp(param/s +0.5,1.0/256.0,254.0/256.0);
    auto p = torch::clamp(param/2+0.5,1.0/256.0,255.0/256.0);
    return torch::log(p/(1-p));
}


std::vector<torch::Tensor> actquan_forward(const torch::Tensor& x, const torch::Tensor& s,const float neg, const float pos){
    auto x_bar = x/s;
    auto y = torch::round(torch::clamp(x_bar, neg, pos)) * s;
    return {y, x_bar};
}

torch::Tensor actquan_backward(const torch::Tensor& x, const torch::Tensor& s, torch::Tensor& y, torch::Tensor& x_bar, const float neg, const float pos){
    torch::Tensor grad_s = torch::zeros_like(s);
    x_bar = ( y - (1-(x_bar < neg - 0.5).to(torch::kFloat) - (x_bar > pos+0.5).to(torch::kFloat)) * x) * (y-x) / s;
    if (s.dim() == 1){
        grad_s = x_bar.sum().view(s.sizes())/pow(y.numel(), 0.5);
    }else if(s.dim() == 2){
        grad_s = x_bar.sum(1).view(s.sizes())*s.size(0)/pow(y[0].numel(), 0.5);
    }else if(s.dim() == 4){
        grad_s = x_bar.sum({1,2,3}).view(s.sizes())*s.size(0)/pow(y[0].numel(), 0.5);
    }
    return grad_s;
}

torch::Tensor actquan_init(const torch::Tensor& x, const torch::Tensor& param, bool isFirstBatch){
    auto s = torch::max(0.99*x.abs().max(), 0.5 * torch::ones_like(param));
    if (! isFirstBatch)
        s = s * 0.1 + param * 0.9;
    // std::cout << "param" << param << std::endl;
    // std::cout << "s" << s << std::endl;
    return s;
    // return torch::max(x.max(), param);
}


// part3:pybind11 （将python与C++11进行绑定， 注意这里的forward，backward名称就是后来在python中可以引用的方法名）
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("weiquan_forward", &weiquan_forward, "weiquan forward");
    m.def("weiquan_backward", &weiquan_backward, "weiquan backward");
    m.def("weiquan_init", &weiquan_init, "weiquan_init");

    m.def("quant_base_forward", &quant_base_forward, "quant_base_forward");
    m.def("quant_base_backward", &quant_base_backward, "quant_base_backward");
    m.def("quant_base_init", &quant_base_init, "quant_base_init");

    m.def("actquan_forward", &actquan_forward, "actquan forward");
    m.def("actquan_backward", &actquan_backward, "actquan backward");
    m.def("actquan_init", &actquan_init, "actquan_init");
}
