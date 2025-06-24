#pragma once
#include <kompute/Kompute.hpp>

namespace faiss {
namespace hexagon{

void calResidual(
    kp::Manager* mgr,
    idx_t dim,
    std::shared_ptr<kp::TensorT<float>> vecs,
    std::shared_ptr<kp::TensorT<idx_t>> ids,
    std::shared_ptr<kp::TensorT<float>> src,
    std::shared_ptr<kp::TensorT<float>> residuals);

}
}
