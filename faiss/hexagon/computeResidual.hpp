#pragma once
#include <kompute/Kompute.hpp>

namespace faiss {
namespace hexagon{

void calResidual(
    kp::Manager* mgr,
    idx_t dim,
    std::shared_ptr<kp::tensorT<float>> vecs,
    std::shared_ptr<kp::tensorT<idx_t>> ids,
    std::shared_ptr<kp::tensorT<float>> src,
    std::shared_ptr<kp::tensorT<float>> residuals);

}
}
