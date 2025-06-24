#include <kompute/Kompute.hpp>

#include <memory>

#include <faiss/hexagon/HexagonGpuIndexFlat.hpp>

namespace faiss {
namespace hexagon { 

HexagonGpuIndex::HexagonGpuIndex(
    std::shared_ptr<kp::Manager> mgr,
    int dims,
    faiss::MetricType metric,
    float metricArg)
    : Index(dims, metric) {
    
    this->mgr_ = mgr;
    // this->config_ = HexagonGpuIndexConfig();
    this->is_trained = true;
    this->resetIndex_(dims);
}

size_t HexagonGpuIndex::getNumVecs() {
    return this->data_->getSize();
}

void HexagonGpuIndex::reset(uint dim) {
    this->d = dim;
    this->resetIndex_(dim);
}

void train(idx_t n, const float* x) {
    // 不需要
    return;
}

void HexagonGpuIndex::reconstruct(idx_t key, float* out) {
    // 返回指定id的对应向量
    this->data_->reconstruct(key, 1, out);
    return;
}

void HexagonGpuIndex::reconstruct_n(idx_t i0, idx_t num, float* out) {
    this->data_->reconstruct(i0, num, out);
}

void HexagonGpuIndex::reconstruct_batch(idx_t n, const idx_t* keys, float* out) {
    std::vector<idx_t> ids(keys, keys + n);
    this->data_->reconstruct(ids, out);
}

void HexagonGpuIndex::add(idx_t n, const float* x) {
    this->addImpl_(n, x);
}

void HexagonGpuIndex::add_with_ids(idx_t n, const float* x, const idx_t* ids = nullptr) {
    this->addImpl_(n, x);
}

std::shared_ptr<kp::Manager> HexagonGpuIndex::getKomputeManager() {
    return this->mgr_;
}

void HexagonGpuIndex::resetIndex_(uint dim) {
    this->data_ = std::make_shared<hexagon::FlatIndex>(this->mgr_, dim, false);
}

void HexagonGpuIndex::addImpl_(idx_t n, const float* x, const idx_t* ids = nullptr) {
    this->data_->add(x, n);
}

void HexagonGpuIndex::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) {
    // 查找            
    this->data_->query(x, k, this->metric_type, this->metric_arg, distances, labels, false);
    return;
}
 
void HexagonGpuIndex::compute_residual_n(idx_t n, const float* xs, float* residuals, const idx_t* keys) {
    
    this->data_->computeResidual(xs, keys, residuals);
    return;
}

void HexagonGpuIndex::compute_residual(const float* x, float* residual, idx_t key) {
    compute_residual_n(1, x, residual, &key);
}

void HexagonGpuIndex::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params = nullptr) 
{   
    // 调用FlatIndex::query方法
    this->data_->query(x, k, this->metric_type, 0.0, distances, labels, false);
    return;
}

void HexagonGpuIndex::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* recons,
        const SearchParameters* params = nullptr) 
{
    this->search(n, x, k, distances, labels, nullptr);
    this->reconstruct_batch(n, labels, recons);
}

void HexagonGpuIndex::assign(idx_t n, const float* x, idx_t* labels, idx_t k = 1) {
    // 调用FlatIndex::query方法
    this->data_->query(x, k, this->metric_type, 0.0, nullptr, labels, false);
    return;
}


}
}