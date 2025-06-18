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
    this->config_ = HexagonGpuIndexConfig();
    this->is_trained = true;
    this->resetIndex_(dims);
}

void HexagonGpuIndex::getNumVecs() {
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

std::shared_ptr<Kompute::Manager> HexagonGpuIndex::getKomputeManager() {
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

    // 创建需要的tensor，包括查询向量vecs、返回的距离向量、返回的labels数组，且大小需要匹配
    // distance 和 label 是数字，不需要考虑维度dim！！！
    std::shared_ptr<kp::TensorT<float>> vecs
                                    = this->mgr_->tensorT<float>(std::vector<float>(x, x + n * this->d));   // (x, x + n * Vec)
    std::shared_ptr<kp::TensorT<float>> outDistances 
                                    = this->mgr_->tensorT<float>(std::vector<float>(k * n, -1.0));    // n * (k)
    std::shared_ptr<kp::TensorT<idx_t>> outIndices 
                                    = this->mgr_->tensorT<idx_t>(std::vector<idx_t>{k * n, -1});            // n * k
            
    this->data_->query(vecs, k, this->metric_type, 0.0, outDistances, outIndices, false);

    // 把输出放回到指定的指针中
    memcpy(distances, outDistances->data(), k * n * sizeof(float));
    memcpy(labels, outIndices->data(), k * n * sizeof(idx_t));

    return;
}
 
void HexagonGpuIndex::compute_residual_n(idx_t n, const float* xs, float* residuals, const idx_t* keys) {
    std::shared_ptr<kp::TensorT<float>> vecs
                                        = this->mgr_->tensorT<float>(std::vector<float>(xs, xs + n * this->d)); // n * Vec
    std::shared_ptr<kp::TensorT<idx_t>> ids
                                        = this->mgr_->tensorT<float>(std::vector<idx_t>(keys, keys + n));       // n 个label
    std::shared_ptr<kp::TensorT<float>> outResiduals
                                        = this->mgr_->tensorT<float>(std::vector<float>(n * this->d, -0.0));    // n * Vec
    
    this->data_->computeResidual(vecs, ids, outResiduals);

    memcpy(residuals, this->outResiduals, n * this->d * sizeof(float));
    
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
    // data_->query
    // 创建计算结果的tensor
    std::shared_ptr<kp::TensorT<float>> vecs = 
                                            this->mgr_->tensorT(std::vector<float>(x, x + n * this->d));        // n * Vecs
    std::shared_ptr<kp::TensorT<float>> outDistances = 
                                            this->mgr_->tensorT(std::vector<float>(n * k, 0.0));      // n * k
    std::shared_ptr<kp::TensorT<idx_t>> outIndices = 
                                            this->mgr_->tensorT(std::vector<idx_t>(n * k, 0));          // n * k
    
    // 调用FlatIndex::query方法
    this->data_->query(vecs, k, this->metric_type, 0.0, outDistances, outIndices, false);

    // 拷贝数据到返回指针中
    memcpy(distances, outDistances->data(), n * k * sizeof(float));
    memcpy(labels, outIndices, n * k * sizeof(idx_t));

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
    // 与search逻辑相同，但是search还需要返回具体的计算结果，assign只需要返回对应的labels即可（即实际存储时的下标）
    // 创建计算结果的tensor
    std::shared_ptr<kp::TensorT<float>> vecs = 
                                            this->mgr_->tensorT(std::vector<float>(x, x + n * this->d));    // n * Vec
    std::shared_ptr<kp::TensorT<float>> outDistances = 
                                            this->mgr_->tensorT(std::vector<float>(n * k, 0.0));      // n * k
    std::shared_ptr<kp::TensorT<idx_t>> outIndices = 
                                            this->mgr_->tensorT(std::vector<idx_t>(n * k, 0));    // n * k
    
    // 调用FlatIndex::query方法
    this->data_->query(vecs, k, this->metric_type, 0.0, outDistances, outIndices, false);

    // 拷贝数据到返回指针中
    memcpy(labels, outIndices, n * sizeof(idx_t));

    return;
}


}
}