#include <faiss/hexagon/FlatIndex.hpp>
#include <faiss/hexagon/computeResidual.hpp>

#include <math.h>
namespace faiss {
namespace hexagon{

FlatIndex::FlatIndex(kp::Manager* mgr, int dim, bool useFloat16 = false) {
    this->mgr_ = mgr;
    this->dim_ = dim;
    this->useFloat16_ = useFloat16;
    this->data32_ = this->mgr_.tensorT<float>(std::vector<float>{});
    this->data16_ = this->mgr_.tensorT<float>(std::vector<float>{});
    this->capacity = 0;
    this->num_ = 0;
}

bool FlatIndex::getUseFloat16() const {
    return this->useFloat16_;
}

idx_t FlatIndex::getSize() const {
    if (!this->useFloat16_)
        return this->data32_->data<float>().size() / this->dim_;
    else
        return this->data16_->data<float>().size() / this->dim_ * 2;
}

int FlatIndex::getDim() const {
    return this->dim_;
}

void FlatIndex::reserve(size_t numVecs) {
    // 暂存原数据后重新rebuild
    std::vector<float> tmp = this->data32_->data<float>();
    for (int i = numVecs * this->dim_ - tmp.size(); i > 0; i--) {
        tmp.push_back(0.0);
    }
    this->data32_->rebuild(tmp.data(), numVecs * this->dim_, numVecs * this->dim_ * sizeof(float));
    this->capacity = numVecs;
}

kp::Tensor& FlatIndex::getVectorFloat32Ref() {
    return this->data32_;
}

kp::Tensor& FlatIndex::getVectorFloat16Ref() {
    return this->data16_;
}

void FlatIndex::computeResidual(
    std::shared_ptr<kp::tensorT<float>> vecs,
    std::shared_ptr<kp::tensorT<idx_t>> ids,
    std::shared_ptr<kp::tensorT<float>> residuals) {
    
    // 调用外部函数完成计算过程
    faiss::hexagon::calResidual(mgr_, dim_, vecs, ids, this->data32_, residuals);

}

void FlatIndex::reconstruct(
    std::vector<idx_t> ids,
    std::shared_ptr<kp::tensorT<float>> vecs) {
    // 根据id返回对应的向量
    // TODO: 两种实现办法，一种全部在CPU完成，一种在GPU完成，但是会在CPU和GPU传输数据
    if (!this->useFloat16_) {
        std::vector<float>* tmp = vecs->data<float>();
        int j = 0;
        for (auto id : ids) {
            for (int i = 0; i < this->dim_; ++i) {
                vecs->data<float>()[j++] = this->data32_->data<vector>()[id * dim_ + i];
            }
        }
    }
    else {
        // TODO
        return;
    }    
}

void FlatIndex::reconstruct(
    idx_t start,
    idx_t num,
    std::shared_ptr<kp::tensorT<idx_t>> vecs) {
    // 从start开始返回num个vec
    if (start + num - 1 > this->num_ || 
        vecs->data<float>().size() < num * sozeof(float)) {
        return;
    }

    if (!this->useFloat16_) {
        for (int i = 0; i <= num * dim_; ++i) {
            vecs->data<float>()[i] = this->data32_->data<float>()[start * dim_ + i];
        }
    }
    else {
        return;
    }

}

void FlatIndex::add(const float* data, idx_t numVecs) {
    if (data == nullptr || numVecs == 0)
        return;
    
    if (numVecs > (capacity_ - num_)) {
        // 需要扩容，扩容策略选择二倍
        while (num_ + numVecs > capacity_)
            capacity_ *= 2;
        std::vector<float> tmp = this->data32_->data<float>();
        this->data32_->rebuild(tmp, num_ * sizeof(float), capacity_ * sizeof(float));
    }

    // 添加
    for (int i = 0; i < numVecs; ++i) {
        for (int j = 0; j < dim_; j++){
            this->data32_->data<float>()[num_ * dim_ + j] = data[i * dim_ + j]
        }
        num_++;
    }

    // 预计算对应数据的平方并储存
    for (int i = 0; i < numVecs; ++i) {
        norm_.push_back(data[i] * data[i]);
    }
}



}
}