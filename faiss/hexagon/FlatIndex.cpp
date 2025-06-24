#include <faiss/hexagon/FlatIndex.hpp>
#include <faiss/hexagon/computeResidual.hpp>

#include <memory>
#include <math.h>
namespace faiss {
namespace hexagon{

FlatIndex::FlatIndex(kp::Manager* mgr, int dim, bool useFloat16 = false) {
    this->mgr_ = mgr;
    this->dim_ = dim;
    this->useFloat16_ = useFloat16;
    this->data32_ = std::vector<float>();
    this->data16_ = std::vector<float>();
    this->capacity_ = 0;
    this->num_ = 0;
}

bool FlatIndex::getUseFloat16() const {
    return this->useFloat16_;
}

idx_t FlatIndex::getSize() const {
    if (!this->useFloat16_)
        return this->data32_.size() / this->dim_;
    else
        return this->data16_.size() / this->dim_;
}

int FlatIndex::getDim() const {
    return this->dim_;
}

void FlatIndex::reserve(size_t numVecs) {
    // 预留空间
    this->data32_.reserve(numVecs * dim_);
}

kp::Tensor& FlatIndex::getVectorFloat32Ref() {
    // TODO
    return;
}

kp::Tensor& FlatIndex::getVectorsFloat16Ref() {
    // TODO
    return;
}

void FlatIndex::computeResidual(
    const float* vecs,
    const idx_t* ids,
    float* residuals) {
    // 计算残差，在次数把需要的张量拼接好，并把指针传递给对应的函数

}

void FlatIndex::reconstruct(std::vector<idx_t> ids, float* vecs) {
    // 根据id返回对应的向量
    if (!this->useFloat16_) {
        for (int i = 0; i < ids.size(); ++i) {
            // 检查是否越界
            if (ids[i] < 0 || ids[i] >= data32_.size() / dim_)
                continue;
            memcpy( vecs + i * this->dim_ * sizeof(float), 
                    this->data32_.data() + ids[i] * this->dim_ * sizeof(float),
                    this->dim_ * sizeof(float));
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
    float* vecs) {
    // 把一段连续的数据恢复到vecs中    
    if (!this->useFloat16_) {
        memcpy( vecs, 
                this->data32_.data() + start * this->dim_ * sizeof(float), 
                num * this->dim_ * sizeof(float));
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
        this->data32_.resize(capacity_ * dim_);
        this->norm_.reserve(capacity_ * dim_);
    }

    // 添加
    memcpy(this->data32_.data() + num_ * dim_, data, numVecs * dim_ * sizeof(float));
    // 预计算对应数据的平方并储存
    for (int i = 0; i < numVecs * dim_; ++i) {
        norm_[(num_ - 1) * dim_ + i] = data[i] * data[i];
    }

    num_ += numVecs;
}



}
}