#pragma once

#include <faiss/cpu/MetricType.h>

#include <kompute/Kompute.hpp>

namespace faiss {
namespace hexagon {
    class FlatIndex {
        public:
            FlatIndex(kp::Manager* mgr, int dim, bool useFloat16 = false);
    
            bool getUseFloat16() const;

            // 返回已经储存的向量数目
            idx_t getSize() const;

            int getDim() const;

            void reserve(size_t numVecs);

            kp::Tensor& getVectorFloat32Ref();

            kp::Tensor& getVectorsFloat16Ref();

            virtual void query(
                std::shared_ptr<kp::tensorT<float>> vecs,
                int k,
                faiss::MetricType metric,
                float metricArg,
                std::shared_ptr<kp::tensorT<float>> outDistance,
                std::shared_ptr<kp::tensorT<idx_t>> outIndices,
                bool exactDistance = false
            );

            // query 16版本

            void FlatIndex::computeResidual(
                std::shared_ptr<kp::tensorT<float>> vecs,
                std::shared_ptr<kp::tensorT<idx_t>> ids,
                std::shared_ptr<kp::tensorT<float>> residuals);

            void reconstruct(
                std::vector<idx_t> ids,
                std::shared_ptr<kp::tensorT<float>> vecs
            );

            void reconstruct(
                idx_t start,
                idx_t num,
                std::shared_ptr<kp::tensorT<idx_t>> vecs
            );

            void add(
                const float* data, 
                idx_t numVecs
            );

        protected:
            kp::Manager* mgr_;
            const int dim_;
            const bool useFloat16_;
            //memory type
            idx_t num_;             // 以vector为单位
            idx_t capacity_;
            std::shared_ptr<kp::TensorT<float>> data32_;
            std::shared_ptr<kp::TensorT<float>> data16_;
            std::vector<float> norm_;
    }



}
}
