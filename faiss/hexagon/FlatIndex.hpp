#pragma once

#include <faiss/cpu/MetricType.h>

#include <kompute/Kompute.hpp>

#include <vector>

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
                const float* vecs,
                int k,
                faiss::MetricType metric,
                float metricArg,
                float* outDistance,
                idx_t* outIndices,
                bool exactDistance = false      // 
            );

            // query 16版本

            void FlatIndex::computeResidual(
                const float* vecs,
                const idx_t* ids,
                float* residuals);

            void reconstruct(
                std::vector<idx_t> ids,
                float* vecs
            );

            void reconstruct(
                idx_t start,
                idx_t num,
                float* vecs
            );

            void add(
                const float* data, 
                idx_t numVecs
            );

        protected:
            kp::Manager* mgr_;
            int dim_;
            bool useFloat16_;
            //memory type
            idx_t num_;             // 以vector为单位
            idx_t capacity_;
            std::vector<float> data32_;
            std::vector<float> data16_;
            std::vector<float> norm_;
    };



}
}
