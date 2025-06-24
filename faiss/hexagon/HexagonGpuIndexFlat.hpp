#pragma once

#include <faiss/cpu/Index.h>
#include <faiss/hexagon/FlatIndex.hpp>
// Kompute头文件
#include <kompute/Kompute.hpp>

namespace faiss {
namespace hexagon {

struct HexagonGpuIndexConfig {
    int dump = 1145;         // 占位，与原代码结构保持一致
};

class HexagonGpuIndex : public faiss::Index {
    public:
        HexagonGpuIndex(        // √
            std::shared_ptr<kp::Manager> mgr,
            int dims,
            faiss::MetricType metric,
            float metricArg
        );

        std::shared_ptr<kp::Manager> getKomputeManager();  // √

        size_t getNumVecs();  // √

        void reset(uint dim);  // √

        void train(idx_t n, const float* x);  // √

        void reconstruct(idx_t key, float* out);  // √

        void reconstruct_n(idx_t i0, idx_t num, float* out);  // √

        void reconstruct_batch(idx_t n, const idx_t* keys, float* out);  // √

        void add(idx_t n, const float* x);  // √

        void add_with_ids(idx_t n, const float* x, const idx_t* ids = nullptr);  // √

        void assign(idx_t n, const float* x, idx_t* labels, idx_t k = 1);

        void search(    // √
                idx_t n,
                const float* x,
                idx_t k,
                float* distances,
                idx_t* labels,
                const SearchParameters* params = nullptr);

        /// `x`, `distances` and `labels` and `recons` can be resident on the CPU or
        /// any GPU; copies are performed as needed
        void search_and_reconstruct(
                idx_t n,
                const float* x,
                idx_t k,
                float* distances,
                idx_t* labels,
                float* recons,
                const SearchParameters* params = nullptr);

        void compute_residual(const float* x, float* residual, idx_t key);  // √

        void compute_residual_n(  // √
                idx_t n,
                const float* xs,
                float* residuals,
                const idx_t* keys);

    protected:

        virtual void addImpl_(idx_t n, const float* x, const idx_t* ids = nullptr);  // √

        virtual void searchImpl_(       // √
                idx_t n,
                const float* x,
                int k,
                float* distances,
                idx_t* labels,
                const SearchParameters* params);
        
        virtual void resetIndex_(uint dim);     // √

        std::shared_ptr<kp::Manager> mgr_;
        const HexagonGpuIndexConfig config_;
        std::shared_ptr<hexagon::FlatIndex> data_;
        // size_t minPagedSize_;
};

}
}