#pragma once

#include <faiss/cpu/Index.h>

// Kompute头文件
#include <kompute/Kompute.hpp>

namespace faiss {
namespace hexagon {

struct HexagonGpuIndexConfig {
    int dump = 1145;         // 占位，与原代码结构保持一致
}

class HexagonGpuIndex : public faiss::Index {
    public:
        HexagonGpuIndex(
            std::shared_ptr<kompute::Manager> kmg,
            int dims,
            faiss::MetricType metric,
            float metricArg,
            HexagonGpuIndexConfig config = HexagonGpuIndexConfig()
        )

        std::shared_ptr<Kompute::Manager> getKomputeManager();

        void add(idx_t n, const float* x) override;

        void add_with_ids(idx_t n, const float* x, const idx_t* ids) override;

        void assign(idx_t n, const float* x, idx_t* labels, idx_t k = 1) const override;

        void search(
                idx_t n,
                const float* x,
                idx_t k,
                float* distances,
                idx_t* labels,
                const SearchParameters* params = nullptr) const override;

        /// `x`, `distances` and `labels` and `recons` can be resident on the CPU or
        /// any GPU; copies are performed as needed
        void search_and_reconstruct(
                idx_t n,
                const float* x,
                idx_t k,
                float* distances,
                idx_t* labels,
                float* recons,
                const SearchParameters* params = nullptr) const override;

        void compute_residual(const float* x, float* residual, idx_t key)
        const override;

        void compute_residual_n(
                idx_t n,
                const float* xs,
                float* residuals,
                const idx_t* keys) const override;

    protected:

        virtual void addImpl_(idx_t n, const float* x, const idx_t* ids) = 0;

        virtual void searchImpl_(
                idx_t n,
                const float* x,
                int k,
                float* distances,
                idx_t* labels,
                const SearchParameters* params) const = 0;
        
        std::shared_ptr<kompute::Manager> kmg_;
        const HexagonGpuIndexConfig config_;
        // size_t minPagedSize_;
}

}
}