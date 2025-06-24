#include <kompute/Kompute.hpp>
#include <faiss/hexagon/distance.hpp>
#include <faiss/hexagon/readShader.hpp>
#include <faiss/cpu/MetricType.h>

namespace faiss{
namespace hexagon{

void calDistance (
    kp::Manager* mgr,                                       // 资源管理器
    bool isL2,                                              // 是否计算L2，否则计算IP
    uint dim,                                               // 向量维度
    int k,                                                  // 应该返回的k个向量
    int numVecs,                                            // 向量数量
    int numQueries,                                         // 查询向量数量
    std::shared_ptr<kp::TensorT<float>> vecs,               // 原向量           numVecs    * dim
    std::shared_ptr<kp::TensorT<float>> vecsNorm,           // 原向量Norm数据   numVecs    * dim
    std::shared_ptr<kp::TensorT<float>> queries,            // 待查询向量       numQueries * dim
    std::shared_ptr<kp::TensorT<float>> outDistances,       // 输出距离         numQueryes * k
    std::shared_ptr<kp::TensorT<idx_t>> outIndices,         // 输出id           numQueries * k
    bool ignoreOutDistance
) {
    // 首先计算IP，即内积
    // numQueries * dim dot (numVecs * dim)^T
    // 结果是 numQueries * numVecs
    auto IP = mgr->tensorT<float>(std::vector<float>(numQueries * numVecs, 0.0f), kp::Tensor::TensorTypes::eDevice);
    matmul(mgr, queries, vecs, IP, false, numQueries, dim, numVecs);

    // 如果只计算内积，则进行排序并返回
    if (!isL2) {
        for (int i = 0; i < numQueries; ++i) {
            // 对每个查询向量的内积结果进行排序
            std::vector<std::pair<float, idx_t>> ipPairs;
            for (int j = 0; j < numVecs; ++j) {
                ipPairs.emplace_back(IP->data()[i * numVecs + j], j);
            }
            std::sort(ipPairs.begin(), ipPairs.end(), std::greater<>());
            
            // 将前k个结果存入输出张量
            for (int j = 0; j < k; ++j) {
                outDistances->data()[i * k + j] = ipPairs[j].first;
                outIndices->data()[i * k + j] = ipPairs[j].second;
            }
        }
        return;
    }

    // 否则，需要计算L2距离
    // 首先检查Norm是否存在，如果不存在，则计算
    if (!vecsNorm) {
        vecsNorm = mgr->tensorT<float>(std::vector<float>(numVecs * dim, 0.0f), kp::Tensor::TensorTypes::eDevice);
        norm(mgr, vecsNorm, vecs, dim * numVecs);
    }
    auto queriesNorm = mgr->tensorT<float>(std::vector<float>(numQueries * dim, 0.0f), kp::Tensor::TensorTypes::eDevice);
    norm(mgr, queriesNorm, queries, dim * numQueries);

    // 计算L2距离（不计算开根号）
    auto L2Distances = mgr->tensorT<float>(std::vector<float>(numQueries * numVecs, 0.0f), kp::Tensor::TensorTypes::eDevice);
    calL2(mgr, vecsNorm, queriesNorm, IP, L2Distances, numQueries, dim, numVecs);

    // 对L2距离进行排序，并选择前k个结果储存
    for (int i = 0; i < numQueries; ++i) {
        // 对每个查询向量的L2距离结果进行排序
        std::vector<std::pair<float, idx_t>> l2Pairs;
        for (int j = 0; j < numVecs; ++j) {
            l2Pairs.emplace_back(L2Distances->data()[i * numVecs + j], j);
        }
        std::sort(l2Pairs.begin(), l2Pairs.end());

        // 将前k个结果存入输出张量
        for (int j = 0; j < k; ++j) {
            outDistances->data()[i * k + j] = l2Pairs[j].first;
            outIndices->data()[i * k + j] = l2Pairs[j].second;
        }
    }
    return;

}

void norm(
    kp::Manager* mgr,                                       // 资源管理器
    std::shared_ptr<kp::TensorT<float>> v,                 // 原向量1
    std::shared_ptr<kp::TensorT<float>> outNorms,           // 输出向量平方
    uint dim                                                // 向量维度
) {
    // TODO

}

void matmul(
    kp::Manager* mgr,                                       // 资源管理器
    std::shared_ptr<kp::TensorT<float>> a,                  // 矩阵A
    std::shared_ptr<kp::TensorT<float>> b,                  // 矩阵B
    std::shared_ptr<kp::TensorT<float>> out,                // 输出矩阵
    bool transB,                                        // 是否转置第二个矩阵
    uint m,                                             // A的行数
    uint k,                                             // A的列数
    uint n                                              // B的列数
) {
    auto spvShader = readSpvFile("/faiss/hexagon/shader/matmul.spv");

    std::vector<uint32_t> pushConsts = {
        static_cast<uint32_t>(m),
        static_cast<uint32_t>(k),
        static_cast<uint32_t>(n)
    };

    std::vector<std::shared_ptr<kp::Tensor>> memories = {a, b, out};

    auto algorithm = mgr->algorithm(
        memories,
        spvShader,
        kp::Workgroup({static_cast<uint32_t>(m), static_cast<uint32_t>(n), 1}),  // 工作组大小
        std::vector<float>{},
        pushConsts
    );

    auto seq = mgr->sequence()
        ->record<kp::OpTensorSyncDevice>(memories)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpTensorSyncLocal>({out});

    seq->eval();  // 执行序列
}

void calL2 (
    kp::Manager* mgr,                                       // 资源管理器
    std::shared_ptr<kp::TensorT<float>> vecsNorm,           // 原向量Norm数据           n * k
    std::shared_ptr<kp::TensorT<float>> queriesNorm,        // 查询向量Norm数据         m * k
    std::shared_ptr<kp::TensorT<float>> IP,                 // 内积结果                 m * n
    std::shared_ptr<kp::TensorT<float>> outDistances,       // 输出距离                 m * n
    int m,                                                  // 应该返回的k个向量
    int k,                                                  // 向量数量
    int n                                                   // 查询向量数量
) {
    // TODO

}

}
}
