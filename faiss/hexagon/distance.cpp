#include <kompute/Kompute.hpp>
#include <faiss/hexagon/distance.hpp>

namespace faiss{
namespace hexagon{
    void runL2Distance(
        kp::Manager* mgr,                                       // 资源管理器
        std::shared_ptr<kp::TensorT<float>> vecs,               // 库中的向量
        uint dim,                                               // 向量维度
        std::shared_ptr<kp::TensorT<float>> vecsNorm,           // 向量平方
        std::shared_ptr<kp::TensorT<float>> queries,            // 待查询的向量
        int k,                                                  // 每个向量保留k个最邻近的结果
        std::shared_ptr<kp::TensorT<float>> outDistances,       // 输出距离
        std::shared_ptr<kp::TensorT<uint>>  outIndices,         // 输出索引id
        bool ignoreOutDistance = false ) 
    {
        // TODO
        return;



    }
}
}
