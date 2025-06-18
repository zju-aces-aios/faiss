#include <kompute/Kompute.hpp>

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
        bool ignoreOutDistance = false        
    );

    /* 
        1. 检查vecsNorm是否存在
        2. 如果需要计算L2，预先计算vecsNorm与queryNorm    -----> calNorm() todo
        3. 计算内积并储存                                -----> calIP()   todo
        4. 如果只需要计算内积，此处计算结束
        5. 计算L2（a1^2 + b1^2 - 2a1b1, ...）           ------> calL2()   todo

        TODO 这些计算需要编写shader在GPU上进行，可能需要考虑分块（tile）问题，以免爆显存
    */
    void calDistance {
        kp::Manager* mgr,                                       // 资源管理器
        bool isL2,                                              // 是否计算L2，否则计算IP
        uint dim,                                               // 向量维度
        int k,                                                  // 应该返回的k个向量
        std::shared_ptr<kp::TensorT<float>> vecs,               // 原向量
        std::shared_ptr<kp::TensorT<float>> vecsNorm,           // 原向量Norm数据
        std::shared_ptr<kp::TensorT<float>> queries,            // 待查询向量
        std::shared_ptr<kp::TensorT<float>> outDistances,       // 输出距离
        std::shared_ptr<kp::TensorT<float>> outIndices,         // 输出id
        bool ignoreOutDistance
    };
}
}

