#include <faiss/hexagon/computeResidual.hpp>

#include <faiss/hexagon/readShader.hpp>
namespace faiss {
namespace hexagon{

// 推送常量结构体 (必须与着色器匹配)
struct ResidualPushConstants {
    uint32_t dim;
    uint32_t num_residuals;
};

void calResidual(
    kp::Manager* mgr,
    idx_t dim,
    std::shared_ptr<kp::TensorT<float>> vecs,
    std::shared_ptr<kp::TensorT<idx_t>> ids,
    std::shared_ptr<kp::TensorT<float>> src,
    std::shared_ptr<kp::TensorT<float>> residuals) {
    
    if ( !mgr || !vecs || !src || !residuals || !ids ) 
        return;
    if (ids->size() == 0)
        return;

    size_t expect_return_size = ids.size() * static_cast<size_t>(dim);
    if (residuals->size() < expect_return_size)
        return;

    // 准备推送常量
    ResidualPushConstants push_constants;
    push_constants.dim = static_cast<uint32_t>(dim);
    push_constants.num_residuals = static_cast<uint32_t>(ids.size());
    // 整理shader需要的张量
    std::vector<std::shared_ptr<kp::Tensor>> compute_params = {
        std::static_pointer_cast<kp::Tensor>(vecs),
        std::static_pointer_cast<kp::Tensor>(ids),
        std::static_pointer_cast<kp::Tensor>(src),
        std::static_pointer_cast<kp::Tensor>(residuals)
    };

    // 定义计算着色器的派发大小。
    const uint32_t WORKGROUP_SIZE_X = 256;
    uint32_t num_workgroups_x = (push_constants.num_residuals + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
    kp::Workgroup dispatch_size = {num_workgroups_x, 1, 1};

    // 创建shader
    auto spvShader = readSpvFile("/faiss/hexagon/shader/residual.spv");
    auto algorithm = mgr->algorithm(
        compute_params,
        spvShader,
        dispatch_size,
        std::vector<float>{},
        push_constants
    );

    // 记录操作同步到GPU
    auto seq = mgr->sequence()
        ->record<kp::OpTensorSyncDevice>({                          // CPU数据同步GPU
            std::static_pointer_cast<kp::Tensor>(vecs),
            std::static_pointer_cast<kp::Tensor>(ids),
            std::static_pointer_cast<kp::Tensor>(src)})
        ->record<kp::OpAlgoDispatch>(algorithm)                     // shader运算
        ->record<kp::OpTensorSyncLocal>({
            std::static_pointer_cast<kp::Tensor>(residuals)});      // GPU数据同步会CPU

    seq->eval();

    return;
}

}
}