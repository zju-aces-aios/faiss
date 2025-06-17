#pragma once

#include <vector>
#include <string>
#include <fstream>  // 用于文件流操作
#include <iostream> // 用于错误输出 (你可以替换为你项目中的日志系统)
#include <cstdint>  // 用于 uint32_t

namespace faiss {
namespace hexagon {

std::vector<uint32_t> readSpvFile(const std::string& filename);

}

}