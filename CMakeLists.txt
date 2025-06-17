# @lint-ignore-every LICENSELINT
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)
include(FetchContent)

set(FAISS_LANGUAGES CXX)

if(FAISS_ENABLE_GPU)
  if (FAISS_ENABLE_ROCM)
    list(APPEND FAISS_LANGUAGES HIP)
    list(PREPEND CMAKE_MODULE_PATH "/opt/rocm/lib/cmake")
    list(PREPEND CMAKE_PREFIX_PATH "/opt/rocm")
  else()
    list(APPEND FAISS_LANGUAGES CUDA)
  endif()
endif()

if(FAISS_ENABLE_CUVS)
  include(cmake/thirdparty/fetch_rapids.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)

  rapids_cuda_init_architectures(faiss)
  rapids_cuda_init_architectures(pyfaiss)
  rapids_cuda_init_architectures(faiss_c_library)
endif()

project(faiss
  VERSION 1.10.0
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES ${FAISS_LANGUAGES})
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Valid values are "generic", "avx2", "avx512", "avx512_spr", "sve".
option(FAISS_OPT_LEVEL "" "generic")
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." OFF)
option(FAISS_ENABLE_CUVS "Enable cuVS for GPU indexes." OFF)
option(FAISS_ENABLE_ROCM "Enable ROCm for GPU indexes." OFF)
option(FAISS_ENABLE_MKL "Enable MKL." ON)
option(FAISS_ENABLE_PYTHON "Build Python extension." ON)
option(FAISS_ENABLE_C_API "Build C API." OFF)
option(FAISS_ENABLE_EXTRAS "Build extras like benchmarks and demos" ON)
option(FAISS_USE_LTO "Enable Link-Time optimization" OFF)
option(FAISS_ENABLE_HEXAGON "Enable support for Hexagon indexes using Kompute." ON)


if(FAISS_ENABLE_GPU)
  if(FAISS_ENABLE_ROCM)
    enable_language(HIP)
    add_definitions(-DUSE_AMD_ROCM)
    find_package(HIP REQUIRED)
    find_package(hipBLAS REQUIRED)
    set(GPU_EXT_PREFIX "hip")
    execute_process(COMMAND ${PROJECT_SOURCE_DIR}/faiss/gpu/hipify.sh)
  else ()
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    enable_language(CUDA)
    set(GPU_EXT_PREFIX "cu")
  endif()
endif()

# 添加Kompute依赖管理
if(FAISS_ENABLE_HEXAGON)
  message(STATUS "Enabling Hexagon support with Kompute")
  
  # 使用FetchContent自动获取Kompute
  FetchContent_Declare(
    kompute
    GIT_REPOSITORY https://github.com/KomputeProject/Kompute.git
    GIT_TAG        v0.9.0  # 指定稳定版本
    GIT_SHALLOW    TRUE    # 浅克隆以节省时间
  )
  
  # 设置Kompute编译选项
  set(KOMPUTE_OPT_BUILD_TESTS OFF CACHE BOOL "Build Kompute tests")
  set(KOMPUTE_OPT_BUILD_DOCS OFF CACHE BOOL "Build Kompute documentation")
  set(KOMPUTE_OPT_BUILD_EXAMPLES OFF CACHE BOOL "Build Kompute examples")
  set(KOMPUTE_OPT_BUILD_SINGLE_HEADER OFF CACHE BOOL "Build single header")
  
  # 获取并构建Kompute
  FetchContent_MakeAvailable(kompute)
  
  # 检查是否成功获取
  if(NOT TARGET kompute)
    message(FATAL_ERROR "Failed to fetch and build Kompute")
  endif()
  
  message(STATUS "Kompute successfully integrated")
endif()

if(FAISS_ENABLE_CUVS AND NOT TARGET cuvs::cuvs)
   find_package(cuvs)
endif()

add_subdirectory(faiss/cpu)

if(FAISS_ENABLE_GPU)
  if(FAISS_ENABLE_ROCM)
    add_subdirectory(faiss/gpu-rocm)
  else()
    add_subdirectory(faiss/gpu)
  endif()
endif()

# 添加hexagon子目录
if(FAISS_ENABLE_HEXAGON)
  add_subdirectory(faiss/hexagon)
endif()

if(FAISS_ENABLE_PYTHON)
  add_subdirectory(faiss/python)
endif()

if(FAISS_ENABLE_C_API)
  add_subdirectory(c_api)
endif()

if(FAISS_ENABLE_EXTRAS)
  add_subdirectory(demos)
  add_subdirectory(benchs)
  add_subdirectory(tutorial/cpp)
endif()

# CTest must be included in the top level to enable `make test` target.
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
  add_subdirectory(perf_tests)
  if(FAISS_ENABLE_GPU)
    if(FAISS_ENABLE_ROCM)
      add_subdirectory(faiss/gpu-rocm/test)
    else()
      add_subdirectory(faiss/gpu/test)
    endif()
  endif()
endif()


# cmake ..     -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_HEXAGON=ON    -DFAISS_ENABLE_PYTHON=OFF     -DBUILD_TESTING=ON     -DCMAKE_BUILD_TYPE=Release