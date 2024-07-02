cmake_minimum_required(VERSION 3.10)
project("simpleops")

########################
# 算子开发者配置
########################
# 三方库所在路径
set(OPS_THIRD_PARTY_DIR "$ENV{THIRD_PARTY_DIR}")

# # 算子库输出路径
# set(OPS_OUTPUT_DIR "$ENV{OUTPUT_DIR}")

# 算子库安装路径
set(CMAKE_INSTALL_PREFIX "${OPS_OUTPUT_DIR}/simpleops")

# 算子库根目录
set(OPS_PROJECT_ROOT_DIR "${PROJECT_SOURCE_DIR}")

# # MKI安装路径
# set(MKI_PACKAGE_DIR "$ENV{MKI_DIR}")

# op_list.yaml所在路径
set(OP_LIST_YAML_DIR "${MKI_PACKAGE_DIR}/configs")

# ABI配置
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
message(STATUS "USE_CXX11_ABI:${USE_CXX11_ABI}")

# 算子库namespace
set(NAMESPACE "SimpleOps")

include(${MKI_PACKAGE_DIR}/cmake/host_config.cmake)

#########################
# 自定义配置项
#########################

add_subdirectory(simpleops)
