# cmake_minimum_required(VERSION 3.8)
# project("mki_test")
# set(CMAKE_CXX_STANDARD 17)

# option(BUILD_TEST_FRAMEWORK "BUILD_TEST_FRAMEWORK" OFF)
# option(USE_PYTHON_TEST "USE_PYTHON_TEST" OFF)
# option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
# message(STATUS "BUILD_TEST_FRAMEWORK:${BUILD_TEST_FRAMEWORK}")
# message(STATUS "EXAMPLE_TEST:${EXAMPLE_TEST}")
# message(STATUS "USE_PYTHON_TEST:${USE_PYTHON_TEST}")
# message(STATUS "USE_CXX11_ABI:${USE_CXX11_ABI}")

# set(NAMESPACE "SimpleOps")

# include($ENV{CODE_ROOT}/cmake/host_config.cmake)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/../autogen
    $ENV{PYTHON_INCLUDE_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    $ENV{PYTORCH_NPU_INSTALL_PATH}/include
)

link_directories(
    $ENV{PYTHON_LIB_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/lib
    $ENV{PYTORCH_NPU_INSTALL_PATH}/lib
)

# Pytorch testframework
## json->param lib
# add_subdirectory(autogen)
add_library(mki_test_autogen SHARED ${CMAKE_CURRENT_LIST_DIR}/../autogen/op_desc_json_ph.cpp)
if(NOT EXISTS ${CMAKE_INSTALL_PREFIX}/tests/libmki_test_autogen.so)
    MESSAGE(STATUS "libmki_test_autogen.so does NOT exist. Generate placeholder.")
    install(TARGETS mki_test_autogen DESTINATION tests)
endif()
## main framework
file(GLOB_RECURSE TORCH_EXTENSION_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/torch_extension/mki_torch.cpp")
add_library(mki_torch SHARED ${TORCH_EXTENSION_SOURCE_FILES})
target_link_libraries(mki_torch PUBLIC mki_test_autogen mki torch torch_cpu c10 torch_python torch_npu)
target_compile_options(mki_torch PRIVATE
    -Wno-unused-function
    -Wno-attributes
    -Wno-maybe-uninitialized
)
install(TARGETS mki_torch DESTINATION tests)

# CPP kerneltest not supported yet!!!
# add_library(mki_kernel SHARED
#     kerneltest_extention/kerneltest_extention.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/torch_extension/op_desc_json.cpp
# )
# target_include_directories(mki_kernel PUBLIC torch_extension $ENV{ASCEND_HOME_PATH}/include)
# target_link_libraries(mki_kernel PUBLIC mki ascendcl)
# install(TARGETS mki_kernel DESTINATION lib)

# cp header files
install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../autogen/
    DESTINATION tests/autogen
    FILES_MATCHING
    PATTERN "*.txt"
    PATTERN "*.py"
    PATTERN "*.h"
)

# install(
#     DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/test_util/
#     DESTINATION tests/include/test_util
#     FILES_MATCHING PATTERN "*.h"
# )

# install(
#     DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/kerneltest_extension/
#     DESTINATION tests/include/kerneltest_extension
#     FILES_MATCHING PATTERN "*.h"
# )

install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../pythontest/
    DESTINATION tests/pythontest/
    FILES_MATCHING PATTERN "*.py"
)
