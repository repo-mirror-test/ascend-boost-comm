
include_directories(
    ${PROJECT_SOURCE_DIR}/tests/autogen
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

# CPP unittest testframework
file(GLOB_RECURSE TEST_UTIL_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/test_util/*.cpp")
add_library(mki_test_util SHARED ${TEST_UTIL_SOURCE_FILES})
target_include_directories(mki_test_util PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(mki_test_util PUBLIC mki torch torch_cpu c10)
install(TARGETS mki_test_util DESTINATION tests)

# Pytorch testframework
## json->param lib
add_library(mki_test_autogen SHARED ${CMAKE_CURRENT_LIST_DIR}/../autogen/op_desc_json_ph.cpp)
if(NOT EXISTS ${CMAKE_INSTALL_PREFIX}/tests/libmki_test_autogen.so)
    MESSAGE(STATUS "libmki_test_autogen.so does NOT exist. Generate .so placeholder.")
    install(TARGETS mki_test_autogen DESTINATION tests)
endif()
## main framework
file(GLOB_RECURSE TORCH_EXTENSION_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/torch_extension/mki_torch.cpp")
add_library(mki_torch SHARED ${TORCH_EXTENSION_SOURCE_FILES})
target_link_libraries(mki_torch PUBLIC mki_test_util torch_python torch_npu mki_test_autogen)
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
    DESTINATION tests/include/autogen
    FILES_MATCHING PATTERN "*.h"
)

install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/test_util/
    DESTINATION tests/include/test_util
    FILES_MATCHING PATTERN "*.h"
)

install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/kerneltest_extension/
    DESTINATION tests/include/kerneltest_extension
    FILES_MATCHING PATTERN "*.h"
)
