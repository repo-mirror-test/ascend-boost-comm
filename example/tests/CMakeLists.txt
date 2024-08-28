include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/autogen
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

add_subdirectory(autogen)

install(DIRECTORY ${MKI_PACKAGE_DIR}/tests DESTINATION .
        PATTERN "libmki_test_autogen.so" EXCLUDE)