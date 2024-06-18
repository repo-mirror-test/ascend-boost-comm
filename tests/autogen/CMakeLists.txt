cmake_minimum_required(VERSION 3.8)

project("mki_test_autogen")

option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
message(STATUS "USE_CXX11_ABI:${USE_CXX11_ABI}")

include($ENV{CODE_ROOT}/cmake/host_config.cmake)

include_directories(
    $ENV{CODE_ROOT}
    $ENV{CODE_ROOT}/include
    $ENV{THIRD_PARTY_DIR}/nlohmannJson/include
)

link_directories(
    $ENV{CODE_ROOT}/lib
)

execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/op_desc_json_auto.py)
add_library(mki_test_autogen SHARED ${CMAKE_CURRENT_LIST_DIR}/op_desc_json.cpp)
target_link_libraries(mki_test_autogen PRIVATE mki)

install(TARGETS mki_test_autogen DESTINATION tests)
