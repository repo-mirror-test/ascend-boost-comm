cmake_minimum_required(VERSION 3.20)
project("mki")
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TEST_FRAMEWORK "BUILD_TEST_FRAMEWORK" OFF)
option(BUILD_SELF_TEST "BUILD_SELF_TEST" OFF)
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
message(STATUS "BUILD_TEST_FRAMEWORK:${BUILD_TEST_FRAMEWORK}")
message(STATUS "BUILD_SELF_TEST:${BUILD_SELF_TEST}")
message(STATUS "USE_CXX11_ABI:${USE_CXX11_ABI}")

include(${PROJECT_SOURCE_DIR}/cmake/host_config.cmake)
set(ASCEND_DRIVER_DIR /usr/local/Ascend/driver)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/3rdparty/metadef/inc/external
    ${PROJECT_SOURCE_DIR}/3rdparty/nlohmannJson/include
    ${ASCEND_DRIVER_DIR}/kernel/libc_sec/include
)

link_directories(
    ${ASCEND_DRIVER_DIR}/lib64/common
    $ENV{ASCEND_HOME_PATH}/lib64
)

add_subdirectory(src)

if(BUILD_TEST_FRAMEWORK)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests/framework)
endif()

if(BUILD_SELF_TEST)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests/selftest)
endif()

# cmake install stage
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/output/mki" CACHE PATH "..." FORCE)
endif()

# cp files
install(FILES
    ${PROJECT_SOURCE_DIR}/scripts/set_env.sh DESTINATION ./)
file(GLOB_RECURSE SCRIPTS_FILES "${PROJECT_SOURCE_DIR}/scripts/*.py")
list(APPEND SCRIPTS_FILES
    ${PROJECT_SOURCE_DIR}/scripts/build_env_setup.sh
    ${PROJECT_SOURCE_DIR}/scripts/build_ops.sh
    ${PROJECT_SOURCE_DIR}/scripts/set_ops_env.sh
)
install(FILES ${SCRIPTS_FILES} DESTINATION ./scripts)
install(DIRECTORY
    ${PROJECT_SOURCE_DIR}/src/include DESTINATION .)
