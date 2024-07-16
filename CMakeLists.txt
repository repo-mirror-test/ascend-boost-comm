cmake_minimum_required(VERSION 3.8)
project("mki")
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TEST_FRAMEWORK "BUILD_TEST_FRAMEWORK" OFF)
option(USE_PYTHON_TEST "USE_PYTHON_TEST" OFF)
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
message(STATUS "BUILD_TEST_FRAMEWORK:${BUILD_TEST_FRAMEWORK}")
message(STATUS "USE_PYTHON_TEST:${USE_PYTHON_TEST}")
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

file(GLOB_RECURSE SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/*/*.cpp")
list(FILTER SOURCE_FILES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/src/mki_loader/.*")

file(GLOB_RECURSE LOADER_SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/mki_loader/*.cpp")
# Add memset kernel to libmki
add_subdirectory(${PROJECT_SOURCE_DIR}/src/utils/memset/kernel)

set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)
add_library(mki_static STATIC ${SOURCE_FILES} ${BINARY_SRC_LIST})
add_dependencies(mki_static binary_encode_srcs)
target_link_libraries(mki_static PUBLIC dl pthread c_sec mmpa)
target_compile_definitions(mki_static PRIVATE fe=MkiFe ${NAMESPACE}=Mki)

add_library(mki SHARED ${SOURCE_FILES} ${BINARY_SRC_LIST})
add_dependencies(mki binary_encode_srcs)
target_link_libraries(mki PUBLIC dl pthread c_sec mmpa)
target_compile_definitions(mki PRIVATE fe=MkiFe ${NAMESPACE}=Mki)

add_library(mki_loader STATIC ${LOADER_SOURCE_FILES})
target_link_libraries(mki_loader PUBLIC mki dl pthread c_sec mmpa)
target_compile_definitions(mki_loader PRIVATE fe=${NAMESPACE}Fe)

add_executable(sym_check ${PROJECT_SOURCE_DIR}/src/sym_check.cpp)
target_link_libraries(sym_check PRIVATE mki_loader mki)

if(BUILD_TEST_FRAMEWORK)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests/framework)
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

# install libs
install(TARGETS mki_static DESTINATION lib)
install(TARGETS mki DESTINATION lib)
install(TARGETS mki_loader DESTINATION lib)
