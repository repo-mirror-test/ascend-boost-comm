if (OPS_OUTPUT_DIR EQUAL "")
    set(OPS_OUTPUT_DIR "$ENV{OUTPUT_DIR}")
endif()

if (MKI_PACKAGE_DIR EQUAL "")
    set(MKI_PACKAGE_DIR "$ENV{MKI_DIR}")
endif()

set(MKI_SCRIPT_DIR "${MKI_PACKAGE_DIR}/scripts")

if(NOT EXISTS ${OP_LIST_YAML_DIR}/op_list.yaml)
    execute_process(COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/op_list_utils.py -s ${PROJECT_SOURCE_DIR} -d ${OP_LIST_YAML_DIR}
                    ERROR_VARIABLE RESULT_INFO
                    RESULT_VARIABLE RESULT)
    if(NOT RESULT EQUAL 0)
        message(FATAL_ERROR "create op_build.yaml failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
    endif()
endif()

set(OP_LIST_CMD_PARAMS "'${OP_LIST_YAML_DIR}/op_list.yaml', '${MKI_PACKAGE_DIR}/cmake/op_build.cmake'")
execute_process(COMMAND python3 -c "import op_list_utils\nop_list_utils.build_cmake_options(${OP_LIST_CMD_PARAMS})"
                WORKING_DIRECTORY ${MKI_PACKAGE_DIR}/scripts
                ERROR_VARIABLE RESULT_INFO
                RESULT_VARIABLE RESULT)
if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "create op_build.cmake failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
endif()

include(${MKI_PACKAGE_DIR}/cmake/op_build.cmake)
include(${MKI_PACKAGE_DIR}/cmake/kernel_config.cmake)

set(ASCEND_DRIVER_DIR /usr/local/Ascend/driver)

include_directories(
    ${MKI_PACKAGE_DIR}
    ${MKI_PACKAGE_DIR}/include
    ${OPS_THIRD_PARTY_DIR}/metadef/inc/external
    ${OPS_THIRD_PARTY_DIR}/compiler/tikcpp/tikcfw
    ${OPS_THIRD_PARTY_DIR}/nlohmannJson/include
    ${OPS_THIRD_PARTY_DIR}/securec/include
    ${OPS_PROJECT_ROOT_DIR}/${CMAKE_PROJECT_NAME}/include
    ${ASCEND_DRIVER_DIR}/kernel/libc_sec/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl
)

link_directories(
    ${MKI_PACKAGE_DIR}/lib
    ${ASCEND_DRIVER_DIR}/lib64/common
    $ENV{ASCEND_HOME_PATH}/lib64
)

include_directories(
    ${OPS_THIRD_PARTY_DIR}/metadef
    ${OPS_THIRD_PARTY_DIR}/metadef/exe_graph/runtime
    ${OPS_THIRD_PARTY_DIR}/metadef/graph
    ${OPS_THIRD_PARTY_DIR}/metadef/inc
    ${OPS_THIRD_PARTY_DIR}/metadef/inc/external
    ${OPS_THIRD_PARTY_DIR}/metadef/inc/external/exe_graph/runtime
    ${OPS_THIRD_PARTY_DIR}/metadef/inc/external/graph
    ${OPS_THIRD_PARTY_DIR}/metadef/register
)

file(GLOB_RECURSE CMAKELISTS_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
string(REPLACE "/CMakeLists.txt" "" CMAKELISTS_PATHS "${CMAKELISTS_PATHS}")
foreach(SUB_DIR ${CMAKELISTS_PATHS})
    message(STATUS "add subdirectory: ${SUB_DIR}")
    add_subdirectory(${SUB_DIR})
endforeach()

set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)
message("BINARY_SRC_LIST: ${BINARY_SRC_LIST}")
message("BINARY_TARGET_LIST: ${BINARY_TARGET_LIST}")

add_custom_command(
    OUTPUT ${BINARY_SRC_LIST}
    DEPENDS ${BINARY_TARGET_LIST}
    COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/build_util.py --binary_dir ${CMAKE_BINARY_DIR} --op_type=ascendc
)

add_custom_target(BINARY_SRC_TARGET ALL
    DEPENDS ${BINARY_SRC_LIST}
)

#########################
# 设置target名称
#########################
add_library(simple_ops SHARED ${BINARY_SRC_LIST})
add_dependencies(simple_ops BINARY_SRC_TARGET)
target_link_libraries(simple_ops PUBLIC)
target_link_libraries(simple_ops PUBLIC "-Wl,--whole-archive" mki_loader "-Wl,--no-whole-archive")
target_link_libraries(simple_ops PRIVATE ${ops_objects} mki)

add_library(simple_ops_static STATIC ${BINARY_SRC_LIST})
add_dependencies(simple_ops_static BINARY_SRC_TARGET)
target_link_libraries(simple_ops_static PUBLIC "-Wl,--whole-archive" mki_loader "-Wl,--no-whole-archive")
target_link_libraries(simple_ops_static PUBLIC)
target_link_libraries(simple_ops_static PRIVATE ${ops_objects} mki)

install(TARGETS simple_ops DESTINATION ops_lib)
install(TARGETS simple_ops_static DESTINATION ops_lib)
install(DIRECTORY include DESTINATION .)
