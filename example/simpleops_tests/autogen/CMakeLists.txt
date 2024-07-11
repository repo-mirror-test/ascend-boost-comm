include_directories(
    ${MKI_PACKAGE_DIR}
    ${MKI_PACKAGE_DIR}/include
    ${OPS_THIRD_PARTY_DIR}/nlohmannJson/include
    ${OPS_PROJECT_ROOT_DIR}/${CMAKE_PROJECT_NAME}/include
    $ENV{ASCEND_HOME_PATH}/include
)

link_directories(
    ${MKI_PACKAGE_DIR}/lib
)

execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/op_desc_json_auto.py 
                                --namespace ${NAMESPACE}
                                --params_path ${OPS_PROJECT_ROOT_DIR}/${CMAKE_PROJECT_NAME}/include)
add_library(mki_test_autogen SHARED ${CMAKE_CURRENT_LIST_DIR}/op_desc_json.cpp)
target_link_libraries(mki_test_autogen PRIVATE simple_ops)
target_compile_definitions(mki_test_autogen PRIVATE OpSpace=${NAMESPACE})

install(TARGETS mki_test_autogen DESTINATION tests)
