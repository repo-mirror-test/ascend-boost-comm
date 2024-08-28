set(unpad_srcs
    ${CMAKE_CURRENT_LIST_DIR}/unpad_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unpad_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/unpad_tiling.cpp
)

add_operation(UnpadOperation "${unpad_srcs}")


add_kernel(unpad ascend910b vector
    op_kernel/unpad.cpp
    UnpadKernel)

add_kernel(unpad ascend310p vector
    op_kernel/unpad.cpp
    UnpadKernel)
