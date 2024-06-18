set(rope_srcs
    ${CMAKE_CURRENT_LIST_DIR}/rope_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rope_kernel.cpp
)

add_operation(RopeOperation "${rope_srcs}")

add_kernel(rotary_pos_emb ascend910b vector
    op_kernel/rotary_pos_emb.cpp
    RopeKernel)

add_kernel(rotary_pos_emb ascend310p vector
    op_kernel/rotary_pos_emb.cpp
    RopeKernel)
