set(kvcache_srcs
    ${CMAKE_CURRENT_LIST_DIR}/kvcache_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kvcache_kernel.cpp
)

add_operation(KVCacheOperation "${kvcache_srcs}")

add_kernel(kvcache ascend910b vector
    op_kernel/kvcache.cpp
    KVCacheNdKernel)

add_kernel(kvcache_nz ascend310p vector
    op_kernel/kvcache_nz.cpp
    KVCacheNzKernel)

add_kernel(kvcache_dynamic_batch ascend910b vector
    op_kernel/kvcache_dynamic_batch.cpp
    KVCacheNdDynamicBatchKernel)
