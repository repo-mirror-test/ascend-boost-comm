/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef KVCACHE_ND_INIT_TILING_H
#define KVCACHE_ND_INIT_TILING_H

#include "kernel_operator.h"
#include "../tiling_data.h"
#include "../../utils/common/kernel/kernel_utils.h"

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, SimpleOps::KVCacheTilingData *tilingData)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingData->batch = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingData->maxSeqLen = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingData->hiddenSize = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(SimpleOps::KVCacheTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingData->batch = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingData->maxSeqLen = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingData->hiddenSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#endif