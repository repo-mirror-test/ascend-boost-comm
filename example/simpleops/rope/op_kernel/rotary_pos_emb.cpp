/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "../tiling_data.h"
#include "../../utils/common/kernel/kernel_utils.h"
// #include "common/kernel/kernel_utils.h"
#include "rotary_pos_emb_fp16.h"
#include "rotary_pos_emb_fp32.h"
#include "rotary_pos_emb_bf16.h"

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, SimpleOps::RopeTilingData *tilingdata,
                                      AscendC::TPipe *pipe)
{
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(SimpleOps::RopeTilingData), pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->hiddenSizeQ = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->hiddenSizeK = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->headDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->headNumQ = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->headNumK = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->rotaryCoeff = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->ntokens = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->realCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 28));
    tilingdata->cosFormat = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->batch = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 36));
    tilingdata->maxUbSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 40));
    AscendC::PipeBarrier<PIPE_ALL>();
}

#define GET_TILING_DATA(tiling_data, tiling_arg, pipe)                                                                 \
    SimpleOps::RopeTilingData tiling_data;                                                                                \
    InitTilingData(tiling_arg, &(tiling_data), &(pipe))

extern "C" __global__ __aicore__ void rotary_pos_emb(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin,
                                                     GM_ADDR seqLen, GM_ADDR outQ, GM_ADDR outK,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling, pipe);
    if (TILING_KEY_IS(30)) {
        RopeFp16<half, half, true> ropeFp16(&tiling_data, &pipe);
        ropeFp16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp16.Process(workspace);
    } else if (TILING_KEY_IS(31)) {
        RopeFp32<half, float, true> ropeFp32(&tiling_data, &pipe);
        ropeFp32.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp32.Process(workspace);
    } else if (TILING_KEY_IS(32)) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        RopeBf16<bfloat16_t, bfloat16_t, true> ropeBf16(&tiling_data, &pipe);
        ropeBf16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeBf16.Process(workspace);
#endif
    } else if (TILING_KEY_IS(20)) {
        RopeFp16<half, half, false> ropeFp16(&tiling_data, &pipe);
        ropeFp16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp16.Process(workspace);
    } else if (TILING_KEY_IS(21)) {
        RopeFp32<half, float, false> ropeFp32(&tiling_data, &pipe);
        ropeFp32.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp32.Process(workspace);
    } else if (TILING_KEY_IS(22)) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        RopeBf16<bfloat16_t, bfloat16_t, false> ropeBf16(&tiling_data, &pipe);
        ropeBf16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeBf16.Process(workspace);
#endif
    }
}
