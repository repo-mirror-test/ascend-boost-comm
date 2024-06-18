/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../tiling_data.h"
#include "rotary_pos_emb_fp16.h"
#include "rotary_pos_emb_fp32.h"
#include "rotary_pos_emb_bf16.h"
                          
inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, SimpleOps::RopeTilingData *tilingdata)
{
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdata_in_ub), p_tilingdata, 0, 1, 2, 0, 0);
    pipe_barrier(PIPE_ALL);
    tilingdata->hiddenSizeQ = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 0));
    tilingdata->hiddenSizeK = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4));
    tilingdata->headDim = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 8));
    tilingdata->headNumQ = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 12));
    tilingdata->headNumK = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 16));
    tilingdata->rotaryCoeff = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 20));
    tilingdata->ntokens = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 24));
    tilingdata->realCore = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 28));
    tilingdata->cosFormat = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 32));
    tilingdata->batch = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 36));
    tilingdata->maxUbSize = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 40));
    pipe_barrier(PIPE_ALL);
}
               
#define GET_TILING_DATA(tiling_data, tiling_arg)                                                                       \
    SimpleOps::RopeTilingData tiling_data;                                                                              \
    InitTilingData(tiling_arg, &(tiling_data))
  
extern "C" __global__ __aicore__ void rotary_pos_emb(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin,
                                                     GM_ADDR seqLen, GM_ADDR outQ, GM_ADDR outK,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(30)) {
        RopeFp16<half, half, true> ropeFp16(&tiling_data);
        ropeFp16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp16.Process(workspace);
    } else if (TILING_KEY_IS(31)) {
        RopeFp32<half, float, true> ropeFp32(&tiling_data);
        ropeFp32.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp32.Process(workspace);
    } else if (TILING_KEY_IS(32)) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        RopeBf16<bfloat16_t, bfloat16_t, true> ropeBf16(&tiling_data);
        ropeBf16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeBf16.Process(workspace);
#endif
    } else if (TILING_KEY_IS(20)) {
        RopeFp16<half, half, false> ropeFp16(&tiling_data);
        ropeFp16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp16.Process(workspace);
    } else if (TILING_KEY_IS(21)) {
        RopeFp32<half, float, false> ropeFp32(&tiling_data);
        ropeFp32.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp32.Process(workspace);
    } else if (TILING_KEY_IS(22)) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        RopeBf16<bfloat16_t, bfloat16_t, false> ropeBf16(&tiling_data);
        ropeBf16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeBf16.Process(workspace);
#endif
    }
}
   
