/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 * http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "../iterator.h"
/////////////////////////////////////////////////////
// l0c_to_l1
/////////////////////////////////////////////////////

// Partial specialization ZN, half, int32_t
template <ArchType ArchTag>
struct l0c_to_l1<ArchTag, DataFormat::ZN, half, int32_t> {
    using ElementOut = half;
    using ElementIn = int32_t;
    __aicore__ l0c_to_l1(AscendC::LocalTensor<ElementOut> l1Tensor,
                         AscendC::LocalTensor<ElementIn> l0cTensor,
                         AscendC::LocalTensor<uint64_t> deqTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t mTileCeil,
                         uint32_t nActual)
    {
#ifdef USE_ASCENDC
        constexpr uint32_t BLOCK_NUM = 16;
        constexpr uint32_t BLOCK_SIZE = 32;
        AscendC::FixpipeParams<ElementIn> intriParams(
            (nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE),
            0,
            mTileCeil - static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE) *
                            sizeof(ElementOut) / sizeof(ElementIn));
        intriParams.nz2ndParams = {false, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::VDEQF16};
        AscendC::Fixpipe(l1Tensor, l0cTensor, deqTensor, intriParams);
#else
        copy_matrix_cc_to_cbuf((__cbuf__ ElementOut *)l1Tensor.GetPhyAddr(),
                               (__cc__ ElementIn *)l0cTensor.GetPhyAddr(),
                               0,           // sid
                               nTileActual, // NSize
                               mTileActual, // MSize
                               mTileCeil,   // dstStride_dst_D
                               mTileCeil,   // srcStride
                               0,           // UnitFlagMode
                               VDEQF16,     // QuantPRE VDEQF16, NoQuant
                               0,           // ReLUPRE
                               false,       // channelSplit
                               false        // NZ2ND_EN
        );
#endif
    };
};