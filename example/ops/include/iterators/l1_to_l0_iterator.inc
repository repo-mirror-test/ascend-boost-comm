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
// l1_to_l0_a
/////////////////////////////////////////////////////

// Partial specialization for vector
template <ArchType ArchTag, typename DataType, bool IsTransPose>
struct l1_to_l0_a<ArchTag, DataType, IsTransPose, DataFormat::VECTOR, DataFormat::VECTOR> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);

    __aicore__ l1_to_l0_a(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t mTileCeil,
                          uint32_t kPartCeil,
                          uint32_t mSrcStride,
                          uint32_t kSrcStride,
                          uint32_t mDstStride,
                          uint32_t kDstStride)
    {
        AscendC::LoadData(l0Tensor,
                          l1Tensor,
                          AscendC::LoadData2dParams(0,           // baseIdx
                                                    kPartCeil,   // repeat
                                                    kSrcStride,  // srcStride
                                                    0,           // sid
                                                    kDstStride,  // dstStride
                                                    IsTransPose, // transpose
                                                    0));         // addrCalMode
    };
};

// Partial specialization for no transpose, not vector
template <ArchType ArchTag, typename DataType>
struct l1_to_l0_a<ArchTag, DataType, false, DataFormat::ZN, DataFormat::ZZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);
    static constexpr uint32_t BLOCK_NUM_PER_FRACTAL = HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

    __aicore__ l1_to_l0_a(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t mTileCeil,
                          uint32_t kPartCeil,
                          uint32_t mSrcStride,
                          uint32_t kSrcStride,
                          uint32_t mDstStride,
                          uint32_t kDstStride)
    {
        for (uint32_t i = 0; i < mTileCeil / BLOCK_NUM_PER_FRACTAL; i++) {
            AscendC::LoadData(l0Tensor[i * mDstStride * FRACTAL_SIZE],
                              l1Tensor[i * mSrcStride * FRACTAL_SIZE],
                              AscendC::LoadData2dParams(0,                                             // baseIdx
                                                        static_cast<uint16_t>(kPartCeil / BLOCK_SIZE), // repeat
                                                        kSrcStride,                                    // srcStride
                                                        0,                                             // sid
                                                        kDstStride - 1,                                // dstStride
                                                        false,                                         // transpose
                                                        0));                                           // addrCalMode
        }
    };
};

// Partial specialization for transpose, not vector
template <ArchType ArchTag, typename DataType>
struct l1_to_l0_a<ArchTag, DataType, true, DataFormat::ZN, DataFormat::ZZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);
    static constexpr uint32_t BLOCK_NUM_PER_FRACTAL = HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

    __aicore__ l1_to_l0_a(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t mTileCeil,
                          uint32_t kPartCeil,
                          uint32_t mSrcStride,
                          uint32_t kSrcStride,
                          uint32_t mDstStride,
                          uint32_t kDstStride)
    {
        for (uint32_t i = 0; i < mTileCeil / BLOCK_SIZE; i++) {
            AscendC::LoadData(l0Tensor[i * mDstStride * FRACTAL_SIZE],
                              l1Tensor[i * mSrcStride * FRACTAL_SIZE],
                              AscendC::LoadData2dParams(0,
                                                        static_cast<uint16_t>(kPartCeil / BLOCK_NUM_PER_FRACTAL),
                                                        kSrcStride,
                                                        0,
                                                        kDstStride - 1,
                                                        true,
                                                        0));
        }
    };
};

/////////////////////////////////////////////////////
// l1_to_l0_b
/////////////////////////////////////////////////////

// Partial specialization for vector
template <ArchType ArchTag, typename DataType, bool IsTransPose>
struct l1_to_l0_b<ArchTag, DataType, IsTransPose, DataFormat::VECTOR, DataFormat::VECTOR> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);

    __aicore__ l1_to_l0_b(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t nTileCeil,
                          uint32_t kPartCeil,
                          uint32_t nSrcStride,
                          uint32_t kSrcStride,
                          uint32_t nDstStride,
                          uint32_t kDstStride)
    {
        AscendC::LoadData(
            l0Tensor, l1Tensor, AscendC::LoadData2dParams(0, kPartCeil, kSrcStride, 0, kDstStride, IsTransPose, 0));
    };
};

template <ArchType ArchTag>
struct l1_to_l0_b<ArchTag, int8_t, true, DataFormat::NZ, DataFormat::ZN> {
    using HardwareParams = HardwareInfo<ArchTag>;
    using DataType = int8_t;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);

    __aicore__ l1_to_l0_b(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t nTileCeil,
                          uint32_t kPartCeil,
                          uint32_t nSrcStride,
                          uint32_t kSrcStride,
                          uint32_t nDstStride,
                          uint32_t kDstStride)
    {
        for (uint32_t i = 0; i < nTileCeil / BLOCK_SIZE; i++) {
#ifdef USE_ASCENDC
            AscendC::LoadDataWithTranspose(l0Tensor[i * kPartCeil * BLOCK_SIZE],
                                           l1Tensor[i * BLOCK_SIZE * BLOCK_SIZE],
                                           AscendC::LoadData2dTransposeParams(0,                      // startIndexIn
                                                                              kPartCeil / BLOCK_SIZE, // repeatTimesIn
                                                                              nTileCeil / BLOCK_SIZE, // srcStrideIn
                                                                              1,                      // dstGapIn
                                                                              0,                      // dstfracGapIn
                                                                              0)                      // addrModeIn
            );
#else
            load_cbuf_to_cb_transpose((__cb__ DataType *)l0Tensor.GetPhyAddr() + i * kPartCeil * BLOCK_SIZE,
                                      (__cbuf__ DataType *)l1Tensor.GetPhyAddr() + i * BLOCK_SIZE * BLOCK_SIZE,
                                      0,                      // baseIdx
                                      kPartCeil / BLOCK_SIZE, // repeat
                                      nTileCeil / BLOCK_SIZE, // srcStride
                                      1,                      // dstStride
                                      0,                      // addrmode
                                      0);                     // dstFracStride
#endif
        }
    };
};

// Partial specialization for no transpose, not vector
template <ArchType ArchTag, typename DataType>
struct l1_to_l0_b<ArchTag, DataType, false, DataFormat::ZN, DataFormat::NZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);
    static constexpr uint32_t BLOCK_NUM_PER_FRACTAL = HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;

    __aicore__ l1_to_l0_b(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t nTileCeil,
                          uint32_t kPartCeil,
                          uint32_t nSrcStride,
                          uint32_t kSrcStride,
                          uint32_t nDstStride,
                          uint32_t kDstStride)
    {
        for (uint32_t i = 0; i < kPartCeil / BLOCK_NUM_PER_FRACTAL; i++) {
            AscendC::LoadData(l0Tensor[i * kDstStride * FRACTAL_SIZE],
                              l1Tensor[i * kSrcStride * FRACTAL_SIZE],
                              AscendC::LoadData2dParams(0,                                             // baseIdx
                                                        static_cast<uint16_t>(nTileCeil / BLOCK_SIZE), // repeat
                                                        nSrcStride,                                    // srcStride
                                                        0,                                             // sid
                                                        nDstStride - 1,                                // dstStride
                                                        true,                                          // transpose
                                                        0));                                           // addrCalMode
        }
    };
};

// Partial specialization for transpose, not vector
template <ArchType ArchTag, typename DataType>
struct l1_to_l0_b<ArchTag, DataType, true, DataFormat::ZN, DataFormat::NZ> {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::l1l0BlockSize / sizeof(DataType);
    static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(DataType);
    static constexpr uint32_t BLOCK_NUM_PER_FRACTAL = HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;
    __aicore__ l1_to_l0_b(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t nTileCeil,
                          uint32_t kPartCeil,
                          uint32_t nSrcStride,
                          uint32_t kSrcStride,
                          uint32_t nDstStride,
                          uint32_t kDstStride)
    {
        AscendC::LoadData(
            l0Tensor,
            l1Tensor,
            AscendC::LoadData2dParams(0,                                                           // baseIdx
                                      static_cast<uint16_t>(kPartCeil * nTileCeil / FRACTAL_SIZE), // repeat
                                      1,                                                           // srcStride
                                      0,                                                           // sid
                                      0,                                                           // dstStride
                                      false,                                                       // transpose
                                      0));                                                         // addr_cal_mode_t
    };
};