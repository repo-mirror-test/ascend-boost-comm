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
#ifndef INCLUDE_SET_FPC_H
#define INCLUDE_SET_FPC_H

#include "hardware.h"
#include "kernel_tensor.h"

/////////////////////////////////////////////////////
// SetQuantPreAddr
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType>
struct SetQuantPreAddr {
    __aicore__ SetQuantPreAddr(AscendC::LocalTensor<DataType> quantPreTensor) {};
};

template <typename DataType>
struct SetQuantPreAddr<ArchType::ASCEND_V220, DataType> {
    static constexpr uint32_t QUANT_PRE_ADDR_MASK = 0xffff;
    static constexpr uint32_t USELESS_BIT_NUM = 7;
    static constexpr uint32_t QUANT_PRE_BIT_POS_IN_FPC = 8;

    __aicore__ SetQuantPreAddr(AscendC::LocalTensor<DataType> quantPreTensor)
    {
        uint64_t quantPreAddr = (uint64_t)(__fbuf__ uint64_t *)quantPreTensor.GetPhyAddr();
        AscendC::SetFixPipeConfigImpl(quantPreTensor);
    };
};
#endif