/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * MindKernelInfra is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "mki/tensor.h"
#include "mki/utils/log/log.h"

namespace Mki {
int64_t Tensor::Numel() const { return desc.Numel(); }

void Tensor::View(const Mki::SVector<int64_t> &newDims) { return desc.View(newDims); }

std::string Tensor::ToString() const
{
    std::stringstream ss;
    ss << desc.ToString() << ", dataSize:" << dataSize;
#ifdef _DEBUG
    ss << ", data:" << data;
#else
    ss << ", data:" << (data != nullptr);
#endif
    return ss.str();
}
} // namespace Mki
