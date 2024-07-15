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
#include "golden.h"

#include "mki/utils/fp16/fp16_t.h"
#include "float_util.h"

namespace Mki {
Status Golden::InOutTensorEqual(float atol, float rtol, const GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    for (int64_t i = 0; i < inTensor.Numel(); i++) {
        float expect = inTensor.desc.dtype == TENSOR_DTYPE_FLOAT16
                           ? static_cast<float>(static_cast<fp16_t *>(inTensor.data)[i])
                           : static_cast<float *>(inTensor.data)[i];
        float result = outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16
                           ? static_cast<float>(static_cast<fp16_t *>(outTensor.data)[i])
                           : static_cast<float *>(outTensor.data)[i];
        if (!FloatUtil::FloatJudgeEqual(expect, result, atol, rtol)) {
            std::string msg = "pos " + std::to_string(i) + ", expect: " + std::to_string(expect) +
                              ", result:" + std::to_string(result);
            return Status::FailStatus(-1, msg);
        }
    }
    return Status::OkStatus();
}
} // namespace Mki
