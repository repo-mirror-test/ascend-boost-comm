/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
