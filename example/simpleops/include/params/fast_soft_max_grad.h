/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef SIMPLEOPS_PARAMS_FAST_SOFT_MAX_GRAD_H
#define SIMPLEOPS_PARAMS_FAST_SOFT_MAX_GRAD_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/svector/svector.h>

namespace SimpleOps {
namespace OpParam {
struct FastSoftMaxGrad {
    int32_t headNum = 0;
    std::vector<int32_t> qSeqLen;

    bool operator==(const FastSoftMaxGrad &other) const
    {
        return this->headNum == other.headNum &&
            this->qSeqLen == other.qSeqLen;
    }
};

} // namespace OpParam
} // namespace SimpleOps

#endif
