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
#include "float_util.h"

#include <random>

#include "mki/utils/fp16/fp16_t.h"
#include "mki/utils/log/log.h"

bool FloatUtil::FloatJudgeEqual(float expected, float actual, float atol, float rtol)
{
    bool judge = std::abs(expected - actual) <= (atol + rtol * std::abs(actual));
    MKI_LOG_IF(!judge, ERROR) << "judge float expected: " << expected << ", actual: " << actual;
    return judge;
}

template <class T> void FloatUtil::GenerateCode(float lower, float upper, std::vector<T> &code)
{
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> distr(lower, upper);
    for (size_t i = 0; i < code.size(); i++) {
        code[i] = static_cast<T>(distr(eng));
    }
}

template void FloatUtil::GenerateCode(float lower, float upper, std::vector<Mki::fp16_t> &code);
template void FloatUtil::GenerateCode(float lower, float upper, std::vector<float> &code);
