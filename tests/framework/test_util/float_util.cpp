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
